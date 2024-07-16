import copy
import os
from random import randint

import numpy as np
import open3d as o3d
import torch
import wandb
from PIL import Image
from tqdm import tqdm

from commons.classes import (
    Background,
    Capture,
    Camera,
    DensificationVariables,
    GaussianCloudParameterNames,
    GaussianCloudReferenceState,
    Neighborhoods,
)
from commons.loss import calculate_image_and_segmentation_loss
from snapshot_collection.external import densify_gaussians


def _create_densification_variables(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter]
):
    densification_variables = DensificationVariables(
        visibility_count=torch.zeros(
            gaussian_cloud_parameters[GaussianCloudParameterNames.means].shape[0]
        )
        .cuda()
        .float(),
        mean_2d_gradients_accumulated=torch.zeros(
            gaussian_cloud_parameters[GaussianCloudParameterNames.means].shape[0]
        )
        .cuda()
        .float(),
        max_2d_radii=torch.zeros(
            gaussian_cloud_parameters[GaussianCloudParameterNames.means].shape[0]
        )
        .cuda()
        .float(),
    )
    return densification_variables


def _compute_knn_indices_and_squared_distances(numpy_point_cloud: np.ndarray, k: int):
    indices_list = []
    squared_distances_list = []
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(numpy_point_cloud, np.float64)
    )
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    for point in point_cloud.points:
        # The output of search_knn_vector_3d also includes the query point itself.
        # Therefore, we need to search for k + 1 points and exclude the first element of the output
        [_, neighbor_indices, squared_distances_to_neighbors] = (
            kd_tree.search_knn_vector_3d(point, k + 1)
        )
        indices_list.append(neighbor_indices[1:])
        squared_distances_list.append(squared_distances_to_neighbors[1:])
    return np.array(indices_list), np.array(squared_distances_list)


def _get_inverted_foreground_rotations(
    rotations: torch.Tensor, foreground_mask: torch.Tensor
):
    foreground_rotations = rotations[foreground_mask]
    foreground_rotations[:, 1:] = -1 * foreground_rotations[:, 1:]
    return foreground_rotations


def get_timestep_count(timestep_count_limit, dataset_metadata):
    sequence_length = len(dataset_metadata["fn"])
    if timestep_count_limit is None:
        return sequence_length
    else:
        return min(sequence_length, timestep_count_limit)


def load_timestep_captures(
    dataset_metadata, timestep: int, data_directory_path: str, sequence_name: str
):
    timestep_data = []
    for camera_index in range(len(dataset_metadata["fn"][timestep])):
        filename = dataset_metadata["fn"][timestep][camera_index]
        segmentation_mask = (
            torch.tensor(
                np.array(
                    copy.deepcopy(
                        Image.open(
                            os.path.join(
                                data_directory_path,
                                sequence_name,
                                "seg",
                                filename.replace(".jpg", ".png"),
                            )
                        )
                    )
                ).astype(np.float32)
            )
            .float()
            .cuda()
        )
        timestep_data.append(
            Capture(
                camera=Camera(
                    id_=camera_index,
                    image_width=dataset_metadata["w"],
                    image_height=dataset_metadata["h"],
                    near_clipping_plane_distance=1,
                    far_clipping_plane_distance=100,
                    intrinsic_matrix=dataset_metadata["k"][timestep][camera_index],
                    extrinsic_matrix=dataset_metadata["w2c"][timestep][camera_index],
                ),
                image=torch.tensor(
                    np.array(
                        copy.deepcopy(
                            Image.open(
                                os.path.join(
                                    data_directory_path,
                                    sequence_name,
                                    "ims",
                                    filename,
                                )
                            )
                        )
                    )
                )
                .float()
                .cuda()
                .permute(2, 0, 1)
                / 255,
                segmentation_mask=torch.stack(
                    (
                        segmentation_mask,
                        torch.zeros_like(segmentation_mask),
                        1 - segmentation_mask,
                    )
                ),
            )
        )
    return timestep_data


def get_random_element(input_list, fallback_list):
    if not input_list:
        input_list = fallback_list.copy()
    return input_list.pop(randint(0, len(input_list) - 1))


def initialize_post_first_timestep(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    optimizer: torch.optim.Adam,
):

    foreground_mask = (
        gaussian_cloud_parameters[GaussianCloudParameterNames.segmentation_masks][:, 0]
        > 0.5
    )
    foreground_means = gaussian_cloud_parameters[GaussianCloudParameterNames.means][
        foreground_mask
    ]
    background_means = gaussian_cloud_parameters[GaussianCloudParameterNames.means][
        ~foreground_mask
    ]
    background_rotations = torch.nn.functional.normalize(
        gaussian_cloud_parameters[GaussianCloudParameterNames.rotation_quaternions][
            ~foreground_mask
        ]
    )
    neighbor_indices_list, neighbor_squared_distances_list = (
        _compute_knn_indices_and_squared_distances(
            foreground_means.detach().cpu().numpy(), 20
        )
    )
    neighborhoods = Neighborhoods(
        indices=(torch.tensor(neighbor_indices_list).cuda().long().contiguous()),
        weights=(
            torch.tensor(np.exp(-2000 * neighbor_squared_distances_list))
            .cuda()
            .float()
            .contiguous()
        ),
        distances=(
            torch.tensor(np.sqrt(neighbor_squared_distances_list))
            .cuda()
            .float()
            .contiguous()
        ),
    )
    background = Background(
        means=background_means.detach(),
        rotations=background_rotations.detach(),
    )

    previous_timestep_gaussian_cloud_state = GaussianCloudReferenceState(
        means=gaussian_cloud_parameters[GaussianCloudParameterNames.means].detach(),
        rotations=torch.nn.functional.normalize(
            gaussian_cloud_parameters[GaussianCloudParameterNames.rotation_quaternions]
        ).detach(),
    )

    parameters_to_fix = [
        GaussianCloudParameterNames.opacities_logits,
        GaussianCloudParameterNames.log_scales,
        GaussianCloudParameterNames.camera_matrices,
        GaussianCloudParameterNames.camera_centers,
    ]
    for parameter_group in optimizer.param_groups:
        if parameter_group["name"] in parameters_to_fix:
            parameter_group["lr"] = 0.0
    return background, neighborhoods, previous_timestep_gaussian_cloud_state


def initialize_parameters(data_directory_path: str, sequence_name: str):
    initial_point_cloud = np.load(
        os.path.join(
            data_directory_path,
            sequence_name,
            "init_pt_cld.npz",
        )
    )["data"]
    segmentation_masks = initial_point_cloud[:, 6]
    camera_count_limit = 50
    _, squared_distances = _compute_knn_indices_and_squared_distances(
        numpy_point_cloud=initial_point_cloud[:, :3], k=3
    )
    return {
        k: torch.nn.Parameter(
            torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
        )
        for k, v in {
            GaussianCloudParameterNames.means: initial_point_cloud[:, :3],
            GaussianCloudParameterNames.colors: initial_point_cloud[:, 3:6],
            GaussianCloudParameterNames.segmentation_masks: np.stack(
                (
                    segmentation_masks,
                    np.zeros_like(segmentation_masks),
                    1 - segmentation_masks,
                ),
                -1,
            ),
            GaussianCloudParameterNames.rotation_quaternions: np.tile(
                [1, 0, 0, 0], (segmentation_masks.shape[0], 1)
            ),
            GaussianCloudParameterNames.opacities_logits: np.zeros(
                (segmentation_masks.shape[0], 1)
            ),
            GaussianCloudParameterNames.log_scales: np.tile(
                np.log(np.sqrt(squared_distances.mean(-1).clip(min=0.0000001)))[
                    ..., None
                ],
                (1, 3),
            ),
            GaussianCloudParameterNames.camera_matrices: np.zeros(
                (camera_count_limit, 3)
            ),
            GaussianCloudParameterNames.camera_centers: np.zeros(
                (camera_count_limit, 3)
            ),
        }.items()
    }


def convert_parameters_to_numpy(
    parameters: dict[str, torch.nn.Parameter], include_dynamic_parameters_only: bool
):
    if include_dynamic_parameters_only:
        return {
            k: v.detach().cpu().contiguous().numpy()
            for k, v in parameters.items()
            if k
            in [
                GaussianCloudParameterNames.means,
                GaussianCloudParameterNames.colors,
                GaussianCloudParameterNames.rotation_quaternions,
            ]
        }
    else:
        return {k: v.detach().cpu().contiguous().numpy() for k, v in parameters.items()}


def create_optimizer(parameters: dict[str, torch.nn.Parameter], scene_radius: float):
    learning_rates = {
        GaussianCloudParameterNames.means: 0.00016 * scene_radius,
        GaussianCloudParameterNames.colors: 0.0025,
        GaussianCloudParameterNames.segmentation_masks: 0.0,
        GaussianCloudParameterNames.rotation_quaternions: 0.001,
        GaussianCloudParameterNames.opacities_logits: 0.05,
        GaussianCloudParameterNames.log_scales: 0.001,
        GaussianCloudParameterNames.camera_matrices: 1e-4,
        GaussianCloudParameterNames.camera_centers: 1e-4,
    }
    return torch.optim.Adam(
        [
            {"params": [value], "name": name, "lr": learning_rates[name]}
            for name, value in parameters.items()
        ],
        lr=0.0,
        eps=1e-15,
    )


def get_scene_radius(dataset_metadata):
    camera_centers = np.linalg.inv(dataset_metadata["w2c"][0])[:, :3, 3]
    scene_radius = 1.1 * np.max(
        np.linalg.norm(camera_centers - np.mean(camera_centers, 0)[None], axis=-1)
    )
    return scene_radius


def update_previous_timestep_gaussian_cloud_state(
    current_means,
    current_rotations,
    gaussian_cloud_parameters,
    neighborhood_indices,
    previous_timestep_gaussian_cloud_state,
):
    foreground_mask = (
        gaussian_cloud_parameters[GaussianCloudParameterNames.segmentation_masks][:, 0]
        > 0.5
    )
    inverted_foreground_rotations = _get_inverted_foreground_rotations(
        current_rotations, foreground_mask
    )
    foreground_means = current_means[foreground_mask]
    offsets_to_neighbors = (
        foreground_means[neighborhood_indices] - foreground_means[:, None]
    )
    previous_timestep_gaussian_cloud_state.inverted_foreground_rotations = (
        inverted_foreground_rotations.detach()
    )
    previous_timestep_gaussian_cloud_state.offsets_to_neighbors = (
        offsets_to_neighbors.detach()
    )
    previous_timestep_gaussian_cloud_state.colors = gaussian_cloud_parameters[
        GaussianCloudParameterNames.colors
    ].detach()
    previous_timestep_gaussian_cloud_state.means = current_means.detach()
    previous_timestep_gaussian_cloud_state.rotations = current_rotations.detach()


def train_first_timestep(
    data_directory_path,
    sequence_name,
    dataset_metadata,
    gaussian_cloud_parameters,
    optimizer,
    scene_radius,
    method: str,
):
    densification_variables = _create_densification_variables(gaussian_cloud_parameters)
    timestep = 0
    timestep_captures = load_timestep_captures(
        dataset_metadata=dataset_metadata,
        timestep=timestep,
        data_directory_path=data_directory_path,
        sequence_name=sequence_name,
    )
    timestep_capture_buffer = []
    for i in tqdm(range(10_000), desc=f"timestep {timestep}"):
        capture = get_random_element(
            input_list=timestep_capture_buffer, fallback_list=timestep_captures
        )
        loss, densification_variables = calculate_image_and_segmentation_loss(
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            target_capture=capture,
            densification_variables=densification_variables,
        )
        wandb.log(
            {
                f"timestep-{timestep}-loss-{method}": loss.item(),
            }
        )
        loss.backward()
        with torch.no_grad():
            densify_gaussians(
                gaussian_cloud_parameters=gaussian_cloud_parameters,
                densification_variables=densification_variables,
                scene_radius=scene_radius,
                optimizer=optimizer,
                i=i,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
