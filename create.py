import copy
import json
import os
from dataclasses import dataclass, MISSING
from datetime import datetime
from random import randint
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import wandb
from PIL import Image
from tqdm import tqdm

from commons.classes import (
    Capture,
    Camera,
    DensificationVariables,
    GaussianCloudParameters,
)
from commons.classes import (
    GaussianCloudReferenceState,
    Neighborhoods,
    Background,
)
from commons.command import Command
from commons.loss import calculate_full_loss, calculate_image_and_segmentation_loss
from deformation_network import (
    update_parameters,
    DeformationNetwork,
)
from external import densify_gaussians


@dataclass
class Create(Command):
    sequence_name: str = MISSING
    data_directory_path: str = MISSING
    experiment_id: str = datetime.utcnow().isoformat() + "Z"
    learning_rate: float = 0.01
    timestep_count_limit: Optional[int] = None
    output_directory_path: str = "./deformation_networks"

    @staticmethod
    def _compute_knn_indices_and_squared_distances(
        numpy_point_cloud: np.ndarray, k: int
    ):
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

    @staticmethod
    def _create_trainable_parameter(v):
        return torch.nn.Parameter(
            torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
        )

    @staticmethod
    def get_scene_radius(dataset_metadata):
        camera_centers = np.linalg.inv(dataset_metadata["w2c"][0])[:, :3, 3]
        scene_radius = 1.1 * np.max(
            np.linalg.norm(camera_centers - np.mean(camera_centers, 0)[None], axis=-1)
        )
        return scene_radius

    @staticmethod
    def create_optimizer(parameters: GaussianCloudParameters, scene_radius: float):
        return torch.optim.Adam(
            [
                {
                    "params": [parameters.means],
                    "name": "means",
                    "lr": 0.00016 * scene_radius,
                },
                {
                    "params": [parameters.rgb_colors],
                    "name": "rgb_colors",
                    "lr": 0.0025,
                },
                {
                    "params": [parameters.segmentation_colors],
                    "name": "segmentation_colors",
                    "lr": 0.0,
                },
                {
                    "params": [parameters.rotation_quaternions],
                    "name": "rotation_quaternions",
                    "lr": 0.001,
                },
                {
                    "params": [parameters.opacities_logits],
                    "name": "opacities_logits",
                    "lr": 0.05,
                },
                {
                    "params": [parameters.log_scales],
                    "name": "log_scales",
                    "lr": 0.001,
                },
                {
                    "params": [parameters.camera_matrices],
                    "name": "camera_matrices",
                    "lr": 1e-4,
                },
                {
                    "params": [parameters.camera_centers],
                    "name": "camera_centers",
                    "lr": 1e-4,
                },
            ],
            lr=0.0,
            eps=1e-15,
        )

    @staticmethod
    def _create_densification_variables(
        gaussian_cloud_parameters: GaussianCloudParameters,
    ):
        densification_variables = DensificationVariables(
            visibility_count=torch.zeros(gaussian_cloud_parameters.means.shape[0])
            .cuda()
            .float(),
            mean_2d_gradients_accumulated=torch.zeros(
                gaussian_cloud_parameters.means.shape[0]
            )
            .cuda()
            .float(),
            max_2d_radii=torch.zeros(gaussian_cloud_parameters.means.shape[0])
            .cuda()
            .float(),
        )
        return densification_variables

    @staticmethod
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
                        extrinsic_matrix=dataset_metadata["w2c"][timestep][
                            camera_index
                        ],
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

    @staticmethod
    def get_random_element(input_list, fallback_list):
        if not input_list:
            input_list = fallback_list.copy()
        return input_list.pop(randint(0, len(input_list) - 1))

    @staticmethod
    def initialize_post_first_timestep(
        gaussian_cloud_parameters: GaussianCloudParameters,
    ):

        foreground_mask = gaussian_cloud_parameters.segmentation_colors[:, 0] > 0.5
        foreground_means = gaussian_cloud_parameters.means[foreground_mask]
        background_means = gaussian_cloud_parameters.means[~foreground_mask]
        background_rotations = torch.nn.functional.normalize(
            gaussian_cloud_parameters.rotation_quaternions[~foreground_mask]
        )
        neighbor_indices_list, neighbor_squared_distances_list = (
            Create._compute_knn_indices_and_squared_distances(
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
            means=gaussian_cloud_parameters.means.detach(),
            rotations=torch.nn.functional.normalize(
                gaussian_cloud_parameters.rotation_quaternions
            ).detach(),
        )

        return background, neighborhoods, previous_timestep_gaussian_cloud_state

    @staticmethod
    def _get_inverted_foreground_rotations(
        rotations: torch.Tensor, foreground_mask: torch.Tensor
    ):
        foreground_rotations = rotations[foreground_mask]
        foreground_rotations[:, 1:] = -1 * foreground_rotations[:, 1:]
        return foreground_rotations

    @staticmethod
    def _update_previous_timestep_gaussian_cloud_state(
        gaussian_cloud_parameters: GaussianCloudParameters,
        previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
        neighborhood_indices: torch.Tensor,
    ):
        current_means = gaussian_cloud_parameters.means
        current_rotations = torch.nn.functional.normalize(
            gaussian_cloud_parameters.rotation_quaternions
        )

        foreground_mask = gaussian_cloud_parameters.segmentation_colors[:, 0] > 0.5
        inverted_foreground_rotations = Create._get_inverted_foreground_rotations(
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
        previous_timestep_gaussian_cloud_state.colors = (
            gaussian_cloud_parameters.rgb_colors.detach()
        )
        previous_timestep_gaussian_cloud_state.means = current_means.detach()
        previous_timestep_gaussian_cloud_state.rotations = current_rotations.detach()

    def _set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)
        self.output_directory_path = os.path.abspath(self.output_directory_path)

    def get_timestep_count(self, dataset_metadata):
        sequence_length = len(dataset_metadata["fn"])
        if self.timestep_count_limit is None:
            return sequence_length
        else:
            return min(sequence_length, self.timestep_count_limit)

    def initialize_parameters(self):
        initial_point_cloud = np.load(
            os.path.join(
                self.data_directory_path,
                self.sequence_name,
                "init_pt_cld.npz",
            )
        )["data"]
        segmentation_masks = initial_point_cloud[:, 6]
        camera_count_limit = 50
        _, squared_distances = self._compute_knn_indices_and_squared_distances(
            numpy_point_cloud=initial_point_cloud[:, :3], k=3
        )
        return GaussianCloudParameters(
            means=Create._create_trainable_parameter(initial_point_cloud[:, :3]),
            rgb_colors=Create._create_trainable_parameter(initial_point_cloud[:, 3:6]),
            segmentation_colors=Create._create_trainable_parameter(
                np.stack(
                    (
                        segmentation_masks,
                        np.zeros_like(segmentation_masks),
                        1 - segmentation_masks,
                    ),
                    -1,
                )
            ),
            rotation_quaternions=Create._create_trainable_parameter(
                np.tile([1, 0, 0, 0], (segmentation_masks.shape[0], 1))
            ),
            opacities_logits=Create._create_trainable_parameter(
                np.zeros((segmentation_masks.shape[0], 1))
            ),
            log_scales=Create._create_trainable_parameter(
                np.tile(
                    np.log(np.sqrt(squared_distances.mean(-1).clip(min=0.0000001)))[
                        ..., None
                    ],
                    (1, 3),
                )
            ),
            camera_matrices=Create._create_trainable_parameter(
                np.zeros((camera_count_limit, 3))
            ),
            camera_centers=Create._create_trainable_parameter(
                np.zeros((camera_count_limit, 3))
            ),
        )

    def _train_first_timestep(
        self,
        dataset_metadata,
        gaussian_cloud_parameters: GaussianCloudParameters,
    ):
        scene_radius = self.get_scene_radius(dataset_metadata=dataset_metadata)
        optimizer = self.create_optimizer(
            parameters=gaussian_cloud_parameters, scene_radius=scene_radius
        )
        densification_variables = self._create_densification_variables(
            gaussian_cloud_parameters
        )
        timestep = 0
        timestep_captures = self.load_timestep_captures(
            dataset_metadata=dataset_metadata,
            timestep=timestep,
            data_directory_path=self.data_directory_path,
            sequence_name=self.sequence_name,
        )
        timestep_capture_buffer = []
        for i in tqdm(range(10_000), desc=f"timestep {timestep}"):
            capture = self.get_random_element(
                input_list=timestep_capture_buffer, fallback_list=timestep_captures
            )
            loss, densification_variables = calculate_image_and_segmentation_loss(
                gaussian_cloud_parameters=gaussian_cloud_parameters,
                target_capture=capture,
                densification_variables=densification_variables,
            )
            wandb.log(
                {
                    f"timestep-{timestep}-loss": loss.item(),
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

    def get_densified_initial_gaussian_cloud_parameters(self, dataset_metadata):
        gaussian_cloud_parameters = self.initialize_parameters()
        self._train_first_timestep(dataset_metadata, gaussian_cloud_parameters)
        for parameter in gaussian_cloud_parameters.__dict__.values():
            parameter.requires_grad = False
        return gaussian_cloud_parameters

    def _save_and_log_checkpoint(
        self,
        initial_gaussian_cloud_parameters: GaussianCloudParameters,
        deformation_network: DeformationNetwork,
        timestep_count: int,
    ):
        network_directory_path = os.path.join(
            self.output_directory_path,
            self.experiment_id,
            self.sequence_name,
        )

        parameters_save_path = os.path.join(
            network_directory_path,
            "densified_gaussian_cloud_parameters.pth",
        )
        os.makedirs(os.path.dirname(parameters_save_path), exist_ok=True)
        torch.save(initial_gaussian_cloud_parameters, parameters_save_path)

        with open(
            os.path.join(
                network_directory_path,
                "metadata.json",
            ),
            "w",
        ) as file:
            json.dump(
                {
                    "timestep_count": timestep_count,
                },
                file,
                indent=4,
            )

        network_state_dict_path = os.path.join(
            network_directory_path,
            "network_state_dict.pth",
        )
        torch.save(deformation_network.state_dict(), network_state_dict_path)
        wandb.save(
            os.path.join(
                network_directory_path,
                "*",
            ),
            base_path=self.output_directory_path,
        )

    def _train_in_sequential_order(
        self,
        timestep_count,
        dataset_metadata,
        deformation_network,
        initial_gaussian_cloud_parameters: GaussianCloudParameters,
        initial_background: Background,
        initial_neighborhoods: Neighborhoods,
        previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
        optimizer,
    ):
        for timestep in range(1, timestep_count):
            timestep_capture_list = self.load_timestep_captures(
                dataset_metadata, timestep, self.data_directory_path, self.sequence_name
            )
            timestep_capture_buffer = []

            for _ in tqdm(range(10_000), desc=f"timestep {timestep}"):
                capture = self.get_random_element(
                    input_list=timestep_capture_buffer,
                    fallback_list=timestep_capture_list,
                )
                updated_gaussian_cloud_parameters = update_parameters(
                    deformation_network, initial_gaussian_cloud_parameters, timestep
                )
                loss = calculate_full_loss(
                    gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
                    target_capture=capture,
                    initial_background=initial_background,
                    initial_neighborhoods=initial_neighborhoods,
                    previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
                )
                wandb.log(
                    {
                        f"timestep-{timestep}-loss-deformation-network": loss.item(),
                    }
                )
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            timestep_capture_buffer = timestep_capture_list.copy()
            losses = []
            while timestep_capture_buffer:
                with torch.no_grad():
                    capture = timestep_capture_buffer.pop(
                        randint(0, len(timestep_capture_buffer) - 1)
                    )
                    loss = calculate_full_loss(
                        gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
                        target_capture=capture,
                        initial_background=initial_background,
                        initial_neighborhoods=initial_neighborhoods,
                        previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
                    )
                    losses.append(loss.item())
            wandb.log({f"mean-timestep-losses": sum(losses) / len(losses)})
            self._save_and_log_checkpoint(
                initial_gaussian_cloud_parameters, deformation_network, timestep_count
            )
            self._update_previous_timestep_gaussian_cloud_state(
                gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
                previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
                neighborhood_indices=initial_neighborhoods.indices,
            )

    def run(self):
        self._set_absolute_paths()
        wandb.init(project="4d-gaussian-splatting")
        dataset_metadata = json.load(
            open(
                os.path.join(
                    self.data_directory_path,
                    self.sequence_name,
                    "train_meta.json",
                ),
                "r",
            )
        )
        timestep_count = self.get_timestep_count(dataset_metadata)
        deformation_network = DeformationNetwork(timestep_count).cuda()
        optimizer = torch.optim.Adam(params=deformation_network.parameters(), lr=1e-3)

        gaussian_cloud_parameters = (
            self.get_densified_initial_gaussian_cloud_parameters(dataset_metadata)
        )

        (
            initial_background,
            initial_neighborhoods,
            previous_timestep_gaussian_cloud_state,
        ) = self.initialize_post_first_timestep(
            gaussian_cloud_parameters=gaussian_cloud_parameters
        )

        self._update_previous_timestep_gaussian_cloud_state(
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
            neighborhood_indices=initial_neighborhoods.indices,
        )
        self._train_in_sequential_order(
            timestep_count=timestep_count,
            dataset_metadata=dataset_metadata,
            deformation_network=deformation_network,
            initial_gaussian_cloud_parameters=gaussian_cloud_parameters,
            initial_background=initial_background,
            initial_neighborhoods=initial_neighborhoods,
            previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
            optimizer=optimizer,
        )


def main():
    command = Create.__new__(Create)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
