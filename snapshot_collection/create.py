import json
import os
from dataclasses import dataclass, MISSING
from datetime import datetime
from typing import Optional

import numpy as np
import open3d as o3d
import torch
import wandb
from tqdm import tqdm

from commons.command import Command
from commons.helpers import (
    l1_loss_v1,
    l1_loss_v2,
    weighted_l2_loss_v1,
    weighted_l2_loss_v2,
    quat_mult,
    GaussianCloudParameterNames,
)
from commons.training_commons import (
    Capture,
    load_timestep_captures,
    get_random_element,
    get_timestep_count,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from snapshot_collection.external import (
    build_rotation,
    calc_ssim,
    densify_gaussians,
    update_params_and_optimizer,
    DensificationVariables,
)


class GaussianCloud:
    def __init__(self, parameters: dict[str, torch.nn.Parameter]):
        self.means_2d = (
            torch.zeros_like(
                parameters[GaussianCloudParameterNames.means],
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        self.means_3d = parameters[GaussianCloudParameterNames.means]
        self.colors = parameters[GaussianCloudParameterNames.colors]
        self.rotations = torch.nn.functional.normalize(
            parameters[GaussianCloudParameterNames.rotation_quaternions]
        )
        self.opacities = torch.sigmoid(
            parameters[GaussianCloudParameterNames.opacities_logits]
        )
        self.scales = torch.exp(parameters[GaussianCloudParameterNames.log_scales])

    def get_renderer_format(self):
        return {
            "means3D": self.means_3d,
            "colors_precomp": self.colors,
            "rotations": self.rotations,
            "opacities": self.opacities,
            "scales": self.scales,
            "means2D": self.means_2d,
        }


class Neighborhoods:
    def __init__(
        self, distances: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ):
        self.distances: torch.Tensor = distances
        self.weights: torch.Tensor = weights
        self.indices: torch.Tensor = indices


class Background:
    def __init__(self, means: torch.Tensor, rotations: torch.Tensor):
        self.means: torch.Tensor = means
        self.rotations: torch.Tensor = rotations


class GaussianCloudReferenceState:
    def __init__(self, means: torch.Tensor, rotations: torch.Tensor):
        self.means: torch.Tensor = means
        self.rotations: torch.Tensor = rotations
        self.inverted_foreground_rotations: Optional[torch.Tensor] = None
        self.offsets_to_neighbors: Optional[torch.Tensor] = None
        self.colors: Optional[torch.Tensor] = None


@dataclass
class Create(Command):
    sequence_name: str = MISSING
    data_directory_path: str = MISSING
    output_directory_path: str = "./output/"
    experiment_id: str = datetime.utcnow().isoformat() + "Z"
    timestep_count_limit: Optional[int] = None

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
    def _get_scene_radius(dataset_metadata):
        camera_centers = np.linalg.inv(dataset_metadata["w2c"][0])[:, :3, 3]
        scene_radius = 1.1 * np.max(
            np.linalg.norm(camera_centers - np.mean(camera_centers, 0)[None], axis=-1)
        )
        return scene_radius

    @staticmethod
    def _create_optimizer(
        parameters: dict[str, torch.nn.Parameter], scene_radius: float
    ):
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

    @staticmethod
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

    @staticmethod
    def _apply_exponential_transform_and_center_to_image(
        image: torch.Tensor,
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        camera_id: int,
    ):
        camera_matrices = gaussian_cloud_parameters[
            GaussianCloudParameterNames.camera_matrices
        ][camera_id]

        # Apply exponential transformation to the camera matrix parameters.
        # The exponential function ensures all values are positive and scales
        # the parameters non-linearly, which may be required for the transformation.
        exponential_camera_matrices = torch.exp(camera_matrices)[:, None, None]

        # Element-wise multiply the transformed camera matrices with the image.
        # This step applies the transformation to each pixel.
        scaled_image = exponential_camera_matrices * image

        camera_centers = gaussian_cloud_parameters[
            GaussianCloudParameterNames.camera_centers
        ][camera_id]

        # Add the camera center parameters to the scaled image.
        # This re-centers the transformed image based on the camera's center coordinates,
        # completing the adjustment.
        adjusted_image = scaled_image + camera_centers[:, None, None]

        return adjusted_image

    @staticmethod
    def _calculate_image_loss(
        rendered_image,
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        target_capture: Capture,
    ):
        image = Create._apply_exponential_transform_and_center_to_image(
            rendered_image, gaussian_cloud_parameters, target_capture.camera.id_
        )
        return 0.8 * l1_loss_v1(image, target_capture.image) + 0.2 * (
            1.0 - calc_ssim(image, target_capture.image)
        )

    @staticmethod
    def _add_image_loss(
        losses: dict[str, torch.Tensor],
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        target_capture: Capture,
    ):
        gaussian_cloud = GaussianCloud(parameters=gaussian_cloud_parameters)
        gaussian_cloud.means_2d.retain_grad()
        (
            rendered_image,
            radii,
            _,
        ) = Renderer(
            raster_settings=target_capture.camera.gaussian_rasterization_settings
        )(**gaussian_cloud.get_renderer_format())
        losses["im"] = Create._calculate_image_loss(
            rendered_image=rendered_image,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            target_capture=target_capture,
        )
        means_2d = (
            gaussian_cloud.means_2d
        )  # Gradient only accum from colour render for densification
        return means_2d, radii

    @staticmethod
    def _add_segmentation_loss(
        losses: dict[str, torch.Tensor],
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        target_capture: Capture,
    ):
        gaussian_cloud = GaussianCloud(parameters=gaussian_cloud_parameters)
        gaussian_cloud.colors = gaussian_cloud_parameters[
            GaussianCloudParameterNames.segmentation_masks
        ]
        (
            segmentation_mask,
            _,
            _,
        ) = Renderer(
            raster_settings=target_capture.camera.gaussian_rasterization_settings
        )(**gaussian_cloud.get_renderer_format())
        losses["seg"] = 0.8 * l1_loss_v1(
            segmentation_mask, target_capture.segmentation_mask
        ) + 0.2 * (1.0 - calc_ssim(segmentation_mask, target_capture.segmentation_mask))
        return gaussian_cloud

    @staticmethod
    def _update_max_2d_radii_and_visibility_mask(
        radii, densification_variables: DensificationVariables
    ):
        radius_is_positive = radii > 0
        densification_variables.max_2d_radii[radius_is_positive] = torch.max(
            radii[radius_is_positive],
            densification_variables.max_2d_radii[radius_is_positive],
        )
        densification_variables.gaussian_is_visible_mask = radius_is_positive

    @staticmethod
    def _combine_losses(losses: dict[str, torch.Tensor]):
        loss_weights = {
            "im": 1.0,
            "seg": 3.0,
            "rigid": 4.0,
            "rot": 4.0,
            "iso": 2.0,
            "floor": 2.0,
            "bg": 20.0,
            "soft_col_cons": 0.01,
        }
        return torch.sum(
            torch.stack(
                ([torch.tensor(loss_weights[k]) * v for k, v in losses.items()])
            )
        )

    @staticmethod
    def _calculate_image_and_segmentation_loss(
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        target_capture: Capture,
        densification_variables: DensificationVariables,
    ):
        losses = {}
        means_2d, radii = Create._add_image_loss(
            losses=losses,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            target_capture=target_capture,
        )
        _ = Create._add_segmentation_loss(
            losses=losses,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            target_capture=target_capture,
        )
        densification_variables.means_2d = means_2d
        Create._update_max_2d_radii_and_visibility_mask(
            radii=radii, densification_variables=densification_variables
        )
        return Create._combine_losses(losses), densification_variables

    @staticmethod
    def _initialize_post_first_timestep(
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        optimizer: torch.optim.Adam,
    ):

        foreground_mask = (
            gaussian_cloud_parameters[GaussianCloudParameterNames.segmentation_masks][
                :, 0
            ]
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
            means=gaussian_cloud_parameters[GaussianCloudParameterNames.means].detach(),
            rotations=torch.nn.functional.normalize(
                gaussian_cloud_parameters[
                    GaussianCloudParameterNames.rotation_quaternions
                ]
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

    @staticmethod
    def _convert_parameters_to_numpy(
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
            return {
                k: v.detach().cpu().contiguous().numpy() for k, v in parameters.items()
            }

    @staticmethod
    def _get_inverted_foreground_rotations(
        rotations: torch.Tensor, foreground_mask: torch.Tensor
    ):
        foreground_rotations = rotations[foreground_mask]
        foreground_rotations[:, 1:] = -1 * foreground_rotations[:, 1:]
        return foreground_rotations

    @staticmethod
    def _update_variables_and_optimizer_and_gaussian_cloud_parameters(
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
        neighborhood_indices: torch.Tensor,
        optimizer: torch.optim.Adam,
    ):
        current_means = gaussian_cloud_parameters[GaussianCloudParameterNames.means]
        updated_means = current_means + (
            current_means - previous_timestep_gaussian_cloud_state.means
        )
        current_rotations = torch.nn.functional.normalize(
            gaussian_cloud_parameters[GaussianCloudParameterNames.rotation_quaternions]
        )
        updated_rotations = torch.nn.functional.normalize(
            current_rotations
            + (current_rotations - previous_timestep_gaussian_cloud_state.rotations)
        )

        foreground_mask = (
            gaussian_cloud_parameters[GaussianCloudParameterNames.segmentation_masks][
                :, 0
            ]
            > 0.5
        )
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
        previous_timestep_gaussian_cloud_state.colors = gaussian_cloud_parameters[
            GaussianCloudParameterNames.colors
        ].detach()
        previous_timestep_gaussian_cloud_state.means = current_means.detach()
        previous_timestep_gaussian_cloud_state.rotations = current_rotations.detach()

        updated_parameters = {
            GaussianCloudParameterNames.means: updated_means,
            GaussianCloudParameterNames.rotation_quaternions: updated_rotations,
        }
        update_params_and_optimizer(
            updated_parameters, gaussian_cloud_parameters, optimizer
        )

    @staticmethod
    def _calculate_rigidity_loss(
        relative_rotation_quaternion: torch.Tensor,
        foreground_offset_to_neighbors: torch.Tensor,
        previous_offsets_to_neighbors: torch.Tensor,
        neighborhood_weights: torch.Tensor,
    ):
        rotation_matrix = build_rotation(relative_rotation_quaternion)
        curr_offset_in_prev_coord = (
            rotation_matrix.transpose(2, 1)[:, None]
            @ foreground_offset_to_neighbors[:, :, :, None]
        ).squeeze(-1)
        return weighted_l2_loss_v2(
            curr_offset_in_prev_coord,
            previous_offsets_to_neighbors,
            neighborhood_weights,
        )

    @staticmethod
    def _calculate_isometry_loss(
        foreground_offset_to_neighbors: torch.Tensor,
        initial_neighborhoods: Neighborhoods,
    ):
        curr_offset_mag = torch.sqrt(
            (foreground_offset_to_neighbors**2).sum(-1) + 1e-20
        )
        return weighted_l2_loss_v1(
            curr_offset_mag,
            initial_neighborhoods.distances,
            initial_neighborhoods.weights,
        )

    @staticmethod
    def _calculate_background_loss(
        gaussian_cloud: GaussianCloud,
        is_foreground: torch.Tensor,
        initial_background: Background,
    ):
        bg_pts = gaussian_cloud.means_3d[~is_foreground]
        bg_rot = gaussian_cloud.rotations[~is_foreground]
        return l1_loss_v2(bg_pts, initial_background.means) + l1_loss_v2(
            bg_rot, initial_background.rotations
        )

    @staticmethod
    def _add_additional_losses(
        losses: dict[str, torch.Tensor],
        segmentation_mask_gaussian_cloud: GaussianCloud,
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        initial_background: Background,
        initial_neighborhoods: Neighborhoods,
        previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
    ):
        foreground_mask = (
            gaussian_cloud_parameters[GaussianCloudParameterNames.segmentation_masks][
                :, 0
            ]
            > 0.5
        ).detach()
        foreground_means = segmentation_mask_gaussian_cloud.means_3d[foreground_mask]
        foreground_rotations = segmentation_mask_gaussian_cloud.rotations[
            foreground_mask
        ]
        relative_rotation_quaternion = quat_mult(
            foreground_rotations,
            previous_timestep_gaussian_cloud_state.inverted_foreground_rotations,
        )
        foreground_neighbor_means = foreground_means[initial_neighborhoods.indices]
        foreground_offset_to_neighbors = (
            foreground_neighbor_means - foreground_means[:, None]
        )
        losses["rigid"] = Create._calculate_rigidity_loss(
            relative_rotation_quaternion=relative_rotation_quaternion,
            foreground_offset_to_neighbors=foreground_offset_to_neighbors,
            previous_offsets_to_neighbors=previous_timestep_gaussian_cloud_state.offsets_to_neighbors,
            neighborhood_weights=initial_neighborhoods.weights,
        )
        losses["rot"] = weighted_l2_loss_v2(
            relative_rotation_quaternion[initial_neighborhoods.indices],
            relative_rotation_quaternion[:, None],
            initial_neighborhoods.weights,
        )
        losses["iso"] = Create._calculate_isometry_loss(
            foreground_offset_to_neighbors=foreground_offset_to_neighbors,
            initial_neighborhoods=initial_neighborhoods,
        )
        losses["floor"] = torch.clamp(foreground_means[:, 1], min=0).mean()
        losses["bg"] = Create._calculate_background_loss(
            gaussian_cloud=segmentation_mask_gaussian_cloud,
            is_foreground=foreground_mask,
            initial_background=initial_background,
        )
        losses["soft_col_cons"] = l1_loss_v2(
            gaussian_cloud_parameters[GaussianCloudParameterNames.colors],
            previous_timestep_gaussian_cloud_state.colors,
        )

    @staticmethod
    def _calculate_full_loss(
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        target_capture: Capture,
        initial_background: Background,
        initial_neighborhoods: Neighborhoods,
        previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
    ):
        losses: dict[str, torch.Tensor] = {}
        Create._add_image_loss(
            losses=losses,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            target_capture=target_capture,
        )
        segmentation_mask_gaussian_cloud = Create._add_segmentation_loss(
            losses=losses,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            target_capture=target_capture,
        )
        Create._add_additional_losses(
            losses=losses,
            segmentation_mask_gaussian_cloud=segmentation_mask_gaussian_cloud,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            initial_background=initial_background,
            initial_neighborhoods=initial_neighborhoods,
            previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
        )

        return Create._combine_losses(losses)

    def _set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)
        self.output_directory_path = os.path.abspath(self.output_directory_path)

    def _initialize_parameters(self):
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

    def _train_first_timestep(
        self,
        dataset_metadata,
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        optimizer: torch.optim.Adam,
        scene_radius: float,
    ):
        densification_variables = self._create_densification_variables(
            gaussian_cloud_parameters
        )
        timestep = 0
        timestep_captures = load_timestep_captures(
            dataset_metadata=dataset_metadata,
            timestep=timestep,
            data_directory_path=self.data_directory_path,
            sequence_name=self.sequence_name,
        )
        timestep_capture_buffer = []
        for i in tqdm(range(10_000), desc=f"timestep {timestep}"):
            capture = get_random_element(
                input_list=timestep_capture_buffer, fallback_list=timestep_captures
            )
            loss, densification_variables = self._calculate_image_and_segmentation_loss(
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
        return self._initialize_post_first_timestep(
            gaussian_cloud_parameters=gaussian_cloud_parameters, optimizer=optimizer
        ), self._convert_parameters_to_numpy(
            parameters=gaussian_cloud_parameters,
            include_dynamic_parameters_only=False,
        )

    def _train_timestep(
        self,
        timestep: int,
        dataset_metadata,
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        initial_background: Background,
        initial_neighborhoods: Neighborhoods,
        previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
        optimizer: torch.optim.Adam,
    ):
        timestep_captures = load_timestep_captures(
            dataset_metadata=dataset_metadata,
            timestep=timestep,
            data_directory_path=self.data_directory_path,
            sequence_name=self.sequence_name,
        )
        timestep_capture_buffer = []
        self._update_variables_and_optimizer_and_gaussian_cloud_parameters(
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
            neighborhood_indices=initial_neighborhoods.indices,
            optimizer=optimizer,
        )
        iteration_range = range(2000)
        for _ in tqdm(iteration_range, desc=f"timestep {timestep}"):
            capture = get_random_element(
                input_list=timestep_capture_buffer, fallback_list=timestep_captures
            )
            loss = self._calculate_full_loss(
                gaussian_cloud_parameters=gaussian_cloud_parameters,
                target_capture=capture,
                initial_background=initial_background,
                initial_neighborhoods=initial_neighborhoods,
                previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
            )
            wandb.log(
                {
                    f"timestep-{timestep}-loss": loss.item(),
                }
            )
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return self._convert_parameters_to_numpy(
            parameters=gaussian_cloud_parameters,
            include_dynamic_parameters_only=True,
        )

    def _save_sequence(
        self, gaussian_cloud_parameters_sequence: list[dict[str, torch.nn.Parameter]]
    ):
        to_save = {}
        for k in gaussian_cloud_parameters_sequence[0].keys():
            if k in gaussian_cloud_parameters_sequence[1].keys():
                to_save[k] = np.stack(
                    [parameters[k] for parameters in gaussian_cloud_parameters_sequence]
                )
            else:
                to_save[k] = gaussian_cloud_parameters_sequence[0][k]

        parameters_save_path = os.path.join(
            self.output_directory_path,
            self.experiment_id,
            self.sequence_name,
            "params.npz",
        )
        os.makedirs(os.path.dirname(parameters_save_path), exist_ok=True)
        print(f"Saving parameters at path: {parameters_save_path}")
        np.savez(parameters_save_path, **to_save)

    def run(self):
        self._set_absolute_paths()
        torch.cuda.empty_cache()
        gaussian_cloud_parameters = self._initialize_parameters()
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
        scene_radius = self._get_scene_radius(dataset_metadata)
        optimizer = self._create_optimizer(
            parameters=gaussian_cloud_parameters, scene_radius=scene_radius
        )
        gaussian_cloud_parameters_sequence = []
        timestep_count = get_timestep_count(
            timestep_count_limit=self.timestep_count_limit,
            dataset_metadata=dataset_metadata,
        )
        (
            initial_background,
            initial_neighborhoods,
            previous,
        ), new_gaussian_cloud_parameters = self._train_first_timestep(
            dataset_metadata=dataset_metadata,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            optimizer=optimizer,
            scene_radius=scene_radius,
        )
        gaussian_cloud_parameters_sequence.append(new_gaussian_cloud_parameters)
        for timestep in range(1, timestep_count):
            new_gaussian_cloud_parameters = self._train_timestep(
                timestep=timestep,
                dataset_metadata=dataset_metadata,
                gaussian_cloud_parameters=gaussian_cloud_parameters,
                initial_background=initial_background,
                initial_neighborhoods=initial_neighborhoods,
                previous_timestep_gaussian_cloud_state=previous,
                optimizer=optimizer,
            )
            gaussian_cloud_parameters_sequence.append(new_gaussian_cloud_parameters)
            self._save_sequence(gaussian_cloud_parameters_sequence)


def main():
    command = Create.__new__(Create)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
