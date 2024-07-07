import json
import os
from dataclasses import dataclass, MISSING
from datetime import datetime
from typing import Optional

import numpy as np
import open3d as o3d
import torch
from tqdm import tqdm

import external
from command import Command
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import (
    l1_loss_v1,
    l1_loss_v2,
    weighted_l2_loss_v1,
    weighted_l2_loss_v2,
    quat_mult,
    GaussianCloudParameterNames,
    Variables,
)
from training_commons import Capture, load_timestep_captures, get_random_element, get_timestep_count


@dataclass
class Trainer(Command):
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
    def _create_optimizer(parameters, scene_radius: float):
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
    def _get_inverted_foreground_rotations(
        rotations: torch.Tensor, foreground_mask: torch.Tensor
    ):
        foreground_rotations = rotations[foreground_mask]
        foreground_rotations[:, 1:] = -1 * foreground_rotations[:, 1:]
        return foreground_rotations

    @staticmethod
    def _update_variables_and_optimizer_and_gaussian_cloud_parameters(
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        variables: Variables,
        optimizer: torch.optim.Adam,
    ):
        current_means = gaussian_cloud_parameters[GaussianCloudParameterNames.means]
        updated_means = current_means + (current_means - variables.previous_means)
        current_rotations = torch.nn.functional.normalize(
            gaussian_cloud_parameters[GaussianCloudParameterNames.rotation_quaternions]
        )
        updated_rotations = torch.nn.functional.normalize(
            current_rotations + (current_rotations - variables.previous_rotations)
        )

        foreground_mask = (
            gaussian_cloud_parameters[GaussianCloudParameterNames.segmentation_masks][
                :, 0
            ]
            > 0.5
        )
        inverted_foreground_rotations = Trainer._get_inverted_foreground_rotations(
            current_rotations, foreground_mask
        )
        foreground_means = current_means[foreground_mask]
        offsets_to_neighbors = (
            foreground_means[variables.neighbor_indices_list]
            - foreground_means[:, None]
        )
        variables.previous_inverted_foreground_rotations = (
            inverted_foreground_rotations.detach()
        )
        variables.previous_offsets_to_neighbors = offsets_to_neighbors.detach()
        variables.previous_colors = gaussian_cloud_parameters[
            GaussianCloudParameterNames.colors
        ].detach()
        variables.previous_means = current_means.detach()
        variables.previous_rotations = current_rotations.detach()

        updated_parameters = {
            GaussianCloudParameterNames.means: updated_means,
            GaussianCloudParameterNames.rotation_quaternions: updated_rotations,
        }
        external.update_params_and_optimizer(
            updated_parameters, gaussian_cloud_parameters, optimizer
        )

    @staticmethod
    def _create_gaussian_cloud(parameters: dict[str, torch.nn.Parameter]):
        return {
            "means3D": parameters[GaussianCloudParameterNames.means],
            "colors_precomp": parameters[GaussianCloudParameterNames.colors],
            "rotations": torch.nn.functional.normalize(
                parameters[GaussianCloudParameterNames.rotation_quaternions]
            ),
            "opacities": torch.sigmoid(
                parameters[GaussianCloudParameterNames.opacities_logits]
            ),
            "scales": torch.exp(parameters[GaussianCloudParameterNames.log_scales]),
            "means2D": torch.zeros_like(
                parameters[GaussianCloudParameterNames.means],
                requires_grad=True,
                device="cuda",
            )
            + 0,
        }

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
        image = Trainer._apply_exponential_transform_and_center_to_image(
            rendered_image, gaussian_cloud_parameters, target_capture.camera.id_
        )
        return 0.8 * l1_loss_v1(image, target_capture.image) + 0.2 * (
            1.0 - external.calc_ssim(image, target_capture.image)
        )

    @staticmethod
    def _calculate_rigidity_loss(
        relative_rotation_quaternion,
        foreground_offset_to_neighbors,
        variables: Variables,
    ):
        rotation_matrix = external.build_rotation(relative_rotation_quaternion)
        curr_offset_in_prev_coord = (
            rotation_matrix.transpose(2, 1)[:, None]
            @ foreground_offset_to_neighbors[:, :, :, None]
        ).squeeze(-1)
        return weighted_l2_loss_v2(
            curr_offset_in_prev_coord,
            variables.previous_offsets_to_neighbors,
            variables.neighbor_weight,
        )

    @staticmethod
    def _calculate_isometry_loss(foreground_offset_to_neighbors, variables: Variables):
        curr_offset_mag = torch.sqrt(
            (foreground_offset_to_neighbors**2).sum(-1) + 1e-20
        )
        return weighted_l2_loss_v1(
            curr_offset_mag,
            variables.neighbor_distances_list,
            variables.neighbor_weight,
        )

    @staticmethod
    def _calculate_background_loss(gaussian_cloud, is_foreground, variables: Variables):
        bg_pts = gaussian_cloud["means3D"][~is_foreground]
        bg_rot = gaussian_cloud["rotations"][~is_foreground]
        return l1_loss_v2(bg_pts, variables.initial_background_means) + l1_loss_v2(
            bg_rot, variables.initial_background_rotations
        )

    @staticmethod
    def _combine_losses(losses):
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
    def _calculate_loss(
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        target_capture: Capture,
        variables: Variables,
        calculate_only_subset_of_losses: bool,
    ):
        losses = {}

        gaussian_cloud = Trainer._create_gaussian_cloud(gaussian_cloud_parameters)
        gaussian_cloud["means2D"].retain_grad()
        (
            rendered_image,
            radii,
            _,
        ) = Renderer(
            raster_settings=target_capture.camera.gaussian_rasterization_settings
        )(**gaussian_cloud)
        losses["im"] = Trainer._calculate_image_loss(
            rendered_image, gaussian_cloud_parameters, target_capture
        )
        variables.external_means2D = gaussian_cloud[
            "means2D"
        ]  # Gradient only accum from colour render for densification

        gaussian_cloud = Trainer._create_gaussian_cloud(gaussian_cloud_parameters)
        gaussian_cloud["colors_precomp"] = gaussian_cloud_parameters[
            GaussianCloudParameterNames.segmentation_masks
        ]
        (
            segmentation_mask,
            _,
            _,
        ) = Renderer(
            raster_settings=target_capture.camera.gaussian_rasterization_settings
        )(**gaussian_cloud)
        losses["seg"] = 0.8 * l1_loss_v1(
            segmentation_mask, target_capture.segmentation_mask
        ) + 0.2 * (
            1.0
            - external.calc_ssim(segmentation_mask, target_capture.segmentation_mask)
        )

        if not calculate_only_subset_of_losses:
            foreground_mask = (
                gaussian_cloud_parameters[
                    GaussianCloudParameterNames.segmentation_masks
                ][:, 0]
                > 0.5
            ).detach()
            foreground_means = gaussian_cloud["means3D"][foreground_mask]
            foreground_rotations = gaussian_cloud["rotations"][foreground_mask]

            relative_rotation_quaternion = quat_mult(
                foreground_rotations, variables.previous_inverted_foreground_rotations
            )
            foreground_neighbor_means = foreground_means[
                variables.neighbor_indices_list
            ]
            foreground_offset_to_neighbors = (
                foreground_neighbor_means - foreground_means[:, None]
            )
            losses["rigid"] = Trainer._calculate_rigidity_loss(
                relative_rotation_quaternion, foreground_offset_to_neighbors, variables
            )

            losses["rot"] = weighted_l2_loss_v2(
                relative_rotation_quaternion[variables.neighbor_indices_list],
                relative_rotation_quaternion[:, None],
                variables.neighbor_weight,
            )
            losses["iso"] = Trainer._calculate_isometry_loss(
                foreground_offset_to_neighbors, variables
            )
            losses["floor"] = torch.clamp(foreground_means[:, 1], min=0).mean()
            losses["bg"] = Trainer._calculate_background_loss(
                gaussian_cloud, foreground_mask, variables
            )
            losses["soft_col_cons"] = l1_loss_v2(
                gaussian_cloud_parameters[GaussianCloudParameterNames.colors],
                variables.previous_colors,
            )

        seen = radii > 0
        variables.external_max_2D_radius[seen] = torch.max(
            radii[seen], variables.external_max_2D_radius[seen]
        )
        variables.external_seen = seen
        return Trainer._combine_losses(losses)

    @staticmethod
    def _report_progress(
        params, dataset_element: Capture, progress_bar: tqdm, report_interval
    ):
        (
            rendered_image,
            _,
            _,
        ) = Renderer(
            raster_settings=dataset_element.camera.gaussian_rasterization_settings
        )(**Trainer._create_gaussian_cloud(params))
        camera_id = dataset_element.camera.id_
        image = Trainer._apply_exponential_transform_and_center_to_image(
            rendered_image, params, camera_id
        )
        progress_bar.set_postfix(
            {
                "train img 0 PSNR": f"{external.calc_psnr(image, dataset_element.image).mean() :.{7}f}"
            }
        )
        progress_bar.update(report_interval)

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
    def _initialize_post_first_timestep(
        gaussian_cloud_parameters, variables: Variables, optimizer
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
            Trainer._compute_knn_indices_and_squared_distances(
                foreground_means.detach().cpu().numpy(), 20
            )
        )
        variables.neighbor_indices_list = (
            torch.tensor(neighbor_indices_list).cuda().long().contiguous()
        )
        variables.neighbor_weight = (
            torch.tensor(np.exp(-2000 * neighbor_squared_distances_list))
            .cuda()
            .float()
            .contiguous()
        )
        variables.neighbor_distances_list = (
            torch.tensor(np.sqrt(neighbor_squared_distances_list))
            .cuda()
            .float()
            .contiguous()
        )
        variables.initial_background_means = background_means.detach()
        variables.initial_background_rotations = background_rotations.detach()
        variables.previous_means = gaussian_cloud_parameters[
            GaussianCloudParameterNames.means
        ].detach()
        variables.previous_rotations = torch.nn.functional.normalize(
            gaussian_cloud_parameters[GaussianCloudParameterNames.rotation_quaternions]
        ).detach()
        parameters_to_fix = [
            GaussianCloudParameterNames.opacities_logits,
            GaussianCloudParameterNames.log_scales,
            GaussianCloudParameterNames.camera_matrices,
            GaussianCloudParameterNames.camera_centers,
        ]
        for parameter_group in optimizer.param_groups:
            if parameter_group["name"] in parameters_to_fix:
                parameter_group["lr"] = 0.0

    def _set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)
        self.output_directory_path = os.path.abspath(self.output_directory_path)

    def _initialize_parameters_and_variables(self, metadata):
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
            initial_point_cloud[:, :3], 3
        )
        parameters = {
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
        camera_centers = np.linalg.inv(metadata["w2c"][0])[:, :3, 3]
        return parameters, Variables(
            max_2D_radius=torch.zeros(
                parameters[GaussianCloudParameterNames.means].shape[0]
            )
            .cuda()
            .float(),
            scene_radius=1.1
            * np.max(
                np.linalg.norm(
                    camera_centers - np.mean(camera_centers, 0)[None], axis=-1
                )
            ),
            means2D_gradient_accum=torch.zeros(
                parameters[GaussianCloudParameterNames.means].shape[0]
            )
            .cuda()
            .float(),
            denom=torch.zeros(parameters[GaussianCloudParameterNames.means].shape[0])
            .cuda()
            .float(),
        )

    def _save_sequence(self, gaussian_cloud_parameters_sequence):
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
        gaussian_cloud_parameters, variables = (
            self._initialize_parameters_and_variables(dataset_metadata)
        )
        optimizer = self._create_optimizer(
            gaussian_cloud_parameters, variables.external_scene_radius
        )
        gaussian_cloud_parameters_sequence = []
        timestep_count = get_timestep_count(self.timestep_count_limit, dataset_metadata)
        for timestep in range(timestep_count):
            timestep_captures = load_timestep_captures(
                dataset_metadata, timestep, self.data_directory_path, self.sequence_name
            )
            timestep_capture_buffer = []
            is_initial_timestep = timestep == 0
            if not is_initial_timestep:
                self._update_variables_and_optimizer_and_gaussian_cloud_parameters(
                    gaussian_cloud_parameters, variables, optimizer
                )
            iteration_range = range(10000 if is_initial_timestep else 2000)
            progress_bar = tqdm(iteration_range, desc=f"timestep {timestep}")
            for i in iteration_range:
                capture = get_random_element(
                    input_list=timestep_capture_buffer, fallback_list=timestep_captures
                )
                loss = self._calculate_loss(
                    gaussian_cloud_parameters=gaussian_cloud_parameters,
                    target_capture=capture,
                    variables=variables,
                    calculate_only_subset_of_losses=is_initial_timestep,
                )
                loss.backward()
                with torch.no_grad():
                    report_interval = 100
                    if i % report_interval == 0:
                        self._report_progress(
                            gaussian_cloud_parameters,
                            timestep_captures[0],
                            progress_bar,
                            report_interval,
                        )
                    if is_initial_timestep:
                        external.densify_gaussians(
                            gaussian_cloud_parameters, variables, optimizer, i
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            progress_bar.close()
            gaussian_cloud_parameters_sequence.append(
                self._convert_parameters_to_numpy(
                    parameters=gaussian_cloud_parameters,
                    include_dynamic_parameters_only=not is_initial_timestep,
                )
            )
            if is_initial_timestep:
                self._initialize_post_first_timestep(
                    gaussian_cloud_parameters, variables, optimizer
                )
            if not is_initial_timestep:
                self._save_sequence(gaussian_cloud_parameters_sequence)


def main():
    command = Trainer.__new__(Trainer)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
