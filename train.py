import json
import math
import os
from dataclasses import dataclass, MISSING
from typing import Optional

import imageio
import numpy as np
import torch
import wandb
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from commons.classes import (
    GaussianCloudReferenceState,
    Neighborhoods,
    GaussianCloudParameters,
    Camera,
)
from commons.command import Command
from commons.helpers import (
    compute_knn_indices_and_squared_distances,
    load_timestep_views,
)
from commons.loss import calculate_full_loss, GaussianCloud
from deformation_network import (
    update_parameters,
    DeformationNetwork,
    normalize_means_and_rotations,
)


@dataclass
class Create(Command):
    sequence_name: str = MISSING
    data_directory_path: str = MISSING
    learning_rate: float = 0.01
    timestep_count_limit: Optional[int] = None
    output_directory_path: str = "./deformation_networks"
    iteration_count: int = 200_000
    warmup_iteration_ratio: float = 0.075
    fps: int = 30

    @staticmethod
    def initialize_post_first_timestep(
        gaussian_cloud_parameters: GaussianCloudParameters,
    ):

        foreground_mask = gaussian_cloud_parameters.segmentation_colors[:, 0] > 0.5
        foreground_means = gaussian_cloud_parameters.means[foreground_mask]
        neighbor_indices_list, neighbor_squared_distances_list = (
            compute_knn_indices_and_squared_distances(
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

        previous_timestep_gaussian_cloud_state = GaussianCloudReferenceState(
            means=gaussian_cloud_parameters.means.detach(),
            rotations=torch.nn.functional.normalize(
                gaussian_cloud_parameters.rotation_quaternions
            ).detach(),
        )

        return neighborhoods, previous_timestep_gaussian_cloud_state

    @staticmethod
    def _get_linear_warmup_cos_annealing(optimizer, warmup_iters, total_iters):
        scheduler_warmup = LinearLR(
            optimizer, start_factor=1 / 1000, total_iters=warmup_iters
        )
        scheduler_cos_decay = CosineAnnealingLR(
            optimizer, T_max=total_iters - warmup_iters
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler_warmup, scheduler_cos_decay],
            milestones=[warmup_iters],
        )

        return scheduler

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
            inverted_foreground_rotations.detach().clone()
        )
        previous_timestep_gaussian_cloud_state.offsets_to_neighbors = (
            offsets_to_neighbors.detach().clone()
        )
        previous_timestep_gaussian_cloud_state.colors = (
            gaussian_cloud_parameters.rgb_colors.detach().clone()
        )
        previous_timestep_gaussian_cloud_state.means = current_means.detach().clone()
        previous_timestep_gaussian_cloud_state.rotations = (
            current_rotations.detach().clone()
        )

    @staticmethod
    def create_transformation_matrix(yaw_degrees, height, distance_to_center):
        yaw_radians = np.radians(yaw_degrees)
        return np.array(
            [
                [np.cos(yaw_radians), 0.0, -np.sin(yaw_radians), 0.0],
                [0.0, 1.0, 0.0, height],
                [np.sin(yaw_radians), 0.0, np.cos(yaw_radians), distance_to_center],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def _set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)
        self.output_directory_path = os.path.abspath(self.output_directory_path)

    def _load_densified_initial_parameters(self):
        parameters: GaussianCloudParameters = torch.load(
            os.path.join(
                self.data_directory_path,
                self.sequence_name,
                "densified_initial_gaussian_cloud_parameters.pth",
            ),
            map_location="cuda",
        )
        for parameter in parameters.__dict__.values():
            parameter.requires_grad = False
        return parameters

    def _get_timestep_count(self, dataset_metadata):
        sequence_length = len(dataset_metadata["fn"])
        if self.timestep_count_limit is None:
            return sequence_length
        else:
            return min(sequence_length, self.timestep_count_limit)

    def _load_all_views(self, dataset_metadata, timestep_count):
        views = []
        for timestep in range(1, timestep_count + 1):
            views += [
                load_timestep_views(
                    dataset_metadata,
                    timestep,
                    self.data_directory_path,
                    self.sequence_name,
                )
            ]
        return views

    def _export_deformation_network(
        self,
        initial_gaussian_cloud_parameters: GaussianCloudParameters,
        deformation_network: DeformationNetwork,
        timestep_count: int,
    ):
        run_output_directory_path = os.path.join(
            self.output_directory_path,
            f"{self.sequence_name}_{wandb.run.name}",
        )
        network_directory_path = os.path.join(
            run_output_directory_path,
            f"deformation_network_{self.sequence_name}_{wandb.run.name}",
        )
        os.makedirs(network_directory_path, exist_ok=True)
        parameters_save_path = os.path.join(
            network_directory_path,
            f"{self.sequence_name}_densified_initial_gaussian_cloud_parameters.pth",
        )
        torch.save(initial_gaussian_cloud_parameters, parameters_save_path)

        with open(
            os.path.join(
                network_directory_path,
                "timestep_count",
            ),
            "w",
        ) as file:
            file.write(f"{timestep_count}")

        network_state_dict_path = os.path.join(
            network_directory_path,
            f"deformation_network_state_dict_{self.sequence_name}_{wandb.run.name}.pth",
        )
        torch.save(deformation_network.state_dict(), network_state_dict_path)
        wandb.save(
            os.path.join(
                network_directory_path,
                "*",
            ),
            base_path=run_output_directory_path,
        )

    def _export_visualization(
        self,
        deformation_network,
        name,
        extrinsic_matrix,
        initial_gaussian_cloud_parameters,
        means_norm,
        pos_smol,
        render_images,
        rotations_norm,
        timestep_count,
        visualizations_directory_path,
    ):
        for timestep in tqdm(range(timestep_count), desc="Rendering progress"):
            if timestep == 0:
                timestep_gaussian_cloud_parameters = initial_gaussian_cloud_parameters
            else:
                timestep_gaussian_cloud_parameters = update_parameters(
                    deformation_network=deformation_network,
                    positional_encoding=pos_smol,
                    normalized_means=means_norm,
                    normalized_rotations=rotations_norm,
                    parameters=initial_gaussian_cloud_parameters,
                    timestep=timestep,
                    timestep_count=timestep_count,
                )

            gaussian_cloud = GaussianCloud(
                parameters=timestep_gaussian_cloud_parameters
            )

            image_width = 1280
            image_height = 720
            aspect_ratio: float = 0.82
            camera = Camera(
                id_=0,
                image_width=image_width,
                image_height=image_height,
                near_clipping_plane_distance=1,
                far_clipping_plane_distance=100,
                intrinsic_matrix=np.array(
                    [
                        [aspect_ratio * image_width, 0, image_width / 2],
                        [0, aspect_ratio * image_width, image_height / 2],
                        [0, 0, 1],
                    ]
                ),
                extrinsic_matrix=extrinsic_matrix,
            )
            (
                image,
                _,
                _,
            ) = Renderer(
                raster_settings=camera.gaussian_rasterization_settings
            )(**gaussian_cloud.get_renderer_format())
            render_images.append(
                (
                    255
                    * np.clip(
                        image.cpu().numpy(),
                        0,
                        1,
                    )
                )
                .astype(np.uint8)
                .transpose(1, 2, 0)
            )
        rendered_sequence_path = os.path.join(
            visualizations_directory_path,
            f"{self.sequence_name}_{name}_{wandb.run.name}.mp4",
        )
        imageio.mimwrite(
            rendered_sequence_path,
            render_images,
            fps=self.fps,
        )

    def _export_visualizations(
        self,
        initial_gaussian_cloud_parameters: GaussianCloudParameters,
        deformation_network: DeformationNetwork,
        timestep_count: int,
    ):
        run_output_directory_path = os.path.join(
            self.output_directory_path,
            f"{self.sequence_name}_{wandb.run.name}",
        )
        visualizations_directory_path = os.path.join(
            run_output_directory_path,
            f"visualizations_{self.sequence_name}_{wandb.run.name}",
        )
        os.makedirs(visualizations_directory_path, exist_ok=True)

        render_images = []

        deformation_network.eval()

        means_norm, pos_smol, rotations_norm = normalize_means_and_rotations(
            initial_gaussian_cloud_parameters
        )

        distance_to_center: float = 2.4
        height: float = 1.3
        extrinsic_matrices = {
            "000": self.create_transformation_matrix(
                yaw_degrees=0, height=height, distance_to_center=distance_to_center
            ),
            "090": self.create_transformation_matrix(
                yaw_degrees=90, height=height, distance_to_center=distance_to_center
            ),
            "180": self.create_transformation_matrix(
                yaw_degrees=180, height=height, distance_to_center=distance_to_center
            ),
            "270": self.create_transformation_matrix(
                yaw_degrees=270, height=height, distance_to_center=distance_to_center
            ),
            "top": np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 3.5],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
        }
        for name, extrinsic_matrix in extrinsic_matrices.items():
            self._export_visualization(
                deformation_network,
                name,
                extrinsic_matrix,
                initial_gaussian_cloud_parameters,
                means_norm,
                pos_smol,
                render_images,
                rotations_norm,
                timestep_count,
                visualizations_directory_path,
            )
        wandb.save(
            os.path.join(
                visualizations_directory_path,
                "*",
            ),
            base_path=run_output_directory_path,
        )

    def run(self):
        self._set_absolute_paths()
        wandb.init(project="animating-gaussian-splats")
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
        timestep_count = self._get_timestep_count(dataset_metadata)
        deformation_network = DeformationNetwork().cuda()
        optimizer = torch.optim.Adam(
            params=deformation_network.parameters(), lr=self.learning_rate
        )
        scheduler = self._get_linear_warmup_cos_annealing(
            optimizer,
            warmup_iters=int(self.iteration_count * self.warmup_iteration_ratio),
            total_iters=self.iteration_count,
        )

        initial_gaussian_cloud_parameters = self._load_densified_initial_parameters()

        (
            initial_neighborhoods,
            previous_timestep_gaussian_cloud_state,
        ) = self.initialize_post_first_timestep(
            gaussian_cloud_parameters=initial_gaussian_cloud_parameters
        )
        normalized_means, pos_smol, normalized_rotations = (
            normalize_means_and_rotations(initial_gaussian_cloud_parameters)
        )
        views = self._load_all_views(dataset_metadata, timestep_count)
        for i in tqdm(range(self.iteration_count)):
            timestep = i % timestep_count
            camera_index = torch.randint(0, len(views[0]), (1,))

            if timestep == 0:
                self._update_previous_timestep_gaussian_cloud_state(
                    gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
                    previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
                    neighborhood_indices=initial_neighborhoods.indices,
                )

            view = views[timestep][camera_index]
            updated_gaussian_cloud_parameters = update_parameters(
                deformation_network=deformation_network,
                positional_encoding=pos_smol,
                normalized_means=normalized_means,
                normalized_rotations=normalized_rotations,
                parameters=initial_gaussian_cloud_parameters,
                timestep=timestep,
                timestep_count=timestep_count,
            )
            loss = calculate_full_loss(
                gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
                target_view=view,
                initial_neighborhoods=initial_neighborhoods,
                previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
                rigidity_loss_weight=(
                    2.0 / (1.0 + math.exp(-6 * (i / self.iteration_count))) - 1
                ),
            )
            self._update_previous_timestep_gaussian_cloud_state(
                gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
                previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
                neighborhood_indices=initial_neighborhoods.indices,
            )
            wandb.log(
                {
                    f"loss-random": loss.item(),
                    f"lr": optimizer.param_groups[0]["lr"],
                }
            )

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        for timestep_views in views:
            losses = []
            with torch.no_grad():
                for view in timestep_views:
                    loss = calculate_full_loss(
                        gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
                        target_view=view,
                        initial_neighborhoods=initial_neighborhoods,
                        previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
                        rigidity_loss_weight=1.0,
                    )
                    losses.append(loss.item())
            wandb.log({f"mean-timestep-losses": sum(losses) / len(losses)})
        self._export_deformation_network(
            initial_gaussian_cloud_parameters, deformation_network, timestep_count
        )
        self._export_visualizations(
            initial_gaussian_cloud_parameters, deformation_network, timestep_count
        )


def main():
    command = Create.__new__(Create)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
