import json
import math
import os
from dataclasses import dataclass, MISSING
from typing import Optional

import numpy as np
import torch
import wandb
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from commons.classes import (
    GaussianCloudReferenceState,
    Neighborhoods,
    GaussianCloudParameters,
)
from commons.command import Command
from commons.helpers import (
    compute_knn_indices_and_squared_distances,
    load_timestep_views,
)
from commons.loss import calculate_full_loss
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

    def _save_and_log_checkpoint(
        self,
        initial_gaussian_cloud_parameters: GaussianCloudParameters,
        deformation_network: DeformationNetwork,
        timestep_count: int,
    ):
        network_directory_path = os.path.join(
            self.output_directory_path,
            wandb.run.name,
            self.sequence_name,
        )

        parameters_save_path = os.path.join(
            network_directory_path,
            "densified_initial_gaussian_cloud_parameters.pth",
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
            optimizer, warmup_iters=15_000, total_iters=self.iteration_count
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
        self._save_and_log_checkpoint(
            initial_gaussian_cloud_parameters, deformation_network, timestep_count
        )


def main():
    command = Create.__new__(Create)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
