import json
import os
from dataclasses import dataclass, MISSING
from datetime import datetime
from random import randint
from typing import Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

from commons.classes import (
    GaussianCloudParameters,
)
from commons.classes import (
    GaussianCloudReferenceState,
    Neighborhoods,
    Background,
)
from commons.command import Command
from commons.helpers import (
    compute_knn_indices_and_squared_distances,
    load_timestep_captures,
)
from commons.loss import calculate_full_loss
from deformation_network import (
    update_parameters,
    DeformationNetwork,
)


@dataclass
class Create(Command):
    sequence_name: str = MISSING
    data_directory_path: str = MISSING
    experiment_id: str = datetime.utcnow().isoformat() + "Z"
    timestep_count_limit: Optional[int] = None
    output_directory_path: str = "./deformation_networks"

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
            timestep_capture_list = load_timestep_captures(
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

        gaussian_cloud_parameters = self._load_densified_initial_parameters()

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
