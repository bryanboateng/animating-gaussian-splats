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

import snapshot_collection.create
from commons.classes import GaussianCloudParameterNames
from commons.command import Command
from commons.create_commons import (
    load_timestep_captures,
    get_scene_radius,
    create_optimizer,
    convert_parameters_to_numpy,
    get_random_element,
    get_timestep_count,
    initialize_post_first_timestep,
    initialize_parameters,
    update_previous_timestep_gaussian_cloud_state,
    train_first_timestep,
)
from commons.loss import calculate_full_loss
from deformation_network.deformation_network import (
    update_parameters,
    DeformationNetwork,
)


@dataclass
class Create(Command):
    sequence_name: str = MISSING
    data_directory_path: str = MISSING
    experiment_id: str = datetime.utcnow().isoformat() + "Z"
    learning_rate: float = 0.01
    timestep_count_limit: Optional[int] = None
    output_directory_path: str = "./deformation_networks"

    @staticmethod
    def _update_previous_timestep_gaussian_cloud_state(
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        previous_timestep_gaussian_cloud_state: snapshot_collection.create.GaussianCloudReferenceState,
        neighborhood_indices: torch.Tensor,
    ):
        current_means = gaussian_cloud_parameters[GaussianCloudParameterNames.means]
        current_rotations = torch.nn.functional.normalize(
            gaussian_cloud_parameters[GaussianCloudParameterNames.rotation_quaternions]
        )

        update_previous_timestep_gaussian_cloud_state(
            current_means,
            current_rotations,
            gaussian_cloud_parameters,
            neighborhood_indices,
            previous_timestep_gaussian_cloud_state,
        )

    def _set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)
        self.output_directory_path = os.path.abspath(self.output_directory_path)

    def _train_first_timestep(
        self,
        dataset_metadata,
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    ):
        scene_radius = get_scene_radius(dataset_metadata=dataset_metadata)
        optimizer = create_optimizer(
            parameters=gaussian_cloud_parameters, scene_radius=scene_radius
        )
        train_first_timestep(
            data_directory_path=self.data_directory_path,
            sequence_name=self.sequence_name,
            dataset_metadata=dataset_metadata,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            optimizer=optimizer,
            scene_radius=scene_radius,
            method="deformation-network",
        )
        return initialize_post_first_timestep(
            gaussian_cloud_parameters=gaussian_cloud_parameters, optimizer=optimizer
        )

    def _save_and_log_checkpoint(
        self, initial_gaussian_cloud_parameters, deformation_network, timestep_count
    ):
        network_directory_path = os.path.join(
            self.output_directory_path,
            self.experiment_id,
            self.sequence_name,
        )

        to_save = {}
        parameters_numpy = convert_parameters_to_numpy(
            initial_gaussian_cloud_parameters, False
        )
        for k in parameters_numpy.keys():
            to_save[k] = parameters_numpy[k]
        parameters_save_path = os.path.join(
            network_directory_path,
            "initial_gaussian_cloud_parameters.npz",
        )
        os.makedirs(os.path.dirname(parameters_save_path), exist_ok=True)
        np.savez(parameters_save_path, **to_save)

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
        initial_gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        initial_background: snapshot_collection.create.Background,
        initial_neighborhoods: snapshot_collection.create.Neighborhoods,
        previous_timestep_gaussian_cloud_state: snapshot_collection.create.GaussianCloudReferenceState,
        optimizer,
    ):
        for timestep in range(1, timestep_count):
            timestep_capture_list = load_timestep_captures(
                dataset_metadata, timestep, self.data_directory_path, self.sequence_name
            )
            timestep_capture_buffer = []

            for _ in tqdm(range(10_000)):
                capture = get_random_element(
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
            wandb.log({f"mean-timestep-losses-deformation-network": sum(losses) / len(losses)})
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
        timestep_count = get_timestep_count(self.timestep_count_limit, dataset_metadata)
        deformation_network = DeformationNetwork(timestep_count).cuda()
        optimizer = torch.optim.Adam(params=deformation_network.parameters(), lr=1e-3)

        gaussian_cloud_parameters = initialize_parameters(
            data_directory_path=self.data_directory_path,
            sequence_name=self.sequence_name,
        )

        (
            initial_background,
            initial_neighborhoods,
            previous_timestep_gaussian_cloud_state,
        ) = self._train_first_timestep(dataset_metadata, gaussian_cloud_parameters)

        for parameter in gaussian_cloud_parameters.values():
            parameter.requires_grad = False
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
