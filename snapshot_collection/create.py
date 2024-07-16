import json
import os
from dataclasses import dataclass, MISSING
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import wandb
from tqdm import tqdm

from commons.classes import (
    Background,
    GaussianCloudParameterNames,
    GaussianCloudReferenceState,
    Neighborhoods,
)
from commons.command import Command
from commons.create_commons import (
    load_timestep_captures,
    get_random_element,
    get_timestep_count,
    initialize_post_first_timestep,
    initialize_parameters,
    convert_parameters_to_numpy,
    get_scene_radius,
    create_optimizer,
    update_previous_timestep_gaussian_cloud_state,
    train_first_timestep,
)
from commons.loss import calculate_full_loss
from snapshot_collection.external import update_params_and_optimizer


@dataclass
class Create(Command):
    sequence_name: str = MISSING
    data_directory_path: str = MISSING
    output_directory_path: str = "./output/"
    experiment_id: str = datetime.utcnow().isoformat() + "Z"
    timestep_count_limit: Optional[int] = None

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

        update_previous_timestep_gaussian_cloud_state(
            current_means,
            current_rotations,
            gaussian_cloud_parameters,
            neighborhood_indices,
            previous_timestep_gaussian_cloud_state,
        )

        updated_parameters = {
            GaussianCloudParameterNames.means: updated_means,
            GaussianCloudParameterNames.rotation_quaternions: updated_rotations,
        }
        update_params_and_optimizer(
            updated_parameters, gaussian_cloud_parameters, optimizer
        )

    def _set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)
        self.output_directory_path = os.path.abspath(self.output_directory_path)

    def _train_first_timestep(
        self,
        dataset_metadata,
        gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
        optimizer: torch.optim.Adam,
        scene_radius: float,
    ):
        train_first_timestep(
            data_directory_path=self.data_directory_path,
            sequence_name=self.sequence_name,
            dataset_metadata=dataset_metadata,
            gaussian_cloud_parameters=gaussian_cloud_parameters,
            optimizer=optimizer,
            scene_radius=scene_radius,
            method="snapshot-collection",
        )
        return initialize_post_first_timestep(
            gaussian_cloud_parameters=gaussian_cloud_parameters, optimizer=optimizer
        ), convert_parameters_to_numpy(
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
            loss = calculate_full_loss(
                gaussian_cloud_parameters=gaussian_cloud_parameters,
                target_capture=capture,
                initial_background=initial_background,
                initial_neighborhoods=initial_neighborhoods,
                previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
            )
            wandb.log(
                {
                    f"timestep-{timestep}-loss-snapshot-collection": loss.item(),
                }
            )
            loss.backward()
            with torch.no_grad():
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return convert_parameters_to_numpy(
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
        wandb.init(project="4d-gaussian-splatting")
        torch.cuda.empty_cache()
        gaussian_cloud_parameters = initialize_parameters(
            data_directory_path=self.data_directory_path,
            sequence_name=self.sequence_name,
        )
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
        scene_radius = get_scene_radius(dataset_metadata)
        optimizer = create_optimizer(
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
