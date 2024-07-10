import json
import os
import shutil
from dataclasses import dataclass, MISSING
from datetime import datetime
from random import randint
from typing import Optional

import torch
import wandb
from tqdm import tqdm

from commons.command import Command
from commons.training_commons import (
    load_timestep_captures,
    get_random_element,
    Capture,
    get_timestep_count,
)
from deformation_network.common import (
    update_parameters,
    load_and_freeze_parameters,
    DeformationNetwork,
    create_gaussian_cloud,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer


@dataclass
class Create(Command):
    sequence_name: str = MISSING
    data_directory_path: str = MISSING
    experiment_id: str = datetime.utcnow().isoformat() + "Z"
    learning_rate: float = 0.01
    timestep_count_limit: Optional[int] = None
    output_directory_path: str = "./deformation_networks"

    @staticmethod
    def _get_loss(parameters, target_capture: Capture):
        gaussian_cloud = create_gaussian_cloud(parameters)
        # gaussian_cloud['means2D'].retain_grad()
        (
            rendered_image,
            _,
            _,
        ) = Renderer(
            raster_settings=target_capture.camera.gaussian_rasterization_settings
        )(**gaussian_cloud)
        return torch.nn.functional.l1_loss(rendered_image, target_capture.image)

    def _set_absolute_paths(self):
        self.data_directory_path = os.path.abspath(self.data_directory_path)
        self.output_directory_path = os.path.abspath(self.output_directory_path)

    def _get_initial_gaussian_cloud_parameters_path(self):
        return os.path.join(
            self.data_directory_path,
            self.sequence_name,
            "params.pth",
        )

    def _save_and_log_checkpoint(self, deformation_network, timestep_count):
        network_directory_path = os.path.join(
            self.output_directory_path,
            self.experiment_id,
            self.sequence_name,
        )

        destination = os.path.join(
            network_directory_path,
            "initial_gaussian_cloud_parameters.pth",
        )
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        shutil.copy(
            self._get_initial_gaussian_cloud_parameters_path(),
            destination,
        )

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
        parameters,
        optimizer,
    ):
        for timestep in range(timestep_count):
            timestep_capture_list = load_timestep_captures(
                dataset_metadata, timestep, self.data_directory_path, self.sequence_name
            )
            timestep_capture_buffer = []

            for _ in tqdm(range(10_000)):
                capture = get_random_element(
                    input_list=timestep_capture_buffer,
                    fallback_list=timestep_capture_list,
                )
                updated_parameters = update_parameters(
                    deformation_network, parameters, timestep
                )
                loss = self._get_loss(updated_parameters, capture)
                wandb.log(
                    {
                        f"loss-{timestep}": loss.item(),
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
                    loss = self._get_loss(updated_parameters, capture)
                    losses.append(loss.item())
            wandb.log({f"mean-losses": sum(losses) / len(losses)})
            self._save_and_log_checkpoint(deformation_network, timestep_count)

    def _train_in_random_order(
        self,
        timestep_count,
        dataset_metadata,
        deformation_network,
        parameters,
        optimizer,
    ):
        list_of_timestep_capture_lists = []
        for timestep in range(timestep_count):
            list_of_timestep_capture_lists += [
                load_timestep_captures(
                    dataset_metadata,
                    timestep,
                    self.data_directory_path,
                    self.sequence_name,
                )
            ]
        for _ in tqdm(range(10_000)):
            random_timestep = torch.randint(
                0, len(list_of_timestep_capture_lists), (1,)
            )
            random_camera_index = torch.randint(
                0, len(list_of_timestep_capture_lists[0]), (1,)
            )
            capture = list_of_timestep_capture_lists[random_timestep][
                random_camera_index
            ]

            updated_parameters = update_parameters(
                deformation_network, parameters, random_timestep
            )
            loss = self._get_loss(updated_parameters, capture)
            wandb.log(
                {
                    f"loss-new": loss.item(),
                }
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        for timestep_capture_list in list_of_timestep_capture_lists:
            losses = []
            with torch.no_grad():
                for capture in timestep_capture_list:
                    loss = self._get_loss(updated_parameters, capture)
                    losses.append(loss.item())

            wandb.log({f"mean-losses-new": sum(losses) / len(losses)})
        self._save_and_log_checkpoint(deformation_network, timestep_count)

    def run(self):
        self._set_absolute_paths()
        wandb.init(project="new-dynamic-gaussians")
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

        parameters = load_and_freeze_parameters(
            self._get_initial_gaussian_cloud_parameters_path()
        )
        self._train_in_sequential_order(
            timestep_count,
            dataset_metadata,
            deformation_network,
            parameters,
            optimizer,
        )

        self._train_in_random_order(
            timestep_count,
            dataset_metadata,
            deformation_network,
            parameters,
            optimizer,
        )


def main():
    command = Create.__new__(Create)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
