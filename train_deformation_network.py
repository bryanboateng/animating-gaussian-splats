import copy
import json
import os
from dataclasses import dataclass, MISSING
from datetime import datetime
from random import randint

import torch
import wandb
from torch import nn
from tqdm import tqdm

from command import Command
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from training_commons import load_timestep_captures, get_random_element, Capture


class DeformationNetwork(nn.Module):
    def __init__(self, input_size, sequence_length) -> None:
        super(DeformationNetwork, self).__init__()
        self.embedding_dimension = 2
        self.embedding = nn.Embedding(sequence_length, self.embedding_dimension)
        self.fc1 = nn.Linear(input_size + self.embedding_dimension, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, input_size)

        self.relu = nn.ReLU()

    def forward(self, input_tensor, timestep):
        batch_size = input_tensor.shape[0]
        initial_input_tensor = input_tensor
        embedding_tensor = self.embedding(timestep).repeat(batch_size, 1)
        input_with_embedding = torch.cat((input_tensor, embedding_tensor), dim=1)

        x = self.relu(self.fc1(input_with_embedding))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        return initial_input_tensor + x


@dataclass
class TrainDeformationNetwork(Command):
    sequence_name: str = MISSING
    data_directory_path: str = MISSING
    experiment_id: str = datetime.utcnow().isoformat() + "Z"
    learning_rate = 0.01

    @staticmethod
    def _load_and_freeze_parameters(path: str):
        parameters = torch.load(path)
        for parameter in parameters.values():
            parameter.requires_grad = False
        return parameters

    @staticmethod
    def _create_gaussian_cloud(parameters):
        return {
            "means3D": parameters["means"],
            "colors_precomp": parameters["colors"],
            "rotations": torch.nn.functional.normalize(parameters["rotations"]),
            "opacities": torch.sigmoid(parameters["opacities"]),
            "scales": torch.exp(parameters["scales"]),
            "means2D": torch.zeros_like(
                parameters["means"], requires_grad=True, device="cuda"
            )
            + 0,
        }

    @staticmethod
    def get_loss(parameters, target_capture: Capture):
        gaussian_cloud = TrainDeformationNetwork._create_gaussian_cloud(parameters)
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

    def _update_parameters(
        self, deformation_network: DeformationNetwork, parameters, timestep
    ):
        delta = deformation_network(
            torch.cat((parameters["means"], parameters["rotations"]), dim=1),
            torch.tensor(timestep).cuda(),
        )
        means_delta = delta[:, :3]
        rotations_delta = delta[:, 3:]
        updated_parameters = copy.deepcopy(parameters)
        updated_parameters["means"] = updated_parameters["means"].detach()
        updated_parameters["means"] += means_delta * self.learning_rate
        updated_parameters["rotations"] = updated_parameters["rotations"].detach()
        updated_parameters["rotations"] += rotations_delta * self.learning_rate
        return updated_parameters

    def _save_and_log_checkpoint(self, deformation_network):
        checkpoint_filename = f"{self.experiment_id}.pth"
        torch.save(deformation_network.state_dict(), checkpoint_filename)
        wandb.save(checkpoint_filename)

    def _train_in_sequential_order(
        self,
        sequence_length,
        dataset_metadata,
        deformation_network,
        parameters,
        optimizer,
    ):
        for timestep in range(sequence_length):
            timestep_capture_list = load_timestep_captures(
                dataset_metadata, timestep, self.data_directory_path, self.sequence_name
            )
            timestep_capture_buffer = []

            for _ in tqdm(range(10_000)):
                capture = get_random_element(
                    input_list=timestep_capture_buffer,
                    fallback_list=timestep_capture_list,
                )
                updated_parameters = self._update_parameters(
                    deformation_network, parameters, timestep
                )
                loss = self.get_loss(updated_parameters, capture)
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
                    loss = self.get_loss(updated_parameters, capture)
                    losses.append(loss.item())
            wandb.log({f"mean-losses": sum(losses) / len(losses)})
            self._save_and_log_checkpoint(deformation_network)

    def _train_in_random_order(
        self,
        sequence_length,
        dataset_metadata,
        deformation_network,
        parameters,
        optimizer,
    ):
        list_of_timestep_capture_lists = []
        for timestep in range(sequence_length):
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

            updated_parameters = self._update_parameters(
                deformation_network, parameters, random_timestep
            )
            loss = self.get_loss(updated_parameters, capture)
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
                    loss = self.get_loss(updated_parameters, capture)
                    losses.append(loss.item())

            wandb.log({f"mean-losses-new": sum(losses) / len(losses)})
        self._save_and_log_checkpoint(deformation_network)

    def run(self):
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
        sequence_length = len(dataset_metadata["fn"])
        parameters = self._load_and_freeze_parameters(
            os.path.join(
                self.data_directory_path,
                self.sequence_name,
                "params.pth",
            )
        )
        deformation_network = DeformationNetwork(7, sequence_length).cuda()
        optimizer = torch.optim.Adam(params=deformation_network.parameters(), lr=1e-3)

        self._train_in_sequential_order(
            sequence_length,
            dataset_metadata,
            deformation_network,
            parameters,
            optimizer,
        )

        self._train_in_random_order(
            sequence_length,
            dataset_metadata,
            deformation_network,
            parameters,
            optimizer,
        )


def main():
    command = TrainDeformationNetwork.__new__(TrainDeformationNetwork)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
