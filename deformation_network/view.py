import json
import os
from dataclasses import dataclass, MISSING

import imageio
import torch
from tqdm import tqdm

from commons.command import Command
from commons.helpers import render_and_increase_yaw
from deformation_network.common import (
    create_gaussian_cloud,
    update_parameters,
    load_and_freeze_parameters,
    DeformationNetwork,
)


@dataclass
class View(Command):
    experiment_id: str = MISSING
    sequence_name: str = MISSING
    networks_directory_path: str = MISSING

    rendered_sequence_directory_path: str = "./renders/"
    fps: int = 30
    starting_yaw: float = 90
    yaw_degrees_per_second: float = 30

    image_width: int = 640
    image_height: int = 360
    near_clipping_plane_distance: float = 0.01
    far_clipping_plane_distance: int = 100
    distance_to_center: float = 2.4
    height: float = 1.3
    aspect_ratio: float = 0.82

    @staticmethod
    def _get_timestep_parameters(deformation_network, parameters, timestep):
        if timestep == 0:
            updated_parameters = parameters
        else:
            updated_parameters = update_parameters(
                deformation_network, parameters, timestep
            )
        return updated_parameters

    def _set_absolute_paths(self):
        self.networks_directory_path = os.path.abspath(self.networks_directory_path)
        self.rendered_sequence_directory_path = os.path.abspath(
            self.rendered_sequence_directory_path
        )

    def run(self):
        self._set_absolute_paths()

        network_directory_path = os.path.join(
            self.networks_directory_path,
            self.experiment_id,
            self.sequence_name,
        )
        initial_timestep_gaussian_cloud_parameters = load_and_freeze_parameters(
            os.path.join(
                network_directory_path,
                "initial_gaussian_cloud_parameters.pth",
            )
        )

        render_images = []
        yaw = self.starting_yaw
        timestep_count = json.load(
            open(
                os.path.join(
                    network_directory_path,
                    "metadata.json",
                ),
                "r",
            )
        )["timestep_count"]

        network_state_dict_path = os.path.join(
            network_directory_path,
            "network_state_dict.pth",
        )
        deformation_network = DeformationNetwork(timestep_count).cuda()
        deformation_network.load_state_dict(torch.load(network_state_dict_path))
        deformation_network.eval()

        for timestep in tqdm(range(timestep_count), desc="Rendering progress"):
            timestep_gaussian_cloud_parameters = self._get_timestep_parameters(
                deformation_network,
                initial_timestep_gaussian_cloud_parameters,
                timestep,
            )
            gaussian_cloud = create_gaussian_cloud(timestep_gaussian_cloud_parameters)
            render_and_increase_yaw(
                gaussian_cloud=gaussian_cloud,
                render_images=render_images,
                image_width=self.image_width,
                image_height=self.image_height,
                near_clipping_plane_distance=self.near_clipping_plane_distance,
                far_clipping_plane_distance=self.far_clipping_plane_distance,
                yaw=yaw,
                yaw_degrees_per_second=self.yaw_degrees_per_second,
                distance_to_center=self.distance_to_center,
                height=self.height,
                aspect_ratio=self.aspect_ratio,
                fps=self.fps,
            )

        rendered_sequence_path = os.path.join(
            self.rendered_sequence_directory_path,
            self.experiment_id,
            f"{self.sequence_name}.mp4",
        )
        os.makedirs(os.path.dirname(rendered_sequence_path), exist_ok=True)
        imageio.mimwrite(
            rendered_sequence_path,
            render_images,
            fps=self.fps,
        )


def main():
    command = View.__new__(View)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
