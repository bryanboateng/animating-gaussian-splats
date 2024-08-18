import json
import os
from dataclasses import dataclass, MISSING

import imageio
import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tqdm import tqdm

from commons.classes import GaussianCloudParameters, Camera
from commons.command import Command
from deformation_network import (
    normalize_means_and_rotations,
    update_parameters,
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
    def _get_timestep_parameters(
        deformation_network,
        positional_encoding,
        means_norm,
        rotations_norm,
        parameters,
        timestep,
        timestep_count,
    ):
        if timestep == 0:
            updated_parameters = parameters
        else:
            updated_parameters = update_parameters(
                deformation_network=deformation_network,
                positional_encoding=positional_encoding,
                normalized_means=means_norm,
                normalized_rotations=rotations_norm,
                parameters=parameters,
                timestep=timestep,
                timestep_count=timestep_count,
            )
        return updated_parameters

    @staticmethod
    def _create_gaussian_cloud(parameters: GaussianCloudParameters):
        return {
            "means3D": parameters.means,
            "colors_precomp": parameters.rgb_colors,
            "rotations": torch.nn.functional.normalize(parameters.rotation_quaternions),
            "opacities": torch.sigmoid(parameters.opacities_logits),
            "scales": torch.exp(parameters.log_scales),
            "means2D": torch.zeros_like(
                parameters.means,
                requires_grad=True,
                device="cuda",
            )
            + 0,
        }

    def _set_absolute_paths(self):
        self.networks_directory_path = os.path.abspath(self.networks_directory_path)
        self.rendered_sequence_directory_path = os.path.abspath(
            self.rendered_sequence_directory_path
        )

    def run(self):
        with torch.no_grad():
            self._set_absolute_paths()

            network_directory_path = os.path.join(
                self.networks_directory_path,
                self.experiment_id,
                self.sequence_name,
            )
            initial_gaussian_cloud_parameters: GaussianCloudParameters = torch.load(
                os.path.join(
                    network_directory_path,
                    "densified_gaussian_cloud_parameters.pth",
                ),
                map_location="cuda",
            )
            for parameter in initial_gaussian_cloud_parameters.__dict__.values():
                parameter.requires_grad = False

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
            deformation_network = DeformationNetwork().cuda()
            deformation_network.load_state_dict(torch.load(network_state_dict_path))
            deformation_network.eval()

            means_norm, pos_smol, rotations_norm = normalize_means_and_rotations(
                initial_gaussian_cloud_parameters
            )
            for timestep in tqdm(range(timestep_count), desc="Rendering progress"):
                timestep_gaussian_cloud_parameters = self._get_timestep_parameters(
                    deformation_network=deformation_network,
                    positional_encoding=pos_smol,
                    means_norm=means_norm,
                    rotations_norm=rotations_norm,
                    parameters=initial_gaussian_cloud_parameters,
                    timestep=timestep,
                    timestep_count=timestep_count,
                )
                gaussian_cloud = self._create_gaussian_cloud(
                    timestep_gaussian_cloud_parameters
                )
                camera = Camera.from_parameters(
                    id_=0,
                    image_width=self.image_width,
                    image_height=self.image_height,
                    near_clipping_plane_distance=self.near_clipping_plane_distance,
                    far_clipping_plane_distance=self.far_clipping_plane_distance,
                    yaw=yaw,
                    distance_to_center=self.distance_to_center,
                    height=self.height,
                    aspect_ratio=self.aspect_ratio,
                )
                (
                    image,
                    _,
                    _,
                ) = Renderer(
                    raster_settings=camera.gaussian_rasterization_settings
                )(**gaussian_cloud)
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
                yaw += self.yaw_degrees_per_second / self.fps

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
