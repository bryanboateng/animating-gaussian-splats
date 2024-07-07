import os
import time
from dataclasses import dataclass, MISSING

import imageio
import numpy as np
import torch
from tqdm import tqdm

from command import Command

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import (
    Camera,
    GaussianCloudParameterNames,
)


@dataclass
class CloudVideoGenerator(Command):
    parameters_directory_path: str = MISSING
    rendered_sequence_directory_path = "./renders/"
    experiment_id: str = MISSING
    sequence_name: str = MISSING
    render_fps = 30
    render_degrees_per_second = 30

    image_width = 640
    image_height = 360
    near_clipping_plane_distance = 0.01
    far_clipping_plane_distance = 100
    distance_to_center = 2.4
    height = 1.3
    aspect_ratio = 0.82

    seg_as_col = False

    def _set_absolute_paths(self):
        self.parameters_directory_path = os.path.abspath(self.parameters_directory_path)
        self.rendered_sequence_directory_path = os.path.abspath(
            self.rendered_sequence_directory_path
        )

    def _load_gaussian_sequence(self):
        parameters_path = os.path.join(
            self.parameters_directory_path,
            self.experiment_id,
            self.sequence_name,
            "params.npz",
        )
        gaussian_sequence = dict(np.load(parameters_path))
        tensor_data = {
            k: torch.tensor(v).cuda().float() for k, v in gaussian_sequence.items()
        }
        processed_gaussian_sequence = []
        for timestamp in range(len(tensor_data[GaussianCloudParameterNames.means])):
            processed_gaussian_sequence.append(
                {
                    "means3D": tensor_data[GaussianCloudParameterNames.means][
                        timestamp
                    ],
                    "colors_precomp": (
                        tensor_data[GaussianCloudParameterNames.colors][timestamp]
                        if not self.seg_as_col
                        else tensor_data[GaussianCloudParameterNames.segmentation_masks]
                    ),
                    "rotations": torch.nn.functional.normalize(
                        tensor_data[GaussianCloudParameterNames.rotation_quaternions][
                            timestamp
                        ]
                    ),
                    "opacities": torch.sigmoid(
                        tensor_data[GaussianCloudParameterNames.opacities_logits]
                    ),
                    "scales": torch.exp(
                        tensor_data[GaussianCloudParameterNames.log_scales]
                    ),
                    "means2D": torch.zeros_like(
                        tensor_data[GaussianCloudParameterNames.means][0], device="cuda"
                    ),
                }
            )
        return processed_gaussian_sequence

    def _render_gaussians(self, gaussians, yaw: float):
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
        with torch.no_grad():
            (
                image,
                _,
                _,
            ) = Renderer(
                raster_settings=camera.gaussian_rasterization_settings
            )(**gaussians)
            return image

    def run(self):
        self._set_absolute_paths()
        gaussian_sequence = self._load_gaussian_sequence()
        render_images = []
        start_time = time.time()
        yaw = 0
        for gaussians in tqdm(gaussian_sequence, desc="Rendering progress"):
            render_images.append(
                (
                    255
                    * np.clip(
                        self._render_gaussians(gaussians, yaw).cpu().numpy(), 0, 1
                    )
                )
                .astype(np.uint8)
                .transpose(1, 2, 0)
            )
            yaw += self.render_degrees_per_second / self.render_fps
        finish_time = time.time()
        print(
            "Possible FPS:", (len(gaussian_sequence) - 1) / (finish_time - start_time)
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
            fps=self.render_fps,
        )


def main():
    command = CloudVideoGenerator.__new__(CloudVideoGenerator)
    command.parse_args()
    command.run()


if __name__ == "__main__":
    main()
