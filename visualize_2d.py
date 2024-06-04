import argparse
import json
import os
import time

import imageio
import numpy as np
import torch
from tqdm import tqdm

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from helpers import (
    Camera,
    GaussianCloudParameterNames,
)


class Configuration:
    def __init__(self):
        self.parameters_directory_path = "./parameters/"
        self.rendered_sequence_directory_path = "./renders/"
        self.experiment_id = "foo"
        self.sequence_name = "basketball"
        self.render_fps = 30

        self.image_width = 640
        self.image_height = 360
        self.near_clipping_plane_distance = 0.01
        self.far_clipping_plane_distance = 100
        self.yaw = 0.0
        self.distance_to_center = 2.4
        self.height = 1.3
        self.aspect_ratio = 0.82

        self.seg_as_col = False

    def set_absolute_paths(self):
        self.parameters_directory_path = os.path.abspath(self.parameters_directory_path)
        self.rendered_sequence_directory_path = os.path.abspath(
            self.rendered_sequence_directory_path
        )

    def update_from_arguments(self, args):
        for key, value in vars(args).items():
            if value is not None and hasattr(self, key):
                setattr(self, key, value)
        self.set_absolute_paths()

    def update_from_json(self, json_path):
        with open(json_path, "r") as file:
            data = json.load(file)
            for key, value in data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
        self.set_absolute_paths()


def parse_arguments(configuration: Configuration):
    argument_parser = argparse.ArgumentParser(description="???")
    argument_parser.add_argument(
        "--config_file", type=str, help="Path to the JSON config file"
    )

    for key, value in vars(configuration).items():
        t = type(value)
        if t == bool:
            argument_parser.add_argument(f"--{key}", default=value, action="store_true")
        else:
            argument_parser.add_argument(f"--{key}", default=value, type=t)

    return argument_parser.parse_args()


def load_gaussian_sequence():
    parameters_path = os.path.join(
        config.parameters_directory_path,
        config.experiment_id,
        config.sequence_name,
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
                "means3D": tensor_data[GaussianCloudParameterNames.means][timestamp],
                "colors_precomp": (
                    tensor_data[GaussianCloudParameterNames.colors][timestamp]
                    if not config.seg_as_col
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


def render_gaussians(gaussians):
    with torch.no_grad():
        (
            image,
            _,
            _,
        ) = Renderer(
            raster_settings=camera.gaussian_rasterization_settings
        )(**gaussians)
        return image


def render_sequence():
    gaussian_sequence = load_gaussian_sequence()
    render_images = []
    start_time = time.time()
    for gaussians in tqdm(gaussian_sequence, desc="Rendering progress"):
        render_images.append(
            (255 * np.clip(render_gaussians(gaussians).cpu().numpy(), 0, 1))
            .astype(np.uint8)
            .transpose(1, 2, 0)
        )
    finish_time = time.time()
    print("FPS:", (len(gaussian_sequence) - 1) / (finish_time - start_time))

    rendered_sequence_path = os.path.join(
        config.rendered_sequence_directory_path,
        config.experiment_id,
        f"{config.sequence_name}.mp4",
    )
    os.makedirs(os.path.dirname(rendered_sequence_path), exist_ok=True)
    imageio.mimwrite(
        rendered_sequence_path,
        render_images,
        fps=config.render_fps,
    )


if __name__ == "__main__":
    config = Configuration()
    arguments = parse_arguments(config)

    if arguments.config_file:
        config.update_from_json(arguments.config_json)
    else:
        config.update_from_arguments(arguments)

    camera = Camera.from_parameters(
        id_=0,
        image_width=config.image_width,
        image_height=config.image_height,
        near_clipping_plane_distance=config.near_clipping_plane_distance,
        far_clipping_plane_distance=config.far_clipping_plane_distance,
        yaw=config.yaw,
        distance_to_center=config.distance_to_center,
        height=config.height,
        aspect_ratio=config.aspect_ratio,
    )

    render_sequence()
