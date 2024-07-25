import numpy as np
import torch

from commons.classes import Camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

from PIL import Image

import time

def _render_gaussians(
    gaussians,
    image_width,
    image_height,
    near_clipping_plane_distance,
    far_clipping_plane_distance,
    yaw,
    distance_to_center,
    height,
    aspect_ratio,
):
    camera = Camera.from_parameters(
        id_=0,
        image_width=image_width,
        image_height=image_height,
        near_clipping_plane_distance=near_clipping_plane_distance,
        far_clipping_plane_distance=far_clipping_plane_distance,
        yaw=yaw,
        distance_to_center=distance_to_center,
        height=height,
        aspect_ratio=aspect_ratio,
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


def render_and_increase_yaw(
    gaussian_cloud,
    render_images,
    image_width,
    image_height,
    near_clipping_plane_distance,
    far_clipping_plane_distance,
    yaw,
    yaw_degrees_per_second,
    distance_to_center,
    height,
    aspect_ratio,
    fps,
):    
    rendering = (
        (
            255
            * np.clip(
                _render_gaussians(
                    gaussian_cloud,
                    image_width=image_width,
                    image_height=image_height,
                    near_clipping_plane_distance=near_clipping_plane_distance,
                    far_clipping_plane_distance=far_clipping_plane_distance,
                    yaw=yaw,
                    distance_to_center=distance_to_center,
                    height=height,
                    aspect_ratio=aspect_ratio,
                )
                .cpu()
                .numpy(),
                0,
                1,
            )
        )
        .astype(np.uint8)
        .transpose(1, 2, 0)
    )
    render_images.append(rendering)

    yaw += yaw_degrees_per_second / fps
