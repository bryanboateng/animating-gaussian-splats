import math
from typing import Optional

import numpy as np
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings


class GaussianCloudParameters:
    def __init__(
        self,
        means: torch.nn.Parameter,
        rgb_colors: torch.nn.Parameter,
        segmentation_colors: torch.nn.Parameter,
        rotation_quaternions: torch.nn.Parameter,
        opacities_logits: torch.nn.Parameter,
        log_scales: torch.nn.Parameter,
        camera_matrices: torch.nn.Parameter,
        camera_centers: torch.nn.Parameter,
    ):
        self.means = means
        self.rgb_colors = rgb_colors
        self.segmentation_colors = segmentation_colors
        self.rotation_quaternions = rotation_quaternions
        self.opacities_logits = opacities_logits
        self.log_scales = log_scales
        self.camera_matrices = camera_matrices
        self.camera_centers = camera_centers


class Camera:
    def __init__(
        self,
        id_: int,
        image_width: int,
        image_height: int,
        near_clipping_plane_distance: float,
        far_clipping_plane_distance: float,
        intrinsic_matrix,
        extrinsic_matrix,
    ):
        self.id_ = id_

        focal_length_x, focal_length_y, principal_point_x, principal_point_y = (
            intrinsic_matrix[0][0],
            intrinsic_matrix[1][1],
            intrinsic_matrix[0][2],
            intrinsic_matrix[1][2],
        )
        extrinsic_matrix_tensor = torch.tensor(extrinsic_matrix).cuda().float()
        camera_center = torch.inverse(extrinsic_matrix_tensor)[:3, 3]
        extrinsic_matrix_tensor = extrinsic_matrix_tensor.unsqueeze(0).transpose(1, 2)
        opengl_projection_matrix = (
            torch.tensor(
                [
                    [
                        2 * focal_length_x / image_width,
                        0.0,
                        -(image_width - 2 * principal_point_x) / image_width,
                        0.0,
                    ],
                    [
                        0.0,
                        2 * focal_length_y / image_height,
                        -(image_height - 2 * principal_point_y) / image_height,
                        0.0,
                    ],
                    [
                        0.0,
                        0.0,
                        far_clipping_plane_distance
                        / (far_clipping_plane_distance - near_clipping_plane_distance),
                        -(far_clipping_plane_distance * near_clipping_plane_distance)
                        / (far_clipping_plane_distance - near_clipping_plane_distance),
                    ],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            )
            .cuda()
            .float()
            .unsqueeze(0)
            .transpose(1, 2)
        )
        self.gaussian_rasterization_settings = GaussianRasterizationSettings(
            image_height=image_height,
            image_width=image_width,
            tanfovx=image_width / (2 * focal_length_x),
            tanfovy=image_height / (2 * focal_length_y),
            bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
            scale_modifier=1.0,
            viewmatrix=extrinsic_matrix_tensor,
            projmatrix=extrinsic_matrix_tensor.bmm(opengl_projection_matrix),
            sh_degree=0,
            campos=camera_center,
            prefiltered=False,
        )

    @classmethod
    def from_parameters(
        cls,
        id_: int,
        image_width: int,
        image_height: int,
        near_clipping_plane_distance: float,
        far_clipping_plane_distance: float,
        yaw: float,
        distance_to_center: float,
        height: float,
        aspect_ratio: float,
    ):
        yaw_radians = yaw * math.pi / 180
        extrinsic_matrix = np.array(
            [
                [np.cos(yaw_radians), 0.0, -np.sin(yaw_radians), 0.0],
                [0.0, 1.0, 0.0, height],
                [
                    np.sin(yaw_radians),
                    0.0,
                    np.cos(yaw_radians),
                    distance_to_center,
                ],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        intrinsic_matrix = np.array(
            [
                [aspect_ratio * image_width, 0, image_width / 2],
                [0, aspect_ratio * image_width, image_height / 2],
                [0, 0, 1],
            ]
        )
        return cls(
            id_,
            image_width,
            image_height,
            near_clipping_plane_distance,
            far_clipping_plane_distance,
            intrinsic_matrix,
            extrinsic_matrix,
        )


class Capture:
    def __init__(
        self, camera: Camera, image: torch.Tensor, segmentation_mask: torch.Tensor
    ):
        self.camera = camera
        self.image = image
        self.segmentation_mask = segmentation_mask


class Neighborhoods:
    def __init__(
        self, distances: torch.Tensor, weights: torch.Tensor, indices: torch.Tensor
    ):
        self.distances: torch.Tensor = distances
        self.weights: torch.Tensor = weights
        self.indices: torch.Tensor = indices


class GaussianCloudReferenceState:
    def __init__(self, means: torch.Tensor, rotations: torch.Tensor):
        self.means: torch.Tensor = means
        self.rotations: torch.Tensor = rotations
        self.inverted_foreground_rotations: Optional[torch.Tensor] = None
        self.offsets_to_neighbors: Optional[torch.Tensor] = None
        self.colors: Optional[torch.Tensor] = None


class DensificationVariables:
    def __init__(
        self,
        visibility_count: torch.Tensor,
        mean_2d_gradients_accumulated,
        max_2d_radii,
    ):
        self.visibility_count = visibility_count
        self.mean_2d_gradients_accumulated = mean_2d_gradients_accumulated
        self.max_2d_radii = max_2d_radii
        self.gaussian_is_visible_mask = None
        self.means_2d = None
