import math

import torch
from typing import Optional
import numpy as np
from diff_gaussian_rasterization import GaussianRasterizationSettings


class GaussianCloudParameterNames:
    means = "means3D"
    colors = "rgb_colors"
    segmentation_masks = "seg_colors"
    rotation_quaternions = "unnorm_rotations"
    opacities_logits = "logit_opacities"
    log_scales = "log_scales"
    camera_matrices = "cam_m"
    camera_centers = "cam_c"


def l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


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


class Variables:
    def __init__(
        self,
        max_2D_radius,
        scene_radius: float,
        initial_background_points=None,
        initial_background_rotations=None,
        neighbor_indices_list: Optional[torch.Tensor] = None,
        neighbor_weight=None,
        neighbor_distances_list: Optional[torch.Tensor] = None,
        means2D_gradient_accum=None,
        seen=None,
        denom: Optional[torch.Tensor] = None,
        previous_rotations=None,
        previous_inverted_foreground_rotations=None,
        previous_offsets_to_neighbors=None,
        previous_colors=None,
        previous_means=None,
    ):
        self.initial_background_means = initial_background_points
        self.initial_background_rotations = initial_background_rotations
        self.neighbor_indices_list = neighbor_indices_list
        self.neighbor_weight = neighbor_weight
        self.neighbor_distances_list = neighbor_distances_list
        self.external_means2D = None
        self.external_means2D_gradient_accum = means2D_gradient_accum
        self.external_max_2D_radius = max_2D_radius
        self.external_scene_radius = scene_radius
        self.external_seen = seen
        self.external_denom = denom
        self.previous_rotations = previous_rotations
        self.previous_inverted_foreground_rotations = (
            previous_inverted_foreground_rotations
        )
        self.previous_offsets_to_neighbors = previous_offsets_to_neighbors
        self.previous_colors = previous_colors
        self.previous_means = previous_means
