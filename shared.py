import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image
from diff_gaussian_rasterization import GaussianRasterizationSettings


@dataclass
class View:
    camera_index: int
    render_settings: GaussianRasterizationSettings
    image: torch.Tensor
    segmentation_mask: torch.Tensor


@dataclass
class DensificationVariables:
    visibility_count: torch.Tensor
    mean_2d_gradients_accumulated: torch.Tensor
    max_2d_radii: torch.Tensor
    gaussian_is_visible_mask: torch.Tensor = None
    means_2d: torch.Tensor = None


def create_render_arguments(gaussian_cloud_parameters: dict[str, torch.nn.Parameter]):
    return {
        "means3D": gaussian_cloud_parameters["means"],
        "colors_precomp": gaussian_cloud_parameters["colors"],
        "rotations": torch.nn.functional.normalize(
            gaussian_cloud_parameters["rotation_quaternions"]
        ),
        "opacities": torch.sigmoid(gaussian_cloud_parameters["opacity_logits"]),
        "scales": torch.exp(gaussian_cloud_parameters["log_scales"]),
        "means2D": torch.zeros_like(
            gaussian_cloud_parameters["means"], requires_grad=True, device="cuda"
        )
        + 0,
    }


def compute_knn_indices_and_squared_distances(numpy_point_cloud: np.ndarray, k: int):
    indices_list = []
    squared_distances_list = []
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(
        np.ascontiguousarray(numpy_point_cloud, np.float64)
    )
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    for point in point_cloud.points:
        # The output of search_knn_vector_3d also includes the query point itself.
        # Therefore, we need to search for k + 1 points and exclude the first element of the output
        [_, neighbor_indices, squared_distances_to_neighbors] = (
            kd_tree.search_knn_vector_3d(point, k + 1)
        )
        indices_list.append(neighbor_indices[1:])
        squared_distances_list.append(squared_distances_to_neighbors[1:])
    return np.array(indices_list), np.array(squared_distances_list)


def create_render_settings(
    image_width: int,
    image_height: int,
    intrinsic_matrix,
    extrinsic_matrix,
    near_clipping_plane_distance: float = 1.0,
    far_clipping_plane_distance: float = 100,
):
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
    return GaussianRasterizationSettings(
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


def load_view(dataset_metadata, timestep, camera_index, sequence_path: Path):
    filename = dataset_metadata["fn"][timestep][camera_index]
    segmentation_mask = (
        torch.tensor(
            np.array(
                copy.deepcopy(
                    Image.open(sequence_path / "seg" / filename.replace(".jpg", ".png"))
                )
            ).astype(np.float32)
        )
        .float()
        .cuda()
    )
    return View(
        camera_index=camera_index,
        render_settings=create_render_settings(
            image_width=dataset_metadata["w"],
            image_height=dataset_metadata["h"],
            intrinsic_matrix=dataset_metadata["k"][timestep][camera_index],
            extrinsic_matrix=dataset_metadata["w2c"][timestep][camera_index],
        ),
        image=torch.tensor(
            np.array(copy.deepcopy(Image.open(sequence_path / "ims" / filename)))
        )
        .float()
        .cuda()
        .permute(2, 0, 1)
        / 255,
        segmentation_mask=torch.stack(
            (
                segmentation_mask,
                torch.zeros_like(segmentation_mask),
                1 - segmentation_mask,
            )
        ),
    )


def load_timestep_views(dataset_metadata, timestep: int, sequence_path: Path):
    views = []
    for camera_index in range(len(dataset_metadata["fn"][timestep])):
        views.append(
            load_view(
                dataset_metadata=dataset_metadata,
                timestep=timestep,
                camera_index=camera_index,
                sequence_path=sequence_path,
            )
        )
    return views
