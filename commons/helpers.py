import copy
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
from PIL import Image

from commons.classes import (
    View,
    Camera,
)


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


def load_view(
    dataset_metadata,
    timestep,
    camera_index,
    sequence_path: Path,
):
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
        camera=Camera(
            id_=camera_index,
            image_width=dataset_metadata["w"],
            image_height=dataset_metadata["h"],
            near_clipping_plane_distance=1,
            far_clipping_plane_distance=100,
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
    timestep_data = []
    for camera_index in range(len(dataset_metadata["fn"][timestep])):
        timestep_data.append(
            load_view(
                dataset_metadata=dataset_metadata,
                timestep=timestep,
                camera_index=camera_index,
                sequence_path=sequence_path,
            )
        )
    return timestep_data
