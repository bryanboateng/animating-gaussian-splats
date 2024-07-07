import copy
import os
from random import randint

import numpy as np
import torch
from PIL import Image

from helpers import Camera


class Capture:
    def __init__(
        self, camera: Camera, image: torch.Tensor, segmentation_mask: torch.Tensor
    ):
        self.camera = camera
        self.image = image
        self.segmentation_mask = segmentation_mask


def load_timestep_captures(
    dataset_metadata, timestamp: int, data_directory_path: str, sequence_name: str
):
    timestep_data = []
    for camera_index in range(len(dataset_metadata["fn"][timestamp])):
        filename = dataset_metadata["fn"][timestamp][camera_index]
        segmentation_mask = (
            torch.tensor(
                np.array(
                    copy.deepcopy(
                        Image.open(
                            os.path.join(
                                data_directory_path,
                                sequence_name,
                                "seg",
                                filename.replace(".jpg", ".png"),
                            )
                        )
                    )
                ).astype(np.float32)
            )
            .float()
            .cuda()
        )
        timestep_data.append(
            Capture(
                camera=Camera(
                    id_=camera_index,
                    image_width=dataset_metadata["w"],
                    image_height=dataset_metadata["h"],
                    near_clipping_plane_distance=1,
                    far_clipping_plane_distance=100,
                    intrinsic_matrix=dataset_metadata["k"][timestamp][camera_index],
                    extrinsic_matrix=dataset_metadata["w2c"][timestamp][camera_index],
                ),
                image=torch.tensor(
                    np.array(
                        copy.deepcopy(
                            Image.open(
                                os.path.join(
                                    data_directory_path,
                                    sequence_name,
                                    "ims",
                                    filename,
                                )
                            )
                        )
                    )
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
        )
    return timestep_data


def get_random_element(input_list, fallback_list):
    if not input_list:
        input_list = fallback_list.copy()
    return input_list.pop(randint(0, len(input_list) - 1))
