import argparse
import json
from pathlib import Path
from random import randint

import numpy as np
import torch
import wandb
from tqdm import tqdm

from commons.classes import (
    DensificationVariables,
    GaussianCloudParameters,
)
from commons.helpers import (
    compute_knn_indices_and_squared_distances,
    load_timestep_views,
)
from commons.loss import calculate_image_and_segmentation_loss
from external import densify_gaussians


def create_trainable_parameter(v):
    return torch.nn.Parameter(
        torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
    )


def initialize_parameters(sequence_path: Path):
    initial_point_cloud = np.load(sequence_path / "init_pt_cld.npz")["data"]
    segmentation_masks = initial_point_cloud[:, 6]
    camera_count_limit = 50
    _, squared_distances = compute_knn_indices_and_squared_distances(
        numpy_point_cloud=initial_point_cloud[:, :3], k=3
    )
    return GaussianCloudParameters(
        means=create_trainable_parameter(initial_point_cloud[:, :3]),
        rgb_colors=create_trainable_parameter(initial_point_cloud[:, 3:6]),
        segmentation_colors=create_trainable_parameter(
            np.stack(
                (
                    segmentation_masks,
                    np.zeros_like(segmentation_masks),
                    1 - segmentation_masks,
                ),
                -1,
            )
        ),
        rotation_quaternions=create_trainable_parameter(
            np.tile([1, 0, 0, 0], (segmentation_masks.shape[0], 1))
        ),
        opacities_logits=create_trainable_parameter(
            np.zeros((segmentation_masks.shape[0], 1))
        ),
        log_scales=create_trainable_parameter(
            np.tile(
                np.log(np.sqrt(squared_distances.mean(-1).clip(min=0.0000001)))[
                    ..., None
                ],
                (1, 3),
            )
        ),
        camera_matrices=create_trainable_parameter(np.zeros((camera_count_limit, 3))),
        camera_centers=create_trainable_parameter(np.zeros((camera_count_limit, 3))),
    )


def get_scene_radius(dataset_metadata):
    camera_centers = np.linalg.inv(dataset_metadata["w2c"][0])[:, :3, 3]
    scene_radius = 1.1 * np.max(
        np.linalg.norm(camera_centers - np.mean(camera_centers, 0)[None], axis=-1)
    )
    return scene_radius


def create_optimizer(parameters: GaussianCloudParameters, scene_radius: float):
    return torch.optim.Adam(
        [
            {
                "params": [parameters.means],
                "name": "means",
                "lr": 0.00016 * scene_radius,
            },
            {
                "params": [parameters.rgb_colors],
                "name": "rgb_colors",
                "lr": 0.0025,
            },
            {
                "params": [parameters.segmentation_colors],
                "name": "segmentation_colors",
                "lr": 0.0,
            },
            {
                "params": [parameters.rotation_quaternions],
                "name": "rotation_quaternions",
                "lr": 0.001,
            },
            {
                "params": [parameters.opacities_logits],
                "name": "opacities_logits",
                "lr": 0.05,
            },
            {
                "params": [parameters.log_scales],
                "name": "log_scales",
                "lr": 0.001,
            },
            {
                "params": [parameters.camera_matrices],
                "name": "camera_matrices",
                "lr": 1e-4,
            },
            {
                "params": [parameters.camera_centers],
                "name": "camera_centers",
                "lr": 1e-4,
            },
        ],
        lr=0.0,
        eps=1e-15,
    )


def create_densification_variables(
    gaussian_cloud_parameters: GaussianCloudParameters,
):
    densification_variables = DensificationVariables(
        visibility_count=torch.zeros(gaussian_cloud_parameters.means.shape[0])
        .cuda()
        .float(),
        mean_2d_gradients_accumulated=torch.zeros(
            gaussian_cloud_parameters.means.shape[0]
        )
        .cuda()
        .float(),
        max_2d_radii=torch.zeros(gaussian_cloud_parameters.means.shape[0])
        .cuda()
        .float(),
    )
    return densification_variables


def get_random_element(input_list, fallback_list):
    if not input_list:
        input_list = fallback_list.copy()
    return input_list.pop(randint(0, len(input_list) - 1))


def save_and_log_checkpoint(
    sequence_path: Path,
    initial_gaussian_cloud_parameters: GaussianCloudParameters,
):
    parameters_save_path = (
        sequence_path / "densified_initial_gaussian_cloud_parameters.pth"
    )
    torch.save(initial_gaussian_cloud_parameters, parameters_save_path)


def densify(sequence_path: Path):
    wandb.init(project="animating-gaussian-splats")
    dataset_metadata_file_path = sequence_path / "train_meta.json"
    with dataset_metadata_file_path.open() as file:
        dataset_metadata = json.load(file)

    parameters = initialize_parameters(sequence_path=sequence_path)
    scene_radius = get_scene_radius(dataset_metadata=dataset_metadata)
    optimizer = create_optimizer(parameters=parameters, scene_radius=scene_radius)
    densification_variables = create_densification_variables(parameters)
    timestep = 0
    timestep_views = load_timestep_views(
        dataset_metadata=dataset_metadata,
        timestep=timestep,
        sequence_path=sequence_path,
    )
    timestep_view_buffer = []
    for i in tqdm(range(30_000), desc=f"timestep {timestep}"):
        view = get_random_element(
            input_list=timestep_view_buffer, fallback_list=timestep_views
        )
        loss, densification_variables = calculate_image_and_segmentation_loss(
            gaussian_cloud_parameters=parameters,
            target_view=view,
            densification_variables=densification_variables,
        )
        wandb.log(
            {
                f"timestep-{timestep}-loss": loss.item(),
                "gaussian_count": parameters.means.shape[0],
            }
        )
        loss.backward()
        with torch.no_grad():
            densify_gaussians(
                gaussian_cloud_parameters=parameters,
                densification_variables=densification_variables,
                scene_radius=scene_radius,
                optimizer=optimizer,
                i=i,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    save_and_log_checkpoint(
        sequence_path=sequence_path, initial_gaussian_cloud_parameters=parameters
    )


def main():
    argument_parser = argparse.ArgumentParser(
        prog="Animating Gaussian Splats - Densify"
    )
    argument_parser.add_argument("sequence_path", type=Path)
    args = argument_parser.parse_args()
    densify(sequence_path=args.sequence_path)


if __name__ == "__main__":
    main()
