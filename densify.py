import argparse
import json
from pathlib import Path
from random import randint

import numpy as np
import torch
import wandb
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from tqdm import tqdm

from external import densify_gaussians, calc_ssim
from shared import (
    compute_knn_indices_and_squared_distances,
    load_timestep_views,
    GaussianCloudParameters,
    DensificationVariables,
    View,
    l1_loss_v1,
    apply_exponential_transform_and_center_to_image,
    create_render_arguments,
)


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


def add_image_loss_grad(
    losses: dict[str, torch.Tensor],
    gaussian_cloud_parameters: GaussianCloudParameters,
    target_view: View,
):
    render_arguments = create_render_arguments(gaussian_cloud_parameters)
    render_arguments["means2D"].retain_grad()
    (
        rendered_image,
        radii,
        _,
    ) = Renderer(
        raster_settings=target_view.render_settings
    )(**render_arguments)
    image = apply_exponential_transform_and_center_to_image(
        rendered_image, gaussian_cloud_parameters, target_view.camera_index
    )
    losses["im"] = 0.8 * l1_loss_v1(image, target_view.image) + 0.2 * (
        1.0 - calc_ssim(image, target_view.image)
    )
    means_2d = render_arguments[
        "means2D"
    ]  # Gradient only accum from colour render for densification
    return means_2d, radii


def add_segmentation_loss(
    losses: dict[str, torch.Tensor],
    gaussian_cloud_parameters: GaussianCloudParameters,
    target_view: View,
):
    render_arguments = create_render_arguments(gaussian_cloud_parameters)
    render_arguments["colors_precomp"] = gaussian_cloud_parameters.segmentation_colors
    (
        segmentation_mask,
        _,
        _,
    ) = Renderer(
        raster_settings=target_view.render_settings
    )(**render_arguments)
    losses["seg"] = 0.8 * l1_loss_v1(
        segmentation_mask, target_view.segmentation_mask
    ) + 0.2 * (1.0 - calc_ssim(segmentation_mask, target_view.segmentation_mask))


def update_max_2d_radii_and_visibility_mask(
    radii, densification_variables: DensificationVariables
):
    radius_is_positive = radii > 0
    densification_variables.max_2d_radii[radius_is_positive] = torch.max(
        radii[radius_is_positive],
        densification_variables.max_2d_radii[radius_is_positive],
    )
    densification_variables.gaussian_is_visible_mask = radius_is_positive


def combine_losses(losses: dict[str, torch.Tensor]):
    loss_weights = {
        "im": 1.0,
        "seg": 3.0,
        "rigid": 4.0,
        "rot": 4.0,
        "iso": 2.0,
        "floor": 2.0,
        "bg": 20.0,
        "soft_col_cons": 0.01,
    }
    weighted_losses = []
    for name, value in losses.items():
        weighted_loss = torch.tensor(loss_weights[name]) * value
        wandb.log({f"{name}-loss": weighted_loss})
        weighted_losses.append(weighted_loss)
    return torch.sum(torch.stack(weighted_losses))


def calculate_image_and_segmentation_loss(
    gaussian_cloud_parameters: GaussianCloudParameters,
    target_view: View,
    densification_variables: DensificationVariables,
):
    losses = {}
    means_2d, radii = add_image_loss_grad(
        losses=losses,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_view=target_view,
    )
    add_segmentation_loss(
        losses=losses,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_view=target_view,
    )
    densification_variables.means_2d = means_2d
    update_max_2d_radii_and_visibility_mask(
        radii=radii, densification_variables=densification_variables
    )
    return combine_losses(losses), densification_variables


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
    argument_parser = argparse.ArgumentParser(prog="Densify Gaussian Cloud")
    argument_parser.add_argument("sequence_path", type=Path)
    args = argument_parser.parse_args()
    densify(sequence_path=args.sequence_path)


if __name__ == "__main__":
    main()
