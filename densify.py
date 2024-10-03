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
    DensificationVariables,
    View,
    create_render_arguments,
)


def initialize_parameters(sequence_path: Path):
    initial_point_cloud = np.load(sequence_path / "init_pt_cld.npz")["data"]
    segmentation_masks = initial_point_cloud[:, 6]
    camera_count_limit = 50
    _, squared_distances = compute_knn_indices_and_squared_distances(
        numpy_point_cloud=initial_point_cloud[:, :3], k=3
    )
    return {
        k: torch.nn.Parameter(
            torch.tensor(v).cuda().float().contiguous().requires_grad_(True)
        )
        for k, v in {
            "means": initial_point_cloud[:, :3],
            "colors": initial_point_cloud[:, 3:6],
            "segmentation_masks": np.stack(
                (
                    segmentation_masks,
                    np.zeros_like(segmentation_masks),
                    1 - segmentation_masks,
                ),
                -1,
            ),
            "rotation_quaternions": np.tile(
                [1, 0, 0, 0], (segmentation_masks.shape[0], 1)
            ),
            "opacity_logits": np.zeros((segmentation_masks.shape[0], 1)),
            "log_scales": np.tile(
                np.log(np.sqrt(squared_distances.mean(-1).clip(min=0.0000001)))[
                    ..., None
                ],
                (1, 3),
            ),
            "camera_matrices": np.zeros((camera_count_limit, 3)),
            "camera_center": np.zeros((camera_count_limit, 3)),
        }.items()
    }


def get_scene_radius(dataset_metadata):
    camera_centers = np.linalg.inv(dataset_metadata["w2c"][0])[:, :3, 3]
    scene_radius = 1.1 * np.max(
        np.linalg.norm(camera_centers - np.mean(camera_centers, 0)[None], axis=-1)
    )
    return scene_radius


def create_optimizer(parameters: dict[str, torch.nn.Parameter], scene_radius: float):
    learning_rates = {
        "means": 0.00016 * scene_radius,
        "colors": 0.0025,
        "segmentation_masks": 0.0,
        "rotation_quaternions": 0.001,
        "opacity_logits": 0.05,
        "log_scales": 0.001,
        "camera_matrices": 1e-4,
        "camera_center": 1e-4,
    }
    return torch.optim.Adam(
        [
            {"params": [value], "name": name, "lr": learning_rates[name]}
            for name, value in parameters.items()
        ],
        lr=0.0,
        eps=1e-15,
    )


def create_densification_variables(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter]
):
    densification_variables = DensificationVariables(
        visibility_count=torch.zeros(gaussian_cloud_parameters["means"].shape[0])
        .cuda()
        .float(),
        mean_2d_gradients_accumulated=torch.zeros(
            gaussian_cloud_parameters["means"].shape[0]
        )
        .cuda()
        .float(),
        max_2d_radii=torch.zeros(gaussian_cloud_parameters["means"].shape[0])
        .cuda()
        .float(),
    )
    return densification_variables


def get_random_element(input_list, fallback_list):
    if not input_list:
        input_list = fallback_list.copy()
    return input_list.pop(randint(0, len(input_list) - 1))


def calculate_image_loss(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
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
    loss = 0.8 * torch.nn.functional.l1_loss(
        rendered_image, target_view.image
    ) + 0.2 * (1.0 - calc_ssim(rendered_image, target_view.image))
    means_2d = render_arguments[
        "means2D"
    ]  # Gradient only accum from colour render for densification
    return loss, means_2d, radii


def calculate_segmentation_loss(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_view: View,
):
    render_arguments = create_render_arguments(gaussian_cloud_parameters)
    render_arguments["colors_precomp"] = gaussian_cloud_parameters["segmentation_masks"]
    (
        segmentation_mask,
        _,
        _,
    ) = Renderer(
        raster_settings=target_view.render_settings
    )(**render_arguments)
    return 0.8 * torch.nn.functional.l1_loss(
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


def calculate_image_and_segmentation_loss(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_view: View,
    densification_variables: DensificationVariables,
):
    image_loss, means_2d, radii = calculate_image_loss(
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_view=target_view,
    )
    segmentation_loss = calculate_segmentation_loss(
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_view=target_view,
    )
    densification_variables.means_2d = means_2d
    update_max_2d_radii_and_visibility_mask(
        radii=radii, densification_variables=densification_variables
    )
    return (
        image_loss + 3 * segmentation_loss,
        image_loss,
        segmentation_loss,
        densification_variables,
    )


def export_parameters(
    sequence_path: Path,
    parameters: dict[str, torch.nn.Parameter],
):
    parameters_save_path = (
        sequence_path / "densified_initial_gaussian_cloud_parameters.pth"
    )
    torch.save(parameters, parameters_save_path)
    wandb.save(parameters_save_path)


def densify(sequence_path: Path):
    wandb.init(project="densify-gaussian-cloud")
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
    for i in tqdm(range(30_000), desc="Densifying"):
        view = get_random_element(
            input_list=timestep_view_buffer, fallback_list=timestep_views
        )
        total_loss, image_loss, segmentation_loss, densification_variables = (
            calculate_image_and_segmentation_loss(
                gaussian_cloud_parameters=parameters,
                target_view=view,
                densification_variables=densification_variables,
            )
        )
        wandb.log(
            {
                "loss/image": image_loss,
                "loss/segmentation": segmentation_loss,
                "loss/total": total_loss.item(),
                "gaussian_count": parameters["means"].shape[0],
            }
        )
        total_loss.backward()
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
    export_parameters(sequence_path=sequence_path, parameters=parameters)


def main():
    argument_parser = argparse.ArgumentParser(prog="Densify Gaussian Cloud")
    argument_parser.add_argument("sequence_path", type=Path)
    args = argument_parser.parse_args()
    densify(sequence_path=args.sequence_path)


if __name__ == "__main__":
    main()
