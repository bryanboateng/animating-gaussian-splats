import argparse
import json
import math
from dataclasses import dataclass, MISSING, fields
from pathlib import Path
from typing import Optional, get_args, Type

import imageio
import numpy as np
import torch
import wandb
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from commons.classes import (
    GaussianCloudReferenceState,
    Neighborhoods,
    GaussianCloudParameters,
    Camera,
)
from commons.helpers import (
    compute_knn_indices_and_squared_distances,
    load_timestep_views,
    load_view,
)
from commons.loss import calculate_loss, GaussianCloud, calculate_image_loss
from deformation_network import (
    update_parameters,
    DeformationNetwork,
    normalize_means_and_rotations,
    PositionalEncoding,
)


@dataclass
class Config:
    sequence_name: str
    data_directory_path: Path
    learning_rate: float
    timestep_count_limit: Optional[int]
    output_directory_path: Path
    iteration_count: int
    warmup_iteration_ratio: float
    fps: int


def parse_config_arguments(config: Config):
    def extract_type(optional_type: Type) -> Type:
        if (
            hasattr(optional_type, "__origin__")
            and optional_type.__origin__ is Optional
        ):
            return get_args(optional_type)[0]
        return optional_type

    argument_parser = argparse.ArgumentParser(prog="Animating Gaussian Splats - Train")
    for field in fields(config):
        non_optional_type = extract_type(field.type)
        if field.default == MISSING:
            argument_parser.add_argument(field.name, type=non_optional_type)
        else:
            if non_optional_type == bool:
                argument_parser.add_argument(f"--{field.name}", action="store_true")
            else:
                argument_parser.add_argument(
                    f"--{field.name}", type=non_optional_type, default=field.default
                )
    args = argument_parser.parse_args()
    for field in fields(config):
        setattr(config, field.name, getattr(args, field.name))


def get_timestep_count(dataset_metadata, timestep_count_limit: int):
    sequence_length = len(dataset_metadata["fn"])
    if timestep_count_limit is None:
        return sequence_length
    else:
        return min(sequence_length, timestep_count_limit)


def get_linear_warmup_cos_annealing(optimizer, warmup_iters, total_iters):
    scheduler_warmup = LinearLR(
        optimizer, start_factor=1 / 1000, total_iters=warmup_iters
    )
    scheduler_cos_decay = CosineAnnealingLR(optimizer, T_max=total_iters - warmup_iters)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_cos_decay],
        milestones=[warmup_iters],
    )

    return scheduler


def load_densified_initial_parameters(data_directory_path: Path, sequence_name: str):
    parameters: GaussianCloudParameters = torch.load(
        data_directory_path
        / sequence_name
        / "densified_initial_gaussian_cloud_parameters.pth",
        map_location="cuda",
    )
    for parameter in parameters.__dict__.values():
        parameter.requires_grad = False
    return parameters


def initialize_post_first_timestep(
    gaussian_cloud_parameters: GaussianCloudParameters,
):

    foreground_mask = gaussian_cloud_parameters.segmentation_colors[:, 0] > 0.5
    foreground_means = gaussian_cloud_parameters.means[foreground_mask]
    neighbor_indices_list, neighbor_squared_distances_list = (
        compute_knn_indices_and_squared_distances(
            foreground_means.detach().cpu().numpy(), 20
        )
    )
    neighborhoods = Neighborhoods(
        indices=(torch.tensor(neighbor_indices_list).cuda().long().contiguous()),
        weights=(
            torch.tensor(np.exp(-2000 * neighbor_squared_distances_list))
            .cuda()
            .float()
            .contiguous()
        ),
        distances=(
            torch.tensor(np.sqrt(neighbor_squared_distances_list))
            .cuda()
            .float()
            .contiguous()
        ),
    )

    previous_timestep_gaussian_cloud_state = GaussianCloudReferenceState(
        means=gaussian_cloud_parameters.means.detach(),
        rotations=torch.nn.functional.normalize(
            gaussian_cloud_parameters.rotation_quaternions
        ).detach(),
    )

    return neighborhoods, previous_timestep_gaussian_cloud_state


def get_inverted_foreground_rotations(
    rotations: torch.Tensor, foreground_mask: torch.Tensor
):
    foreground_rotations = rotations[foreground_mask]
    foreground_rotations[:, 1:] = -1 * foreground_rotations[:, 1:]
    return foreground_rotations


def update_previous_timestep_gaussian_cloud_state(
    gaussian_cloud_parameters: GaussianCloudParameters,
    previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
    neighborhood_indices: torch.Tensor,
):
    current_means = gaussian_cloud_parameters.means
    current_rotations = torch.nn.functional.normalize(
        gaussian_cloud_parameters.rotation_quaternions
    )

    foreground_mask = gaussian_cloud_parameters.segmentation_colors[:, 0] > 0.5
    inverted_foreground_rotations = get_inverted_foreground_rotations(
        current_rotations, foreground_mask
    )
    foreground_means = current_means[foreground_mask]
    offsets_to_neighbors = (
        foreground_means[neighborhood_indices] - foreground_means[:, None]
    )
    previous_timestep_gaussian_cloud_state.inverted_foreground_rotations = (
        inverted_foreground_rotations.detach().clone()
    )
    previous_timestep_gaussian_cloud_state.offsets_to_neighbors = (
        offsets_to_neighbors.detach().clone()
    )
    previous_timestep_gaussian_cloud_state.colors = (
        gaussian_cloud_parameters.rgb_colors.detach().clone()
    )
    previous_timestep_gaussian_cloud_state.means = current_means.detach().clone()
    previous_timestep_gaussian_cloud_state.rotations = (
        current_rotations.detach().clone()
    )


def export_deformation_network(
    run_output_directory_path: Path,
    sequence_name: str,
    initial_gaussian_cloud_parameters: GaussianCloudParameters,
    deformation_network: DeformationNetwork,
    timestep_count: int,
):
    network_directory_path = (
        run_output_directory_path
        / f"deformation_network_{sequence_name}_{wandb.run.name}"
    )
    network_directory_path.mkdir(exist_ok=True)
    parameters_save_path = (
        network_directory_path
        / f"{sequence_name}_densified_initial_gaussian_cloud_parameters.pth"
    )
    torch.save(initial_gaussian_cloud_parameters, parameters_save_path)

    (network_directory_path / "timestep_count").write_text(f"{timestep_count}")

    network_state_dict_path = (
        network_directory_path
        / f"deformation_network_state_dict_{sequence_name}_{wandb.run.name}.pth"
    )
    torch.save(deformation_network.state_dict(), network_state_dict_path)
    wandb.save(
        network_directory_path / "*",
        base_path=network_directory_path.parent,
    )


def create_transformation_matrix(
    yaw_degrees: float, height: float, distance_to_center: float
):
    yaw_radians = np.radians(yaw_degrees)
    return np.array(
        [
            [np.cos(yaw_radians), 0.0, -np.sin(yaw_radians), 0.0],
            [0.0, 1.0, 0.0, height],
            [np.sin(yaw_radians), 0.0, np.cos(yaw_radians), distance_to_center],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def export_visualization(
    timestep_count: int,
    name: str,
    initial_gaussian_cloud_parameters,
    deformation_network: DeformationNetwork,
    pos_smol: PositionalEncoding,
    means_norm: torch.Tensor,
    rotations_norm: torch.Tensor,
    aspect_ratio: float,
    extrinsic_matrix: np.array,
    visualizations_directory_path: Path,
    sequence_name: str,
    fps: int,
):
    render_images = []
    for timestep in tqdm(range(timestep_count), desc=f"Creating Visualization {name}"):
        if timestep == 0:
            timestep_gaussian_cloud_parameters = initial_gaussian_cloud_parameters

        else:
            timestep_gaussian_cloud_parameters = update_parameters(
                deformation_network=deformation_network,
                positional_encoding=pos_smol,
                normalized_means=means_norm,
                normalized_rotations=rotations_norm,
                parameters=initial_gaussian_cloud_parameters,
                timestep=timestep,
                timestep_count=timestep_count,
            )

        gaussian_cloud = GaussianCloud(parameters=timestep_gaussian_cloud_parameters)

        image_width = 1280
        image_height = 720
        camera = Camera(
            id_=0,
            image_width=image_width,
            image_height=image_height,
            near_clipping_plane_distance=1,
            far_clipping_plane_distance=100,
            intrinsic_matrix=np.array(
                [
                    [aspect_ratio * image_width, 0, image_width / 2],
                    [0, aspect_ratio * image_width, image_height / 2],
                    [0, 0, 1],
                ]
            ),
            extrinsic_matrix=extrinsic_matrix,
        )
        (
            image,
            _,
            _,
        ) = Renderer(
            raster_settings=camera.gaussian_rasterization_settings
        )(**gaussian_cloud.get_renderer_format())
        render_images.append(
            (
                255
                * np.clip(
                    image.cpu().numpy(),
                    0,
                    1,
                )
            )
            .astype(np.uint8)
            .transpose(1, 2, 0)
        )
    rendered_sequence_path = (
        visualizations_directory_path / f"{sequence_name}_{name}_{wandb.run.name}.mp4"
    )
    imageio.mimwrite(
        rendered_sequence_path,
        render_images,
        fps=fps,
    )


def export_visualizations(
    run_output_directory_path: Path,
    sequence_name: str,
    initial_gaussian_cloud_parameters: GaussianCloudParameters,
    deformation_network: DeformationNetwork,
    timestep_count: int,
    fps: int,
):
    visualizations_directory_path = (
        run_output_directory_path / f"visualizations_{sequence_name}_{wandb.run.name}"
    )
    visualizations_directory_path.mkdir(exist_ok=True)

    deformation_network.eval()

    means_norm, pos_smol, rotations_norm = normalize_means_and_rotations(
        initial_gaussian_cloud_parameters
    )

    distance_to_center: float = 2.4
    height: float = 1.3
    extrinsic_matrices = {
        "000": (
            create_transformation_matrix(
                yaw_degrees=0, height=height, distance_to_center=distance_to_center
            ),
            0.82,
        ),
        "090": (
            create_transformation_matrix(
                yaw_degrees=90, height=height, distance_to_center=distance_to_center
            ),
            0.52,
        ),
        "180": (
            create_transformation_matrix(
                yaw_degrees=180, height=height, distance_to_center=distance_to_center
            ),
            0.52,
        ),
        "270": (
            create_transformation_matrix(
                yaw_degrees=270, height=height, distance_to_center=distance_to_center
            ),
            0.52,
        ),
        "top": (
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0],
                    [0.0, 1.0, 0.0, 4.5],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            ),
            0.35,
        ),
    }
    for name, (extrinsic_matrix, aspect_ratio) in extrinsic_matrices.items():
        export_visualization(
            timestep_count=timestep_count,
            name=name,
            initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
            deformation_network=deformation_network,
            pos_smol=pos_smol,
            means_norm=means_norm,
            rotations_norm=rotations_norm,
            aspect_ratio=aspect_ratio,
            extrinsic_matrix=extrinsic_matrix,
            visualizations_directory_path=visualizations_directory_path,
            sequence_name=sequence_name,
            fps=fps,
        )
    wandb.save(
        visualizations_directory_path / "*",
        base_path=visualizations_directory_path.parent,
    )


def train(config: Config):
    wandb.init(project="animating-gaussian-splats")
    dataset_metadata_file_path = (
        config.data_directory_path / config.sequence_name / "train_meta.json"
    )
    with dataset_metadata_file_path.open() as file:
        dataset_metadata = json.load(file)

    timestep_count = get_timestep_count(
        dataset_metadata=dataset_metadata,
        timestep_count_limit=config.timestep_count_limit,
    )
    deformation_network = DeformationNetwork().cuda()
    optimizer = torch.optim.Adam(
        params=deformation_network.parameters(), lr=config.learning_rate
    )
    scheduler = get_linear_warmup_cos_annealing(
        optimizer,
        warmup_iters=int(config.iteration_count * config.warmup_iteration_ratio),
        total_iters=config.iteration_count,
    )

    initial_gaussian_cloud_parameters = load_densified_initial_parameters(
        data_directory_path=config.data_directory_path,
        sequence_name=config.sequence_name,
    )

    (
        initial_neighborhoods,
        previous_timestep_gaussian_cloud_state,
    ) = initialize_post_first_timestep(
        gaussian_cloud_parameters=initial_gaussian_cloud_parameters
    )
    normalized_means, pos_smol, normalized_rotations = normalize_means_and_rotations(
        initial_gaussian_cloud_parameters
    )
    camera_count = len(dataset_metadata["fn"][0])
    for i in tqdm(range(config.iteration_count), desc="Training"):
        timestep = i % timestep_count
        camera_index = torch.randint(0, camera_count, ())

        if timestep == 0:
            update_previous_timestep_gaussian_cloud_state(
                gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
                previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
                neighborhood_indices=initial_neighborhoods.indices,
            )

        view = load_view(
            dataset_metadata=dataset_metadata,
            timestep=timestep,
            camera_index=camera_index,
            sequence_path=config.data_directory_path / config.sequence_name,
        )
        updated_gaussian_cloud_parameters = update_parameters(
            deformation_network=deformation_network,
            positional_encoding=pos_smol,
            normalized_means=normalized_means,
            normalized_rotations=normalized_rotations,
            parameters=initial_gaussian_cloud_parameters,
            timestep=timestep,
            timestep_count=timestep_count,
        )

        total_loss, l1_loss, ssim_loss, image_loss, rigidity_loss = calculate_loss(
            gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
            target_view=view,
            initial_neighborhoods=initial_neighborhoods,
            previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
            rigidity_loss_weight=(
                2.0 / (1.0 + math.exp(-6 * (i / config.iteration_count))) - 1
            ),
        )
        wandb.log(
            {
                f"total-loss": image_loss.item(),
                f"l1-loss": l1_loss.item(),
                f"ssim-loss": ssim_loss.item(),
                f"image-loss": image_loss.item(),
                f"rigidity-loss": rigidity_loss.item(),
                f"learning_rate": optimizer.param_groups[0]["lr"],
            }
        )
        update_previous_timestep_gaussian_cloud_state(
            gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
            previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
            neighborhood_indices=initial_neighborhoods.indices,
        )

        total_loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    for timestep in tqdm(
        range(timestep_count), desc="Calculate Mean Image Loss per Timestep"
    ):
        image_losses = []
        with torch.no_grad():
            updated_gaussian_cloud_parameters = update_parameters(
                deformation_network=deformation_network,
                positional_encoding=pos_smol,
                normalized_means=normalized_means,
                normalized_rotations=normalized_rotations,
                parameters=initial_gaussian_cloud_parameters,
                timestep=timestep,
                timestep_count=timestep_count,
            )
            timestep_views = load_timestep_views(
                dataset_metadata,
                timestep,
                config.data_directory_path / config.sequence_name,
            )
            for view in timestep_views:
                _, _, image_loss = calculate_image_loss(
                    gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
                    target_view=view,
                )
                image_losses.append(image_loss.item())
        wandb.log(
            {f"mean-image-loss": sum(image_losses) / len(image_losses)},
            step=config.iteration_count + timestep,
        )
    with torch.no_grad():
        run_output_directory_path = (
            config.output_directory_path / f"{config.sequence_name}_{wandb.run.name}"
        )
        export_deformation_network(
            run_output_directory_path=run_output_directory_path,
            sequence_name=config.sequence_name,
            initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
            deformation_network=deformation_network,
            timestep_count=timestep_count,
        )
        export_visualizations(
            run_output_directory_path=run_output_directory_path,
            sequence_name=config.sequence_name,
            initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
            deformation_network=deformation_network,
            timestep_count=timestep_count,
            fps=config.fps,
        )


def main():
    argument_parser = argparse.ArgumentParser(prog="Animating Gaussian Splats")
    argument_parser.add_argument("sequence_name", metavar="sequence-name", type=str)
    argument_parser.add_argument(
        "data_directory_path", metavar="data-directory-path", type=Path
    )
    argument_parser.add_argument("-lr", "--learning-rate", type=float, default=0.01)
    argument_parser.add_argument("-t", "--timestep-count-limit", type=int)
    argument_parser.add_argument(
        "-o",
        "--output-directory-path",
        type=Path,
        default=Path("./deformation_networks"),
    )
    argument_parser.add_argument("-is", "--iteration_count", type=int, default=200_000)
    argument_parser.add_argument(
        "-wr", "--warmup_iteration_ratio", type=float, default=0.075
    )
    argument_parser.add_argument("--fps", type=int, default=30)
    args = argument_parser.parse_args()
    config = Config(
        sequence_name=args.sequence_name,
        data_directory_path=args.data_directory_path,
        learning_rate=args.learning_rate,
        timestep_count_limit=args.timestep_count_limit,
        output_directory_path=args.output_directory_path,
        iteration_count=args.iteration_count,
        warmup_iteration_ratio=args.warmup_iteration_ratio,
        fps=args.fps,
    )
    train(config=config)


if __name__ == "__main__":
    main()
