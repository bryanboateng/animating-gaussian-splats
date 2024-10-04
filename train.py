import argparse
import copy
import json
import math
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
import wandb
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

from external import calc_ssim, build_rotation
from shared import (
    compute_knn_indices_and_squared_distances,
    load_timestep_views,
    View,
    create_render_arguments,
    create_render_settings,
)


@dataclass
class Config:
    sequence_name: str
    data_directory_path: Path
    hidden_dimension: int
    hidden_layer_count: int
    learning_rate: float
    timestep_count_limit: Optional[int]
    output_directory_path: Path
    iteration_count: int
    fps: int


@dataclass
class NeighborInfo:
    weights: torch.Tensor
    indices: torch.Tensor


@dataclass
class ForegroundInfo:
    inverted_rotations: Optional[torch.Tensor] = None
    offsets_to_neighbors: Optional[torch.Tensor] = None


class SineLayer(nn.Module):
    def __init__(
        self, input_dimension: int, output_dimension: int, is_first: bool, omega_0: int
    ):
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.input_dimension = input_dimension
        self.linear = nn.Linear(input_dimension, output_dimension)

        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(
                    -1 / self.input_dimension, 1 / self.input_dimension
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.input_dimension) / self.omega_0,
                    np.sqrt(6 / self.input_dimension) / self.omega_0,
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class DeformationNetwork(nn.Module):
    def __init__(self, hidden_dimension, hidden_layer_count) -> None:
        super(DeformationNetwork, self).__init__()

        first_omega_0 = 30
        hidden_omega_0 = 30

        mean_dimension = 3
        rotation_dimension = 4
        progress_dimension = 1
        layers = [
            SineLayer(
                mean_dimension + rotation_dimension + progress_dimension,
                hidden_dimension,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]
        for _ in range(hidden_layer_count):
            layers.append(
                SineLayer(
                    hidden_dimension,
                    hidden_dimension,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )
        fc_out = nn.Linear(hidden_dimension, mean_dimension + rotation_dimension)
        with torch.no_grad():
            fc_out.weight.uniform_(
                -np.sqrt(6 / hidden_dimension) / hidden_omega_0,
                np.sqrt(6 / hidden_dimension) / hidden_omega_0,
            )
        layers.append(fc_out)

        self.layers = nn.Sequential(*layers)

    def forward(
        self,
        initial_means: torch.Tensor,
        initial_rotations: torch.Tensor,
        normalized_initial_means: torch.Tensor,
        normalized_initial_rotations: torch.Tensor,
        progress: float,
    ):
        out = self.layers(
            torch.cat(
                (
                    (
                        torch.cat(
                            (normalized_initial_means, normalized_initial_rotations),
                            dim=1,
                        )
                    ),
                    (
                        torch.tensor(progress)
                        .view(1, 1)
                        .repeat(normalized_initial_means.shape[0], 1)
                        .cuda()
                    ),
                ),
                dim=1,
            )
        )
        out += torch.cat(
            (initial_means, initial_rotations),
            dim=1,
        )

        return out


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
    parameters: dict[str, torch.nn.Parameter] = torch.load(
        data_directory_path
        / sequence_name
        / "densified_initial_gaussian_cloud_parameters.pth"
    )
    for parameter in parameters.values():
        parameter.requires_grad = False
    return parameters


def initialize_variables(gaussian_cloud_parameters: dict[str, torch.nn.Parameter]):
    foreground_mask = gaussian_cloud_parameters["segmentation_masks"][:, 0] > 0.5
    foreground_means = gaussian_cloud_parameters["means"][foreground_mask]
    neighbor_indices_list, neighbor_squared_distances_list = (
        compute_knn_indices_and_squared_distances(
            foreground_means.detach().cpu().numpy(), 20
        )
    )
    neighbor_info = NeighborInfo(
        indices=(torch.tensor(neighbor_indices_list).cuda().long().contiguous()),
        weights=(
            torch.tensor(np.exp(-2000 * neighbor_squared_distances_list))
            .cuda()
            .float()
            .contiguous()
        ),
    )

    previous_timestep_foreground_info = ForegroundInfo()

    return neighbor_info, previous_timestep_foreground_info


def normalize_means_and_rotations(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter]
):
    means = gaussian_cloud_parameters["means"]
    rotations = gaussian_cloud_parameters["rotation_quaternions"]
    normalized_means = means - means.min(dim=0).values
    normalized_means = (
        2.0 * normalized_means / normalized_means.max(dim=0).values
    ) - 1.0
    normalized_rotations = rotations - rotations.min(dim=0).values
    normalized_rotations = (
        2.0 * normalized_rotations / normalized_rotations.max(dim=0).values
    ) - 1.0
    return normalized_means, normalized_rotations


def load_all_views(dataset_metadata, timestep_count: int, sequence_path: Path):
    views = []
    for timestep in range(1, timestep_count):
        views += [
            load_timestep_views(
                dataset_metadata=dataset_metadata,
                timestep=timestep,
                sequence_path=sequence_path,
            )
        ]
    return views


def get_inverted_foreground_rotations(
    rotations: torch.Tensor, foreground_mask: torch.Tensor
):
    foreground_rotations = rotations[foreground_mask]
    foreground_rotations[:, 1:] = -1 * foreground_rotations[:, 1:]
    return foreground_rotations


def update_previous_timestep_foreground_info(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    previous_timestep_foreground_info: ForegroundInfo,
    initial_neighbor_indices: torch.Tensor,
):
    current_means = gaussian_cloud_parameters["means"]
    current_rotations = torch.nn.functional.normalize(
        gaussian_cloud_parameters["rotation_quaternions"]
    )

    foreground_mask = gaussian_cloud_parameters["segmentation_masks"][:, 0] > 0.5
    inverted_foreground_rotations = get_inverted_foreground_rotations(
        current_rotations, foreground_mask
    )
    foreground_means = current_means[foreground_mask]
    offsets_to_foreground_neighbors = (
        foreground_means[initial_neighbor_indices] - foreground_means[:, None]
    )
    previous_timestep_foreground_info.inverted_rotations = (
        inverted_foreground_rotations.detach().clone()
    )
    previous_timestep_foreground_info.offsets_to_neighbors = (
        offsets_to_foreground_neighbors.detach().clone()
    )


def update_gaussian_cloud_parameters(
    deformation_network: DeformationNetwork,
    initial_gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    normalized_initial_means,
    normalized_initial_rotations,
    timestep,
    timestep_count,
):
    delta = deformation_network(
        initial_means=initial_gaussian_cloud_parameters["means"],
        initial_rotations=initial_gaussian_cloud_parameters["rotation_quaternions"],
        normalized_initial_means=normalized_initial_means,
        normalized_initial_rotations=normalized_initial_rotations,
        progress=timestep / (timestep_count - 1),
    )
    means_delta = delta[:, :3]
    rotations_delta = delta[:, 3:]
    updated_gaussian_cloud_parameters: dict[str, torch.nn.Parameter] = copy.deepcopy(
        initial_gaussian_cloud_parameters
    )
    updated_gaussian_cloud_parameters["means"] = updated_gaussian_cloud_parameters[
        "means"
    ].detach()
    updated_gaussian_cloud_parameters["means"] += means_delta * 0.01
    updated_gaussian_cloud_parameters["rotation_quaternions"] = (
        updated_gaussian_cloud_parameters["rotation_quaternions"].detach()
    )
    updated_gaussian_cloud_parameters["rotation_quaternions"] += rotations_delta * 0.01
    return updated_gaussian_cloud_parameters


def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def calculate_rigidity_loss(
    gaussian_cloud_parameters, initial_neighbor_info, previous_timestep_foreground_info
):
    foreground_mask = (
        gaussian_cloud_parameters["segmentation_masks"][:, 0] > 0.5
    ).detach()
    render_arguments = create_render_arguments(gaussian_cloud_parameters)
    foreground_means = render_arguments["means3D"][foreground_mask]
    foreground_rotations = render_arguments["rotations"][foreground_mask]
    foreground_rotations_from_previous_timestep_to_current = build_rotation(
        quat_mult(
            foreground_rotations, previous_timestep_foreground_info.inverted_rotations
        )
    )
    foreground_neighbor_means = foreground_means[initial_neighbor_info.indices]
    foreground_offset_to_neighbors = (
        foreground_neighbor_means - foreground_means[:, None]
    )
    foreground_offset_to_neighbors_in_previous_timestep_coordinates = (
        foreground_rotations_from_previous_timestep_to_current.transpose(2, 1)[:, None]
        @ foreground_offset_to_neighbors[:, :, :, None]
    ).squeeze(-1)
    return weighted_l2_loss_v2(
        foreground_offset_to_neighbors_in_previous_timestep_coordinates,
        previous_timestep_foreground_info.offsets_to_neighbors,
        initial_neighbor_info.weights,
    )


def calculate_l1_and_ssim_loss(gaussian_cloud_parameters, target_view: View):
    (
        rendered_image,
        _,
        _,
    ) = Renderer(
        raster_settings=target_view.render_settings
    )(**create_render_arguments(gaussian_cloud_parameters))
    l1_loss = torch.nn.functional.l1_loss(rendered_image, target_view.image)
    ssim_loss = 1.0 - calc_ssim(rendered_image, target_view.image)
    return l1_loss, ssim_loss


def calculate_base_losses(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_view: View,
    initial_neighbor_info: NeighborInfo,
    previous_timestep_foreground_info: ForegroundInfo,
):
    rigidity_loss = calculate_rigidity_loss(
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        initial_neighbor_info=initial_neighbor_info,
        previous_timestep_foreground_info=previous_timestep_foreground_info,
    )
    l1_loss, ssim_loss = calculate_l1_and_ssim_loss(
        gaussian_cloud_parameters, target_view
    )
    return torch.stack(
        [
            l1_loss,
            ssim_loss,
            rigidity_loss,
        ],
        dim=0,
    )


def combine_l1_and_ssim_loss(l1_loss: torch.Tensor, ssim_loss: torch.tensor):
    return 0.8 * l1_loss + 0.2 * ssim_loss


def calculate_loss(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_views: list[View],
    initial_neighbor_info: NeighborInfo,
    previous_timestep_foreground_info: ForegroundInfo,
    rigidity_loss_weight,
    i: int,
):
    losses = torch.stack(
        [
            calculate_base_losses(
                gaussian_cloud_parameters=gaussian_cloud_parameters,
                target_view=target_view,
                initial_neighbor_info=initial_neighbor_info,
                previous_timestep_foreground_info=previous_timestep_foreground_info,
            )
            for target_view in target_views
        ]
    )
    summed_losses = losses.sum(dim=0)
    l1_loss_sum = summed_losses[0]
    ssim_loss_sum = summed_losses[1]
    rigidity_loss_sum = summed_losses[2]
    image_loss = combine_l1_and_ssim_loss(l1_loss=l1_loss_sum, ssim_loss=ssim_loss_sum)
    total_loss = image_loss + 3 * rigidity_loss_weight * summed_losses[2]
    wandb.log(
        {
            "train-loss/total": total_loss.item(),
            "train-loss/l1": l1_loss_sum.item(),
            "train-loss/ssim": ssim_loss_sum.item(),
            "train-loss/image": image_loss.item(),
            "train-loss/rigidity": rigidity_loss_sum.item(),
        },
        step=i,
    )
    return total_loss


def export_deformation_network(
    run_output_directory_path: Path,
    sequence_name: str,
    data_directory_path: Path,
    deformation_network: DeformationNetwork,
    timestep_count: int,
    hidden_layer_count: int,
    hidden_dimension: int,
):
    network_directory_path = (
        run_output_directory_path / f"{wandb.run.name}_deformation_network"
    )
    network_directory_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        src=data_directory_path
        / sequence_name
        / "densified_initial_gaussian_cloud_parameters.pth",
        dst=network_directory_path / "densified_initial_gaussian_cloud_parameters.pth",
    )

    config_file_path = network_directory_path / "config.json"
    with config_file_path.open("w") as config_file:
        json.dump(
            {
                "timestep_count": timestep_count,
                "hidden_layer_count": hidden_layer_count,
                "hidden_dimension": hidden_dimension,
            },
            config_file,
            indent="\t",
        )

    network_state_dict_path = network_directory_path / "state_dict.pth"
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
    normalized_initial_means: torch.Tensor,
    normalized_initial_rotations: torch.Tensor,
    aspect_ratio: float,
    extrinsic_matrix: np.array,
    visualizations_directory_path: Path,
    fps: int,
):
    frames_directory = visualizations_directory_path / f"{name}_frames"
    frames_directory.mkdir(parents=True, exist_ok=True)
    frames = []
    for timestep in tqdm(range(timestep_count), desc=f"Creating Visualization {name}"):
        if timestep == 0:
            timestep_gaussian_cloud_parameters = initial_gaussian_cloud_parameters

        else:
            timestep_gaussian_cloud_parameters = update_gaussian_cloud_parameters(
                deformation_network=deformation_network,
                initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
                normalized_initial_means=normalized_initial_means,
                normalized_initial_rotations=normalized_initial_rotations,
                timestep=timestep,
                timestep_count=timestep_count,
            )

        image_width = 1280
        image_height = 720
        render_settings = create_render_settings(
            image_width=image_width,
            image_height=image_height,
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
            raster_settings=render_settings
        )(**create_render_arguments(timestep_gaussian_cloud_parameters))
        frame = (
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
        imageio.imwrite(
            frames_directory / f"frame_{timestep:04d}.png",
            frame,
        )
        frames.append(frame)
    imageio.mimwrite(
        visualizations_directory_path / f"{name}.mp4",
        frames,
        fps=fps,
    )


def export_visualizations(
    run_output_directory_path: Path,
    initial_gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    deformation_network: DeformationNetwork,
    timestep_count: int,
    fps: int,
):
    visualizations_directory_path = (
        run_output_directory_path / f"{wandb.run.name}_visualizations"
    )
    visualizations_directory_path.mkdir(parents=True, exist_ok=True)

    (
        normalized_initial_means,
        normalized_initial_rotations,
    ) = normalize_means_and_rotations(initial_gaussian_cloud_parameters)

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
            normalized_initial_means=normalized_initial_means,
            normalized_initial_rotations=normalized_initial_rotations,
            aspect_ratio=aspect_ratio,
            extrinsic_matrix=extrinsic_matrix,
            visualizations_directory_path=visualizations_directory_path,
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
    deformation_network = DeformationNetwork(
        hidden_dimension=config.hidden_dimension,
        hidden_layer_count=config.hidden_layer_count,
    ).cuda()
    optimizer = torch.optim.Adam(
        params=deformation_network.parameters(), lr=config.learning_rate
    )

    initial_gaussian_cloud_parameters = load_densified_initial_parameters(
        data_directory_path=config.data_directory_path,
        sequence_name=config.sequence_name,
    )

    initial_neighbor_info, previous_timestep_foreground_info = initialize_variables(
        gaussian_cloud_parameters=initial_gaussian_cloud_parameters
    )
    (
        normalized_initial_means,
        normalized_initial_rotations,
    ) = normalize_means_and_rotations(initial_gaussian_cloud_parameters)
    views = load_all_views(
        dataset_metadata=dataset_metadata,
        timestep_count=timestep_count,
        sequence_path=config.data_directory_path / config.sequence_name,
    )
    for i in tqdm(range(config.iteration_count), desc="Training"):
        timestep = (i % (timestep_count - 1)) + 1

        if timestep == 1:
            update_previous_timestep_foreground_info(
                gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
                previous_timestep_foreground_info=previous_timestep_foreground_info,
                initial_neighbor_indices=initial_neighbor_info.indices,
            )

        updated_gaussian_cloud_parameters = update_gaussian_cloud_parameters(
            deformation_network=deformation_network,
            initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
            normalized_initial_means=normalized_initial_means,
            normalized_initial_rotations=normalized_initial_rotations,
            timestep=timestep,
            timestep_count=timestep_count,
        )

        rigidity_loss_weight = (
            2.0 / (1.0 + math.exp(-6 * (i / config.iteration_count))) - 1
        )
        loss = calculate_loss(
            gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
            target_views=random.sample(views[timestep - 1], 5),
            initial_neighbor_info=initial_neighbor_info,
            previous_timestep_foreground_info=previous_timestep_foreground_info,
            rigidity_loss_weight=rigidity_loss_weight,
            i=i,
        )
        wandb.log(
            {
                "rigidity-loss-weight": rigidity_loss_weight,
                "learning-rate": optimizer.param_groups[0]["lr"],
            },
            step=i,
        )
        update_previous_timestep_foreground_info(
            gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
            previous_timestep_foreground_info=previous_timestep_foreground_info,
            initial_neighbor_indices=initial_neighbor_info.indices,
        )

        loss.backward()

        total_norm = torch.sqrt(
            torch.sum(
                torch.tensor(
                    [
                        parameter.grad.norm(2).pow(2)
                        for parameter in deformation_network.parameters()
                        if parameter.grad is not None
                    ]
                )
            )
        )
        wandb.log(
            {"gradient-norm": total_norm},
            step=i,
        )

        optimizer.step()
        optimizer.zero_grad()
    for timestep in tqdm(
        range(1, timestep_count), desc="Calculate Mean Image Loss per Timestep"
    ):
        image_losses = []
        with torch.no_grad():
            updated_gaussian_cloud_parameters = update_gaussian_cloud_parameters(
                deformation_network=deformation_network,
                initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
                normalized_initial_means=normalized_initial_means,
                normalized_initial_rotations=normalized_initial_rotations,
                timestep=timestep,
                timestep_count=timestep_count,
            )
            timestep_views = load_timestep_views(
                dataset_metadata,
                timestep,
                config.data_directory_path / config.sequence_name,
            )
            for view in timestep_views:
                l1_loss, ssim_loss = calculate_l1_and_ssim_loss(
                    gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
                    target_view=view,
                )
                image_losses.append(
                    combine_l1_and_ssim_loss(
                        l1_loss=l1_loss, ssim_loss=ssim_loss
                    ).item()
                )
        wandb.log(
            {"mean-image-loss": sum(image_losses) / len(image_losses)},
            step=config.iteration_count + timestep,
        )
    with torch.no_grad():
        run_output_directory_path = (
            config.output_directory_path / f"{config.sequence_name}_{wandb.run.name}"
        )
        export_deformation_network(
            run_output_directory_path=run_output_directory_path,
            sequence_name=config.sequence_name,
            data_directory_path=config.data_directory_path,
            deformation_network=deformation_network,
            timestep_count=timestep_count,
            hidden_layer_count=config.hidden_layer_count,
            hidden_dimension=config.hidden_dimension,
        )
        export_visualizations(
            run_output_directory_path=run_output_directory_path,
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
    argument_parser.add_argument("-t", "--timestep-count-limit", type=int)
    argument_parser.add_argument("-i", "--iteration-count", type=int, default=200_000)
    argument_parser.add_argument(
        "-o", "--output-directory-path", type=Path, default=Path("./out")
    )
    argument_parser.add_argument("-hd", "--hidden-dimension", type=int, default=128)
    argument_parser.add_argument("-hl", "--hidden-layer-count", type=int, default=3)
    argument_parser.add_argument("-lr", "--learning-rate", type=float, default=0.01)
    argument_parser.add_argument("-fps", type=int, default=30)
    args = argument_parser.parse_args()
    config = Config(
        sequence_name=args.sequence_name,
        data_directory_path=args.data_directory_path,
        hidden_dimension=args.hidden_dimension,
        hidden_layer_count=args.hidden_layer_count,
        learning_rate=args.learning_rate,
        timestep_count_limit=args.timestep_count_limit,
        output_directory_path=args.output_directory_path,
        iteration_count=args.iteration_count,
        fps=args.fps,
    )
    train(config=config)


if __name__ == "__main__":
    main()
