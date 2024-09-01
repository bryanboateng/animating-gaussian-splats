import argparse
import copy
import json
import math
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
    load_view,
    GaussianCloudParameters,
    View,
    apply_exponential_transform_and_center_to_image,
    l1_loss_v1,
    create_render_arguments,
    create_render_settings,
)


@dataclass
class Config:
    sequence_name: str
    data_directory_path: Path
    learning_rate: float
    timestep_count_limit: Optional[int]
    output_directory_path: Path
    total_iteration_count: int
    warmup_iteration_count: float
    fps: int


@dataclass
class Neighborhoods:
    distances: torch.Tensor
    weights: torch.Tensor
    indices: torch.Tensor


@dataclass
class GaussianCloudReferenceState:
    means: torch.Tensor
    rotations: torch.Tensor
    inverted_foreground_rotations: Optional[torch.Tensor] = None
    offsets_to_neighbors: Optional[torch.Tensor] = None
    colors: Optional[torch.Tensor] = None


class ResidualBlock(nn.Module):
    def __init__(self, dimension) -> None:
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dimension, dimension, bias=False)
        self.bn1 = nn.BatchNorm1d(dimension)
        self.fc2 = nn.Linear(dimension, dimension, bias=False)
        self.bn2 = nn.BatchNorm1d(dimension)

    def forward(self, x):
        identity = x

        x = self.fc1(x)
        x = self.bn1(x)
        x = nn.functional.gelu(x)
        x = self.fc2(x)
        x = self.bn2(x)

        x += identity
        x = nn.functional.gelu(x)

        return x


class DeformationNetwork(nn.Module):
    def __init__(self) -> None:
        super(DeformationNetwork, self).__init__()
        hidden_dimension = 128
        self.fc_in = nn.Linear(100, hidden_dimension)
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(hidden_dimension) for _ in range(6))
        )
        self.fc_out = nn.Linear(hidden_dimension, 7)

    def forward(self, input_, normalized_input, timestep):
        out = torch.cat((normalized_input, timestep), dim=1)
        out = self.fc_in(out)
        out = self.residual_blocks(out)
        out = self.fc_out(out)

        out += input_

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super(PositionalEncoding, self).__init__()
        self.L = L
        self.consts = ((torch.ones(L) * 2).pow(torch.arange(L)) * torch.pi).cuda()

    def forward(self, x):
        x = x[:, :, None]
        A = (self.consts * x).repeat_interleave(2, 2)
        A[:, :, ::2] = torch.sin(A[:, :, ::2])
        A[:, :, 1::2] = torch.cos(A[:, :, ::2])

        return A.permute(0, 2, 1).flatten(start_dim=1)


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


def encode_means_and_rotations(gaussian_cloud_parameters: GaussianCloudParameters):
    means = gaussian_cloud_parameters.means
    rotations = gaussian_cloud_parameters.rotation_quaternions
    normalized_means = means - means.min(dim=0).values
    normalized_means = (
        2.0 * normalized_means / normalized_means.max(dim=0).values
    ) - 1.0
    normalized_rotations = rotations - rotations.min(dim=0).values
    normalized_rotations = (
        2.0 * normalized_rotations / normalized_rotations.max(dim=0).values
    ) - 1.0
    large_positional_encoding = PositionalEncoding(L=10)
    small_positional_encoding = PositionalEncoding(L=4)
    encoded_normalized_means = large_positional_encoding(normalized_means)
    encoded_normalized_rotations = small_positional_encoding(normalized_rotations)
    return (
        encoded_normalized_means,
        encoded_normalized_rotations,
        small_positional_encoding,
    )


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


def update_gaussian_cloud_parameters(
    deformation_network: DeformationNetwork,
    initial_gaussian_cloud_parameters: GaussianCloudParameters,
    encoded_normalized_initial_means,
    encoded_normalized_initial_rotations,
    small_positional_encoding: PositionalEncoding,
    timestep,
    timestep_count,
):
    encoded_timestep = small_positional_encoding(
        torch.tensor((timestep + 1) / timestep_count)
        .view(1, 1)
        .repeat(encoded_normalized_initial_means.shape[0], 1)
        .cuda()
    )
    delta = deformation_network(
        torch.cat(
            (
                initial_gaussian_cloud_parameters.means,
                initial_gaussian_cloud_parameters.rotation_quaternions,
            ),
            dim=1,
        ),
        torch.cat(
            (encoded_normalized_initial_means, encoded_normalized_initial_rotations),
            dim=1,
        ),
        encoded_timestep,
    )
    means_delta = delta[:, :3]
    rotations_delta = delta[:, 3:]
    updated_gaussian_cloud_parameters = copy.deepcopy(initial_gaussian_cloud_parameters)
    updated_gaussian_cloud_parameters.means = (
        updated_gaussian_cloud_parameters.means.detach()
    )
    updated_gaussian_cloud_parameters.means += means_delta * 0.01
    updated_gaussian_cloud_parameters.rotation_quaternions = (
        updated_gaussian_cloud_parameters.rotation_quaternions.detach()
    )
    updated_gaussian_cloud_parameters.rotation_quaternions += rotations_delta * 0.01
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
    gaussian_cloud_parameters,
    foreground_mask,
    initial_neighborhoods,
    previous_timestep_gaussian_cloud_state,
):
    render_arguments = create_render_arguments(gaussian_cloud_parameters)
    foreground_means = render_arguments["means3D"][foreground_mask]
    foreground_rotations = render_arguments["rotations"][foreground_mask]
    relative_rotation_quaternion = quat_mult(
        foreground_rotations,
        previous_timestep_gaussian_cloud_state.inverted_foreground_rotations,
    )
    rotation_matrix = build_rotation(relative_rotation_quaternion)
    foreground_neighbor_means = foreground_means[initial_neighborhoods.indices]
    foreground_offset_to_neighbors = (
        foreground_neighbor_means - foreground_means[:, None]
    )
    curr_offset_in_prev_coord = (
        rotation_matrix.transpose(2, 1)[:, None]
        @ foreground_offset_to_neighbors[:, :, :, None]
    ).squeeze(-1)
    return weighted_l2_loss_v2(
        curr_offset_in_prev_coord,
        previous_timestep_gaussian_cloud_state.offsets_to_neighbors,
        initial_neighborhoods.weights,
    )


def calculate_image_loss(gaussian_cloud_parameters, target_view: View):
    (
        rendered_image,
        _,
        _,
    ) = Renderer(
        raster_settings=target_view.render_settings
    )(**create_render_arguments(gaussian_cloud_parameters))
    image = apply_exponential_transform_and_center_to_image(
        rendered_image, gaussian_cloud_parameters, target_view.camera_index
    )
    l1_loss = l1_loss_v1(image, target_view.image)
    ssim_loss = 1.0 - calc_ssim(image, target_view.image)
    image_loss = 0.8 * l1_loss + 0.2 * ssim_loss
    return l1_loss, ssim_loss, image_loss


def calculate_loss(
    gaussian_cloud_parameters: GaussianCloudParameters,
    target_view: View,
    initial_neighborhoods: Neighborhoods,
    previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
    rigidity_loss_weight,
):

    foreground_mask = (
        gaussian_cloud_parameters.segmentation_colors[:, 0] > 0.5
    ).detach()
    rigidity_loss = calculate_rigidity_loss(
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        foreground_mask=foreground_mask,
        initial_neighborhoods=initial_neighborhoods,
        previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
    )
    l1_loss, ssim_loss, image_loss = calculate_image_loss(
        gaussian_cloud_parameters, target_view
    )
    return (
        image_loss + 3 * rigidity_loss_weight * rigidity_loss,
        l1_loss,
        ssim_loss,
        image_loss,
        rigidity_loss,
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
    small_positional_encoding: PositionalEncoding,
    encoded_normalized_initial_means: torch.Tensor,
    encoded_normalized_initial_rotations: torch.Tensor,
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
            timestep_gaussian_cloud_parameters = update_gaussian_cloud_parameters(
                deformation_network=deformation_network,
                initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
                encoded_normalized_initial_means=encoded_normalized_initial_means,
                encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
                small_positional_encoding=small_positional_encoding,
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

    (
        encoded_normalized_initial_means,
        encoded_normalized_initial_rotations,
        small_positional_encoding,
    ) = encode_means_and_rotations(initial_gaussian_cloud_parameters)

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
            small_positional_encoding=small_positional_encoding,
            encoded_normalized_initial_means=encoded_normalized_initial_means,
            encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
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
        warmup_iters=config.warmup_iteration_count,
        total_iters=config.total_iteration_count,
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
    (
        encoded_normalized_initial_means,
        encoded_normalized_initial_rotations,
        small_positional_encoding,
    ) = encode_means_and_rotations(initial_gaussian_cloud_parameters)
    camera_count = len(dataset_metadata["fn"][0])
    for i in tqdm(range(config.total_iteration_count), desc="Training"):
        timestep = (i % (timestep_count - 1)) + 1
        camera_index = torch.randint(0, camera_count, ())

        if timestep == 1:
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
        updated_gaussian_cloud_parameters = update_gaussian_cloud_parameters(
            deformation_network=deformation_network,
            initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
            encoded_normalized_initial_means=encoded_normalized_initial_means,
            encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
            small_positional_encoding=small_positional_encoding,
            timestep=timestep,
            timestep_count=timestep_count,
        )

        total_loss, l1_loss, ssim_loss, image_loss, rigidity_loss = calculate_loss(
            gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
            target_view=view,
            initial_neighborhoods=initial_neighborhoods,
            previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
            rigidity_loss_weight=(
                2.0 / (1.0 + math.exp(-6 * (i / config.total_iteration_count))) - 1
            ),
        )
        wandb.log(
            {
                f"train-loss/total": total_loss.item(),
                f"train-loss/l1": l1_loss.item(),
                f"train-loss/ssim": ssim_loss.item(),
                f"train-loss/image": image_loss.item(),
                f"train-loss/rigidity": rigidity_loss.item(),
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
        range(1, timestep_count), desc="Calculate Mean Image Loss per Timestep"
    ):
        image_losses = []
        with torch.no_grad():
            updated_gaussian_cloud_parameters = update_gaussian_cloud_parameters(
                deformation_network=deformation_network,
                initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
                encoded_normalized_initial_means=encoded_normalized_initial_means,
                encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
                small_positional_encoding=small_positional_encoding,
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
            step=config.total_iteration_count + timestep,
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
        "-o", "--output-directory-path", type=Path, default=Path("./out")
    )
    argument_parser.add_argument(
        "-ti", "--total_iteration_count", type=int, default=200_000
    )
    argument_parser.add_argument(
        "-wi", "--warmup_iteration_count", type=float, default=15_000
    )
    argument_parser.add_argument("--fps", type=int, default=30)
    args = argument_parser.parse_args()
    config = Config(
        sequence_name=args.sequence_name,
        data_directory_path=args.data_directory_path,
        learning_rate=args.learning_rate,
        timestep_count_limit=args.timestep_count_limit,
        output_directory_path=args.output_directory_path,
        total_iteration_count=args.total_iteration_count,
        warmup_iteration_count=args.warmup_iteration_count,
        fps=args.fps,
    )
    train(config=config)


if __name__ == "__main__":
    main()
