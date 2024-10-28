import argparse
import copy
import json
import pathlib
import random
import shutil
import typing
from dataclasses import dataclass

import imageio
import numpy as np
import torch
import wandb
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from torch import nn
from tqdm import tqdm

from external import calc_ssim
from shared import (
    View,
    create_render_arguments,
    load_timestep_views,
    create_render_settings,
)


@dataclass
class Config:
    sequence_name: str
    data_directory_path: pathlib.Path
    hidden_dimension: int
    residual_block_count: int
    learning_rate: float
    timestep_count_limit: typing.Optional[int]
    output_directory_path: pathlib.Path
    iteration_count: int
    fps: int


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
    def __init__(self, hidden_dimension, residual_block_count) -> None:
        super(DeformationNetwork, self).__init__()
        self.fc_in = nn.Linear(100, hidden_dimension)
        self.residual_blocks = nn.Sequential(
            *(ResidualBlock(hidden_dimension) for _ in range(residual_block_count))
        )
        self.fc_out = nn.Linear(hidden_dimension, 7)

    def forward(
        self,
        initial_means: torch.Tensor,
        initial_rotations: torch.Tensor,
        encoded_normalized_initial_means: torch.Tensor,
        encoded_normalized_initial_rotations: torch.Tensor,
        encoded_progress: torch.Tensor,
    ):
        encoded_normalized_input = torch.cat(
            (encoded_normalized_initial_means, encoded_normalized_initial_rotations),
            dim=1,
        )
        out = torch.cat((encoded_normalized_input, encoded_progress), dim=1)
        out = self.fc_in(out)
        out = self.residual_blocks(out)
        out = self.fc_out(out)

        out += torch.cat(
            (initial_means, initial_rotations),
            dim=1,
        )

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


def load_densified_initial_parameters(
    data_directory_path: pathlib.Path, sequence_name: str
):
    parameters: dict[str, torch.nn.Parameter] = torch.load(
        data_directory_path
        / sequence_name
        / "densified_initial_gaussian_cloud_parameters.pth"
    )
    for parameter in parameters.values():
        parameter.requires_grad = False
    return parameters


def encode_means_and_rotations(
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
    encoded_normalized_means = PositionalEncoding(L=10)(normalized_means)
    encoded_normalized_rotations = PositionalEncoding(L=4)(normalized_rotations)
    return (
        encoded_normalized_means,
        encoded_normalized_rotations,
    )


def load_all_views(dataset_metadata, timestep_count: int, sequence_path: pathlib.Path):
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


def update_gaussian_cloud_parameters(
    deformation_network: DeformationNetwork,
    initial_gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    encoded_normalized_initial_means,
    encoded_normalized_initial_rotations,
    timestep,
    timestep_count,
):
    encoded_progress = PositionalEncoding(L=4)(
        torch.tensor(timestep / (timestep_count - 1))
        .view(1, 1)
        .repeat(encoded_normalized_initial_means.shape[0], 1)
        .cuda()
    )
    delta = deformation_network(
        initial_means=initial_gaussian_cloud_parameters["means"],
        initial_rotations=initial_gaussian_cloud_parameters["rotation_quaternions"],
        encoded_normalized_initial_means=encoded_normalized_initial_means,
        encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
        encoded_progress=encoded_progress,
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


def calculate_timestep_loss(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter], target_views: list[View]
):
    l1_losses = []
    ssim_losses = []
    for target_view in target_views:
        l1_loss, ssim_loss = calculate_l1_and_ssim_loss(
            gaussian_cloud_parameters, target_view
        )
        l1_losses.append(l1_loss)
        ssim_losses.append(ssim_loss)
    return torch.stack(l1_losses).sum(), torch.stack(ssim_losses).sum()


def combine_l1_and_ssim_loss(l1_loss: torch.Tensor, ssim_loss: torch.tensor):
    return 0.8 * l1_loss + 0.2 * ssim_loss


def calculate_loss(
    deformation_network: DeformationNetwork,
    initial_gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    encoded_normalized_initial_means,
    encoded_normalized_initial_rotations,
    timestep_count,
    views: list[list[View]],
    i: int,
):
    l1_losses = []
    ssim_losses = []

    for timestep in tqdm(range(1, timestep_count), unit="timesteps", leave=False):
        updated_gaussian_cloud_parameters = update_gaussian_cloud_parameters(
            deformation_network=deformation_network,
            initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
            encoded_normalized_initial_means=encoded_normalized_initial_means,
            encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
            timestep=timestep,
            timestep_count=timestep_count,
        )
        l1_loss, ssim_loss = calculate_timestep_loss(
            gaussian_cloud_parameters=updated_gaussian_cloud_parameters,
            target_views=random.sample(views[timestep - 1], 5),
        )
        l1_losses.append(l1_loss)
        ssim_losses.append(ssim_loss)

    total_l1_loss = torch.stack(l1_losses).sum()
    total_ssim_loss = torch.stack(ssim_losses).sum()
    image_loss = combine_l1_and_ssim_loss(
        l1_loss=total_l1_loss, ssim_loss=total_ssim_loss
    )
    total_loss = image_loss
    wandb.log(
        {
            "train-loss/total": total_loss.item(),
            "train-loss/l1": total_l1_loss.item(),
            "train-loss/ssim": total_ssim_loss.item(),
            "train-loss/image": image_loss.item(),
            "train-loss-mean/total": total_loss.item() / (timestep_count - 1),
            "train-loss-mean/l1": total_l1_loss.item() / (timestep_count - 1),
            "train-loss-mean/ssim": total_ssim_loss.item() / (timestep_count - 1),
            "train-loss-mean/image": image_loss.item() / (timestep_count - 1),
        },
        step=i,
    )
    return total_loss


def calculate_gradient_norm(deformation_network: DeformationNetwork):
    return torch.sqrt(
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


def export_deformation_network(
    run_output_directory_path: pathlib.Path,
    sequence_name: str,
    data_directory_path: pathlib.Path,
    deformation_network: DeformationNetwork,
    timestep_count: int,
    residual_block_count: int,
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
                "residual_block_count": residual_block_count,
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
    encoded_normalized_initial_means: torch.Tensor,
    encoded_normalized_initial_rotations: torch.Tensor,
    aspect_ratio: float,
    extrinsic_matrix: np.array,
    visualizations_directory_path: pathlib.Path,
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
                encoded_normalized_initial_means=encoded_normalized_initial_means,
                encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
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
    run_output_directory_path: pathlib.Path,
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
        encoded_normalized_initial_means,
        encoded_normalized_initial_rotations,
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
            encoded_normalized_initial_means=encoded_normalized_initial_means,
            encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
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
        residual_block_count=config.residual_block_count,
    ).cuda()
    optimizer = torch.optim.Adam(
        params=deformation_network.parameters(), lr=config.learning_rate
    )

    initial_gaussian_cloud_parameters = load_densified_initial_parameters(
        data_directory_path=config.data_directory_path,
        sequence_name=config.sequence_name,
    )

    (
        encoded_normalized_initial_means,
        encoded_normalized_initial_rotations,
    ) = encode_means_and_rotations(initial_gaussian_cloud_parameters)
    views = load_all_views(
        dataset_metadata=dataset_metadata,
        timestep_count=timestep_count,
        sequence_path=config.data_directory_path / config.sequence_name,
    )
    for i in tqdm(range(config.iteration_count), desc="Training"):
        loss = calculate_loss(
            deformation_network=deformation_network,
            initial_gaussian_cloud_parameters=initial_gaussian_cloud_parameters,
            encoded_normalized_initial_means=encoded_normalized_initial_means,
            encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
            timestep_count=timestep_count,
            views=views,
            i=i,
        )
        loss.backward()
        wandb.log(
            {"gradient-norm": calculate_gradient_norm(deformation_network)},
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
                encoded_normalized_initial_means=encoded_normalized_initial_means,
                encoded_normalized_initial_rotations=encoded_normalized_initial_rotations,
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
            residual_block_count=config.residual_block_count,
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
        "data_directory_path", metavar="data-directory-path", type=pathlib.Path
    )
    argument_parser.add_argument("-t", "--timestep-count-limit", type=int)
    argument_parser.add_argument("-i", "--iteration-count", type=int, default=200_000)
    argument_parser.add_argument(
        "-o",
        "--output-directory-path",
        type=pathlib.Path,
        default=pathlib.Path("./out"),
    )
    argument_parser.add_argument("-hd", "--hidden-dimension", type=int, default=128)
    argument_parser.add_argument("-r", "--residual-block-count", type=int, default=6)
    argument_parser.add_argument("-lr", "--learning-rate", type=float, default=0.01)
    argument_parser.add_argument("-fps", type=int, default=30)
    args = argument_parser.parse_args()
    config = Config(
        sequence_name=args.sequence_name,
        data_directory_path=args.data_directory_path,
        hidden_dimension=args.hidden_dimension,
        residual_block_count=args.residual_block_count,
        learning_rate=args.learning_rate,
        timestep_count_limit=args.timestep_count_limit,
        output_directory_path=args.output_directory_path,
        iteration_count=args.iteration_count,
        fps=args.fps,
    )
    train(config=config)


if __name__ == "__main__":
    main()
