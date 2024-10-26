import argparse
import copy
import json
import pathlib
import random
import typing
from dataclasses import dataclass

import torch
import wandb
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from torch import nn
from tqdm import tqdm

from external import calc_ssim
from shared import View, create_render_arguments, load_timestep_views


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


def calculate_gradient_norm(deformation_network):
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
    return total_norm


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


def combine_l1_and_ssim_loss(l1_loss: torch.Tensor, ssim_loss: torch.tensor):
    return 0.8 * l1_loss + 0.2 * ssim_loss


def calculate_timestep_loss(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter], target_views: list[View]
):
    l1_loss_sum = torch.tensor(0)
    ssim_loss_sum = torch.tensor(0)
    for target_view in target_views:
        l1_loss, ssim_loss = calculate_l1_and_ssim_loss(
            gaussian_cloud_parameters, target_view
        )
        l1_loss_sum += l1_loss
        ssim_loss_sum += ssim_loss
    return l1_loss_sum, ssim_loss_sum


def calculate_loss(
    deformation_network: DeformationNetwork,
    initial_gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    encoded_normalized_initial_means,
    encoded_normalized_initial_rotations,
    timestep_count,
    views: list[list[View]],
    i: int,
):
    l1_loss_sum = torch.tensor(0)
    ssim_loss_sum = torch.tensor(0)

    for timestep in range(1, timestep_count):
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
        l1_loss_sum += l1_loss
        ssim_loss_sum += ssim_loss

    image_loss = combine_l1_and_ssim_loss(l1_loss=l1_loss_sum, ssim_loss=ssim_loss_sum)
    total_loss = image_loss
    wandb.log(
        {
            "train-loss/total": total_loss.item(),
            "train-loss/l1": l1_loss_sum.item(),
            "train-loss/ssim": ssim_loss_sum.item(),
            "train-loss/image": image_loss.item(),
        },
        step=i,
    )
    return total_loss


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
