import copy

import torch
from torch import nn

from commons.classes import GaussianCloudParameters


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


def normalize_means_and_rotations(gaussian_cloud_parameters: GaussianCloudParameters):
    means = gaussian_cloud_parameters.means
    rotations = gaussian_cloud_parameters.rotation_quaternions
    means_offset = means - means.min(dim=0).values
    normalized_means = (2.0 * means_offset / means_offset.max(dim=0).values) - 1.0
    rotations_offset = rotations - rotations.min(dim=0).values
    normalized_rotations = (
        2.0 * rotations_offset / rotations_offset.max(dim=0).values
    ) - 1.0
    means_encoder = PositionalEncoding(L=10)
    rotations_encoder = PositionalEncoding(L=4)
    encoded_means = means_encoder(normalized_means)
    encoded_rotations = rotations_encoder(normalized_rotations)
    return encoded_means, rotations_encoder, encoded_rotations


def update_parameters(
    deformation_network: DeformationNetwork,
    positional_encoding: PositionalEncoding,
    normalized_means,
    normalized_rotations,
    parameters: GaussianCloudParameters,
    timestep,
    timestep_count,
):
    timestep = positional_encoding(
        torch.tensor((timestep + 1) / timestep_count)
        .view(1, 1)
        .repeat(normalized_means.shape[0], 1)
        .cuda()
    )
    delta = deformation_network(
        torch.cat(
            (parameters.means, parameters.rotation_quaternions),
            dim=1,
        ),
        torch.cat((normalized_means, normalized_rotations), dim=1),
        timestep,
    )
    means_delta = delta[:, :3]
    rotations_delta = delta[:, 3:]
    updated_parameters = copy.deepcopy(parameters)
    updated_parameters.means = updated_parameters.means.detach()
    updated_parameters.means += means_delta * 0.01
    updated_parameters.rotation_quaternions = (
        updated_parameters.rotation_quaternions.detach()
    )
    updated_parameters.rotation_quaternions += rotations_delta * 0.01
    return updated_parameters
