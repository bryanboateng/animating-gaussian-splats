import copy

import torch
from torch import nn


class DeformationNetwork(nn.Module):
    def __init__(self, sequence_length) -> None:
        input_size = 7
        super(DeformationNetwork, self).__init__()
        self.embedding_dimension = 2
        self.embedding = nn.Embedding(sequence_length, self.embedding_dimension)
        self.fc1 = nn.Linear(input_size + self.embedding_dimension, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, input_size)

        self.relu = nn.ReLU()

    def forward(self, input_tensor, timestep):
        batch_size = input_tensor.shape[0]
        initial_input_tensor = input_tensor
        embedding_tensor = self.embedding(timestep).repeat(batch_size, 1)
        input_with_embedding = torch.cat((input_tensor, embedding_tensor), dim=1)

        x = self.relu(self.fc1(input_with_embedding))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)

        return initial_input_tensor + x


def update_parameters(deformation_network: DeformationNetwork, parameters, timestep):
    delta = deformation_network(
        torch.cat((parameters["means"], parameters["rotations"]), dim=1),
        torch.tensor(timestep).cuda(),
    )
    means_delta = delta[:, :3]
    rotations_delta = delta[:, 3:]
    updated_parameters = copy.deepcopy(parameters)
    updated_parameters["means"] = updated_parameters["means"].detach()
    updated_parameters["means"] += means_delta * 0.01
    updated_parameters["rotations"] = updated_parameters["rotations"].detach()
    updated_parameters["rotations"] += rotations_delta * 0.01
    return updated_parameters


def create_gaussian_cloud(parameters):
    return {
        "means3D": parameters["means"],
        "colors_precomp": parameters["colors"],
        "rotations": torch.nn.functional.normalize(parameters["rotations"]),
        "opacities": torch.sigmoid(parameters["opacities"]),
        "scales": torch.exp(parameters["scales"]),
        "means2D": torch.zeros_like(
            parameters["means"], requires_grad=True, device="cuda"
        )
        + 0,
    }


def load_and_freeze_parameters(path: str):
    parameters = torch.load(path)
    for parameter in parameters.values():
        parameter.requires_grad = False
    return parameters
