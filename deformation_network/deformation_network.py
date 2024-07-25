import copy

import torch
from torch import nn

from commons.classes import GaussianCloudParameterNames

class PositionalEncoding(nn.Module):
    def __init__(self, L):
        super(PositionalEncoding, self).__init__()
        self.L = L
        self.consts = ((torch.ones(L)*2).pow(torch.arange(L)) * torch.pi).cuda()
    
    def forward(self, x):
        x = x[:,:,None]
        A = (self.consts * x).repeat_interleave(2,2)
        A[:,:,::2] = torch.sin(A[:,:,::2])
        A[:,:,1::2] = torch.cos(A[:,:,::2])

        return A.permute(0,2,1).flatten(start_dim=1)

class DeformationNetwork(nn.Module):
    # def __init__(self, sequence_length) -> None:
    #     input_size = 7
    #     super(DeformationNetwork, self).__init__()
    #     self.embedding_dimension = 2
    #     self.embedding = nn.Embedding(sequence_length, self.embedding_dimension)
    #     self.fc1 = nn.Linear(input_size + self.embedding_dimension, 128)
    #     self.fc2 = nn.Linear(128, 256)
    #     self.fc3 = nn.Linear(256, 512)
    #     self.fc4 = nn.Linear(512, 256)
    #     self.fc5 = nn.Linear(256, 128)
    #     self.fc6 = nn.Linear(128, input_size)

    #     self.relu = nn.ReLU()

    # def forward(self, input_tensor, timestep):
    #     batch_size = input_tensor.shape[0]
    #     initial_input_tensor = input_tensor
    #     embedding_tensor = self.embedding(timestep).repeat(batch_size, 1)
    #     input_with_embedding = torch.cat((input_tensor, embedding_tensor), dim=1)

    #     x = self.relu(self.fc1(input_with_embedding))
    #     x = self.relu(self.fc2(x))
    #     x = self.relu(self.fc3(x))
    #     x = self.relu(self.fc4(x))
    #     x = self.relu(self.fc5(x))
    #     x = self.fc6(x)

    #     return initial_input_tensor + x

    # def __init__(self, seq_len) -> None:
    #     super(DeformationNetwork, self).__init__()
    #     self.fc1 = nn.Linear(7 + 2, 128)
    #     self.fc2 = nn.Linear(128, 512)
    #     self.fc3 = nn.Linear(512, 1024)
    #     self.fc4 = nn.Linear(1024, 512)
    #     self.fc5 = nn.Linear(512, 128)
    #     self.fc6 = nn.Linear(128, 7)

    #     self.relu = nn.ReLU()

    #     self.emb = nn.Embedding(seq_len, 2)

    # def forward(self, x, t):
    #     B, D = x.shape

    #     x_ = x

    #     e = self.emb(t).repeat(B, 1)
    #     # e = self.dec2bin(t, 3).repeat(B, 1)

    #     x = torch.cat((x, e), dim=1)

    #     x = x # + e
    #     x = self.relu(self.fc1(x))
    #     x1 = x
    #     x = self.relu(self.fc2(x))
    #     x2 = x
    #     x = self.relu(self.fc3(x))
    #     x = self.relu(self.fc4(x))
    #     x = x + x2
    #     x = self.relu(self.fc5(x))
    #     x = x + x1
    #     x =           self.fc6(x)

    #     return x_ + x

    def __init__(self, seq_len) -> None:
        super(DeformationNetwork, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 7)

        self.relu = nn.ReLU()

        self.emb = nn.Embedding(seq_len, 2)

        self.mean_encoder = PositionalEncoding(L=10)
        self.smol_encoder = PositionalEncoding(L=4)

        self.seq_len = seq_len

    def forward(self, x, t):
        t = t/self.seq_len
        # print(f't={t}')

        B, D = x.shape

        x_ = x

        m = self.mean_encoder(x[:,:3])
        r = self.smol_encoder(x[:,3:7])
        t = self.smol_encoder(t.repeat(B,1))

        x = torch.cat((m, r, t), dim=1).contiguous()

        x = self.relu(self.fc1(x))
        x1 = x
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = x + x1
        x2 = x
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = x + x2
        x =           self.fc6(x)

        return x_ + x

def update_parameters(deformation_network: DeformationNetwork, parameters, timestep):
    delta = deformation_network(
        torch.cat(
            (
                parameters[GaussianCloudParameterNames.means],
                parameters[GaussianCloudParameterNames.rotation_quaternions],
            ),
            dim=1,
        ),
        torch.tensor(timestep).cuda(),
    )
    means_delta = delta[:, :3]
    rotations_delta = delta[:, 3:]
    updated_parameters = copy.deepcopy(parameters)
    updated_parameters[GaussianCloudParameterNames.means] = updated_parameters[
        GaussianCloudParameterNames.means
    ].detach()
    updated_parameters[GaussianCloudParameterNames.means] += means_delta * 0.01
    updated_parameters[GaussianCloudParameterNames.rotation_quaternions] = (
        updated_parameters[GaussianCloudParameterNames.rotation_quaternions].detach()
    )
    updated_parameters[GaussianCloudParameterNames.rotation_quaternions] += (
        rotations_delta * 0.01
    )
    return updated_parameters
