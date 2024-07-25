import os
import torch
import numpy as np
from deformation_network.deformation_network import DeformationNetwork
import json
from commons.classes import Camera

def _load_initial_gaussian_cloud_parameters(network_directory_path):
    return {
        parameter_name: torch.tensor(parameter_value).cuda().float()
        for parameter_name, parameter_value in dict(
            np.load(
                os.path.join(
                    network_directory_path,
                    "initial_gaussian_cloud_parameters.npz",
                )
            )
        ).items()
    }

def main():
    path = '/home/cv_defect/YASIN/Mask2Former/tools/4d-gaussian-splatting/out5/test1/basketball'

    initial_cloud = _load_initial_gaussian_cloud_parameters(path)

    for parameter in initial_cloud.values():
        parameter.requires_grad = False

    yaw = 90.

    timestep_count = json.load(
        open(
            os.path.join(
                path,
                "metadata.json",
            ),
            "r",
        )
    )["timestep_count"]

    network_state_dict_path = os.path.join(
        path,
        "network_state_dict.pth",
    )
    deformation_network = DeformationNetwork(timestep_count).cuda()
    # deformation_network.load_state_dict(torch.load(network_state_dict_path))

    deformation_network(
        torch.cat((initial_cloud['means3D'], initial_cloud['unnorm_rotations']), dim=1),
        t=torch.tensor(0).cuda()
    )[100]

    a=1

if __name__ == '__main__':
    main()