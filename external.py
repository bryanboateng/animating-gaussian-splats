"""
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file found here:
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/LICENSE.md
#
# For inquiries contact  george.drettakis@inria.fr

#######################################################################################################################
##### NOTE: CODE IN THIS FILE IS NOT INCLUDED IN THE OVERALL PROJECT'S MIT LICENSE #####
##### USE OF THIS CODE FOLLOWS THE COPYRIGHT NOTICE ABOVE #####
#######################################################################################################################
"""

from math import exp

import torch
import torch.nn.functional as func
from torch.autograd import Variable

from shared import DensificationVariables


def build_rotation(q):
    norm = torch.sqrt(
        q[:, 0] * q[:, 0] + q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2] + q[:, 3] * q[:, 3]
    )
    q = q / norm[:, None]
    rot = torch.zeros((q.size(0), 3, 3), device="cuda")
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - r * z)
    rot[:, 0, 2] = 2 * (x * z + r * y)
    rot[:, 1, 0] = 2 * (x * y + r * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - r * x)
    rot[:, 2, 0] = 2 * (x * z - r * y)
    rot[:, 2, 1] = 2 * (y * z + r * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot


def gaussian(window_size, sigma):
    gauss = torch.Tensor(
        [
            exp(-((x - window_size // 2) ** 2) / float(2 * sigma**2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )
    return window


def calc_ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = func.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = func.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        func.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel)
        - mu1_sq
    )
    sigma2_sq = (
        func.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel)
        - mu2_sq
    )
    sigma12 = (
        func.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel)
        - mu1_mu2
    )

    c1 = 0.01**2
    c2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def accumulate_mean_2d_gradients(densification_variables: DensificationVariables):
    densification_variables.mean_2d_gradients_accumulated[
        densification_variables.gaussian_is_visible_mask
    ] += torch.norm(
        densification_variables.means_2d.grad[
            densification_variables.gaussian_is_visible_mask, :2
        ],
        dim=-1,
    )
    densification_variables.visibility_count[
        densification_variables.gaussian_is_visible_mask
    ] += 1


def update_params_and_optimizer(
    new_params: dict[str, torch.Tensor],
    params: dict[str, torch.nn.Parameter],
    optimizer,
):
    for k, v in new_params.items():
        group = [x for x in optimizer.param_groups if x["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)

        stored_state["exp_avg"] = torch.zeros_like(v)
        stored_state["exp_avg_sq"] = torch.zeros_like(v)
        del optimizer.state[group["params"][0]]

        group["params"][0] = torch.nn.Parameter(v.requires_grad_(True))
        optimizer.state[group["params"][0]] = stored_state
        params[k] = group["params"][0]


def cat_params_to_optimizer(
    new_params: dict[str, torch.Tensor],
    params: dict[str, torch.nn.Parameter],
    optimizer,
):
    for k, v in new_params.items():
        group = [g for g in optimizer.param_groups if g["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = torch.cat(
                (stored_state["exp_avg"], torch.zeros_like(v)), dim=0
            )
            stored_state["exp_avg_sq"] = torch.cat(
                (stored_state["exp_avg_sq"], torch.zeros_like(v)), dim=0
            )
            del optimizer.state[group["params"][0]]
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], v), dim=0).requires_grad_(True)
            )
            optimizer.state[group["params"][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                torch.cat((group["params"][0], v), dim=0).requires_grad_(True)
            )
            params[k] = group["params"][0]


def remove_points(
    to_remove,
    params: dict[str, torch.nn.Parameter],
    densification_variables: DensificationVariables,
    optimizer: torch.optim.Adam,
):
    to_keep = ~to_remove
    keys = [k for k in params.keys() if k not in ["camera_matrices", "camera_center"]]
    for k in keys:
        group = [g for g in optimizer.param_groups if g["name"] == k][0]
        stored_state = optimizer.state.get(group["params"][0], None)
        if stored_state is not None:
            stored_state["exp_avg"] = stored_state["exp_avg"][to_keep]
            stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][to_keep]
            del optimizer.state[group["params"][0]]
            group["params"][0] = torch.nn.Parameter(
                (group["params"][0][to_keep].requires_grad_(True))
            )
            optimizer.state[group["params"][0]] = stored_state
            params[k] = group["params"][0]
        else:
            group["params"][0] = torch.nn.Parameter(
                group["params"][0][to_keep].requires_grad_(True)
            )
            params[k] = group["params"][0]
    densification_variables.mean_2d_gradients_accumulated = (
        densification_variables.mean_2d_gradients_accumulated[to_keep]
    )
    densification_variables.visibility_count = densification_variables.visibility_count[
        to_keep
    ]
    densification_variables.max_2d_radii = densification_variables.max_2d_radii[to_keep]


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def densify_gaussians(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    densification_variables: DensificationVariables,
    scene_radius,
    optimizer,
    i,
):
    if i <= 5000:
        accumulate_mean_2d_gradients(densification_variables)
        gradient_threshold = 0.0002
        if (i >= 500) and (i % 100 == 0):
            average_gradients = (
                densification_variables.mean_2d_gradients_accumulated
                / densification_variables.visibility_count
            )
            average_gradients[average_gradients.isnan()] = 0.0

            scales = torch.exp(gaussian_cloud_parameters["log_scales"])
            max_scales = torch.max(scales, dim=1).values
            scale_threshold = 0.01 * scene_radius
            to_clone = (average_gradients >= gradient_threshold) & (
                max_scales <= scale_threshold
            )
            new_params = {
                parameter_name: parameter[to_clone]
                for parameter_name, parameter in gaussian_cloud_parameters.items()
                if parameter_name not in ["camera_matrices", "camera_center"]
            }
            cat_params_to_optimizer(new_params, gaussian_cloud_parameters, optimizer)
            num_pts = gaussian_cloud_parameters["means"].shape[0]

            padded_gradients = torch.zeros(num_pts, device="cuda")
            padded_gradients[: average_gradients.shape[0]] = average_gradients
            to_split = torch.logical_and(
                padded_gradients >= gradient_threshold,
                torch.max(
                    torch.exp(gaussian_cloud_parameters["log_scales"]), dim=1
                ).values
                > 0.01 * scene_radius,
            )
            n = 2  # number to split into
            new_params = {
                k: v[to_split].repeat(n, 1)
                for k, v in gaussian_cloud_parameters.items()
                if k not in ["camera_matrices", "camera_center"]
            }
            stds = torch.exp(gaussian_cloud_parameters["log_scales"])[to_split].repeat(
                n, 1
            )
            means = torch.zeros((stds.size(0), 3), device="cuda")
            samples = torch.normal(mean=means, std=stds)
            rots = build_rotation(
                gaussian_cloud_parameters["rotation_quaternions"][to_split]
            ).repeat(n, 1, 1)
            new_params["means"] += torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1)
            new_params["log_scales"] = torch.log(
                torch.exp(new_params["log_scales"]) / (0.8 * n)
            )
            cat_params_to_optimizer(new_params, gaussian_cloud_parameters, optimizer)
            num_pts = gaussian_cloud_parameters["means"].shape[0]

            densification_variables.mean_2d_gradients_accumulated = torch.zeros(
                num_pts, device="cuda"
            )
            densification_variables.visibility_count = torch.zeros(
                num_pts, device="cuda"
            )
            densification_variables.max_2d_radii = torch.zeros(num_pts, device="cuda")
            to_remove = torch.cat(
                (
                    to_split,
                    torch.zeros(n * to_split.sum(), dtype=torch.bool, device="cuda"),
                )
            )
            remove_points(
                to_remove, gaussian_cloud_parameters, densification_variables, optimizer
            )

            remove_threshold = 0.25 if i == 5000 else 0.005
            to_remove = (
                torch.sigmoid(gaussian_cloud_parameters["opacity_logits"])
                < remove_threshold
            ).squeeze()
            if i >= 3000:
                big_points_ws = (
                    torch.exp(gaussian_cloud_parameters["log_scales"]).max(dim=1).values
                    > 0.1 * scene_radius
                )
                to_remove = torch.logical_or(to_remove, big_points_ws)
            remove_points(
                to_remove, gaussian_cloud_parameters, densification_variables, optimizer
            )

            torch.cuda.empty_cache()

        if i > 0 and i % 3000 == 0:
            new_params = {
                "opacity_logits": inverse_sigmoid(
                    torch.ones_like(gaussian_cloud_parameters["opacity_logits"]) * 0.01
                )
            }
            update_params_and_optimizer(
                new_params, gaussian_cloud_parameters, optimizer
            )
