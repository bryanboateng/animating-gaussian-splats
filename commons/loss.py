import torch
import wandb

from commons.classes import (
    Background,
    Capture,
    DensificationVariables,
    GaussianCloudParameterNames,
    GaussianCloudReferenceState,
    Neighborhoods,
)
from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from external import (
    calc_ssim,
    build_rotation,
)


class GaussianCloud:
    def __init__(self, parameters: dict[str, torch.nn.Parameter]):
        self.means_2d = (
            torch.zeros_like(
                parameters[GaussianCloudParameterNames.means],
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        self.means_3d = parameters[GaussianCloudParameterNames.means]
        self.colors = parameters[GaussianCloudParameterNames.colors]
        self.rotations = torch.nn.functional.normalize(
            parameters[GaussianCloudParameterNames.rotation_quaternions]
        )
        self.opacities = torch.sigmoid(
            parameters[GaussianCloudParameterNames.opacities_logits]
        )
        self.scales = torch.exp(parameters[GaussianCloudParameterNames.log_scales])

    def get_renderer_format(self):
        return {
            "means3D": self.means_3d,
            "colors_precomp": self.colors,
            "rotations": self.rotations,
            "opacities": self.opacities,
            "scales": self.scales,
            "means2D": self.means_2d,
        }


def _l1_loss_v1(x, y):
    return torch.abs((x - y)).mean()


def _l1_loss_v2(x, y):
    return (torch.abs(x - y).sum(-1)).mean()


def _weighted_l2_loss_v1(x, y, w):
    return torch.sqrt(((x - y) ** 2) * w + 1e-20).mean()


def _weighted_l2_loss_v2(x, y, w):
    return torch.sqrt(((x - y) ** 2).sum(-1) * w + 1e-20).mean()


def _apply_exponential_transform_and_center_to_image(
    image: torch.Tensor,
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    camera_id: int,
):
    camera_matrices = gaussian_cloud_parameters[
        GaussianCloudParameterNames.camera_matrices
    ][camera_id]

    # Apply exponential transformation to the camera matrix parameters.
    # The exponential function ensures all values are positive and scales
    # the parameters non-linearly, which may be required for the transformation.
    exponential_camera_matrices = torch.exp(camera_matrices)[:, None, None]

    # Element-wise multiply the transformed camera matrices with the image.
    # This step applies the transformation to each pixel.
    scaled_image = exponential_camera_matrices * image

    camera_centers = gaussian_cloud_parameters[
        GaussianCloudParameterNames.camera_centers
    ][camera_id]

    # Add the camera center parameters to the scaled image.
    # This re-centers the transformed image based on the camera's center coordinates,
    # completing the adjustment.
    adjusted_image = scaled_image + camera_centers[:, None, None]

    return adjusted_image


def _calculate_image_loss(
    rendered_image,
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_capture: Capture,
):
    image = _apply_exponential_transform_and_center_to_image(
        rendered_image, gaussian_cloud_parameters, target_capture.camera.id_
    )
    return 0.8 * _l1_loss_v1(image, target_capture.image) + 0.2 * (
        1.0 - calc_ssim(image, target_capture.image)
    )


def _add_image_loss_grad(
    losses: dict[str, torch.Tensor],
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_capture: Capture,
):
    gaussian_cloud = GaussianCloud(parameters=gaussian_cloud_parameters)
    gaussian_cloud.means_2d.retain_grad()
    (
        rendered_image,
        radii,
        _,
    ) = Renderer(
        raster_settings=target_capture.camera.gaussian_rasterization_settings
    )(**gaussian_cloud.get_renderer_format())
    losses["im"] = _calculate_image_loss(
        rendered_image=rendered_image,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_capture=target_capture,
    )
    means_2d = (
        gaussian_cloud.means_2d
    )  # Gradient only accum from colour render for densification
    return means_2d, radii


def _add_segmentation_loss(
    losses: dict[str, torch.Tensor],
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_capture: Capture,
):
    gaussian_cloud = GaussianCloud(parameters=gaussian_cloud_parameters)
    gaussian_cloud.colors = gaussian_cloud_parameters[
        GaussianCloudParameterNames.segmentation_masks
    ]
    (
        segmentation_mask,
        _,
        _,
    ) = Renderer(
        raster_settings=target_capture.camera.gaussian_rasterization_settings
    )(**gaussian_cloud.get_renderer_format())
    losses["seg"] = 0.8 * _l1_loss_v1(
        segmentation_mask, target_capture.segmentation_mask
    ) + 0.2 * (1.0 - calc_ssim(segmentation_mask, target_capture.segmentation_mask))
    return gaussian_cloud


def _update_max_2d_radii_and_visibility_mask(
    radii, densification_variables: DensificationVariables
):
    radius_is_positive = radii > 0
    densification_variables.max_2d_radii[radius_is_positive] = torch.max(
        radii[radius_is_positive],
        densification_variables.max_2d_radii[radius_is_positive],
    )
    densification_variables.gaussian_is_visible_mask = radius_is_positive


def _combine_losses(losses: dict[str, torch.Tensor]):
    loss_weights = {
        "im": 1.0,
        "seg": 3.0,
        "rigid": 4.0,
        "rot": 4.0,
        "iso": 2.0,
        "floor": 2.0,
        "bg": 20.0,
        "soft_col_cons": 0.01,
    }
    weighted_losses = []
    for name, value in losses.items():
        weighted_loss = torch.tensor(loss_weights[name]) * value
        wandb.log({f"{name}-loss": weighted_loss})
        weighted_losses.append(weighted_loss)
    return torch.sum(torch.stack(weighted_losses))


def calculate_image_and_segmentation_loss(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_capture: Capture,
    densification_variables: DensificationVariables,
):
    losses = {}
    means_2d, radii = _add_image_loss_grad(
        losses=losses,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_capture=target_capture,
    )
    _ = _add_segmentation_loss(
        losses=losses,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_capture=target_capture,
    )
    densification_variables.means_2d = means_2d
    _update_max_2d_radii_and_visibility_mask(
        radii=radii, densification_variables=densification_variables
    )
    return _combine_losses(losses), densification_variables


def _add_image_loss_no_grad(
    losses: dict[str, torch.Tensor],
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_capture: Capture,
):
    gaussian_cloud = GaussianCloud(parameters=gaussian_cloud_parameters)
    (
        rendered_image,
        _,
        _,
    ) = Renderer(
        raster_settings=target_capture.camera.gaussian_rasterization_settings
    )(**gaussian_cloud.get_renderer_format())
    losses["im"] = _calculate_image_loss(
        rendered_image=rendered_image,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_capture=target_capture,
    )


def _quat_mult(q1, q2):
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack([w, x, y, z]).T


def _calculate_rigidity_loss(
    relative_rotation_quaternion: torch.Tensor,
    foreground_offset_to_neighbors: torch.Tensor,
    previous_offsets_to_neighbors: torch.Tensor,
    neighborhood_weights: torch.Tensor,
):
    rotation_matrix = build_rotation(relative_rotation_quaternion)
    curr_offset_in_prev_coord = (
        rotation_matrix.transpose(2, 1)[:, None]
        @ foreground_offset_to_neighbors[:, :, :, None]
    ).squeeze(-1)
    return _weighted_l2_loss_v2(
        curr_offset_in_prev_coord,
        previous_offsets_to_neighbors,
        neighborhood_weights,
    )


def _calculate_isometry_loss(
    foreground_offset_to_neighbors: torch.Tensor,
    initial_neighborhoods: Neighborhoods,
):
    curr_offset_mag = torch.sqrt((foreground_offset_to_neighbors**2).sum(-1) + 1e-20)
    return _weighted_l2_loss_v1(
        curr_offset_mag,
        initial_neighborhoods.distances,
        initial_neighborhoods.weights,
    )


def _calculate_background_loss(
    gaussian_cloud: GaussianCloud,
    is_foreground: torch.Tensor,
    initial_background: Background,
):
    bg_pts = gaussian_cloud.means_3d[~is_foreground]
    bg_rot = gaussian_cloud.rotations[~is_foreground]
    return _l1_loss_v2(bg_pts, initial_background.means) + _l1_loss_v2(
        bg_rot, initial_background.rotations
    )


def _add_additional_losses(
    losses: dict[str, torch.Tensor],
    segmentation_mask_gaussian_cloud: GaussianCloud,
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    initial_background: Background,
    initial_neighborhoods: Neighborhoods,
    previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
):
    foreground_mask = (
        gaussian_cloud_parameters[GaussianCloudParameterNames.segmentation_masks][:, 0]
        > 0.5
    ).detach()
    foreground_means = segmentation_mask_gaussian_cloud.means_3d[foreground_mask]
    foreground_rotations = segmentation_mask_gaussian_cloud.rotations[foreground_mask]
    relative_rotation_quaternion = _quat_mult(
        foreground_rotations,
        previous_timestep_gaussian_cloud_state.inverted_foreground_rotations,
    )
    foreground_neighbor_means = foreground_means[initial_neighborhoods.indices]
    foreground_offset_to_neighbors = (
        foreground_neighbor_means - foreground_means[:, None]
    )
    losses["rigid"] = _calculate_rigidity_loss(
        relative_rotation_quaternion=relative_rotation_quaternion,
        foreground_offset_to_neighbors=foreground_offset_to_neighbors,
        previous_offsets_to_neighbors=previous_timestep_gaussian_cloud_state.offsets_to_neighbors,
        neighborhood_weights=initial_neighborhoods.weights,
    )
    losses["rot"] = _weighted_l2_loss_v2(
        relative_rotation_quaternion[initial_neighborhoods.indices],
        relative_rotation_quaternion[:, None],
        initial_neighborhoods.weights,
    )
    losses["iso"] = _calculate_isometry_loss(
        foreground_offset_to_neighbors=foreground_offset_to_neighbors,
        initial_neighborhoods=initial_neighborhoods,
    )
    losses["floor"] = torch.clamp(foreground_means[:, 1], min=0).mean()
    losses["bg"] = _calculate_background_loss(
        gaussian_cloud=segmentation_mask_gaussian_cloud,
        is_foreground=foreground_mask,
        initial_background=initial_background,
    )
    losses["soft_col_cons"] = _l1_loss_v2(
        gaussian_cloud_parameters[GaussianCloudParameterNames.colors],
        previous_timestep_gaussian_cloud_state.colors,
    )


def calculate_full_loss(
    gaussian_cloud_parameters: dict[str, torch.nn.Parameter],
    target_capture: Capture,
    initial_background: Background,
    initial_neighborhoods: Neighborhoods,
    previous_timestep_gaussian_cloud_state: GaussianCloudReferenceState,
):
    losses: dict[str, torch.Tensor] = {}
    _add_image_loss_no_grad(
        losses=losses,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_capture=target_capture,
    )
    segmentation_mask_gaussian_cloud = _add_segmentation_loss(
        losses=losses,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        target_capture=target_capture,
    )
    _add_additional_losses(
        losses=losses,
        segmentation_mask_gaussian_cloud=segmentation_mask_gaussian_cloud,
        gaussian_cloud_parameters=gaussian_cloud_parameters,
        initial_background=initial_background,
        initial_neighborhoods=initial_neighborhoods,
        previous_timestep_gaussian_cloud_state=previous_timestep_gaussian_cloud_state,
    )

    return _combine_losses(losses)
