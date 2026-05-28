from __future__ import annotations

"""
Losses for the Implicit Neural Parametric Curve model.

All curve losses operate on the continuous representation γ(t) by:
  1. Querying the model at a batch of t values
  2. Computing differentiable integrals via Monte-Carlo approximation
     (i.e. mean over sampled t values)

Losses:
  - curve_mse_loss    : ∫ ||γ(t) - γ_gt(t)||² dt   (main fitting loss)
  - curve_length_loss : ∫ ||γ'(t)|| dt              (arc-length regulariser)
  - curve_smooth_loss : ∫ ||γ''(t)||² dt             (curvature / smoothness)
  - sdf_loss          : MSE(pred_sdf, gt_sdf)        (surface fitting)
  - combined_loss     : weighted sum of all above
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_derivative(
    model_fn,
    t: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute γ(t) and dγ/dt via autograd.

    Args:
        model_fn: callable (t, **kwargs) → (B, N, 3)
        t:        (B, N) — requires_grad must be True

    Returns:
        gamma:   (B, N, 3)
        dgamma:  (B, N, 3)  — first derivative w.r.t. t
    """
    t = t.detach().requires_grad_(True)                    # чистый leaf для autograd
    gamma = model_fn(t, **kwargs)                          # (B, N, 3)

    dgamma_xyz = []
    for c in range(3):
        g = torch.autograd.grad(
            outputs=gamma[..., c].sum(),
            inputs=t,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, N)
        dgamma_xyz.append(g)

    dgamma = torch.stack(dgamma_xyz, dim=-1)               # (B, N, 3)
    return gamma, dgamma


def _second_derivative(
    model_fn,
    t: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute γ(t), γ'(t), γ''(t) via autograd.

    Returns:
        gamma:    (B, N, 3)
        dgamma:   (B, N, 3)
        d2gamma:  (B, N, 3)
    """
    t = t.detach().requires_grad_(True)                    # чистый leaf для autograd
    gamma = model_fn(t, **kwargs)                          # (B, N, 3)

    dgamma_xyz = []
    d2gamma_xyz = []

    for c in range(3):
        # First derivative
        dg = torch.autograd.grad(
            outputs=gamma[..., c].sum(),
            inputs=t,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, N)
        dgamma_xyz.append(dg)

        # Second derivative
        d2g = torch.autograd.grad(
            outputs=dg.sum(),
            inputs=t,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, N)
        d2gamma_xyz.append(d2g)

    dgamma  = torch.stack(dgamma_xyz,  dim=-1)             # (B, N, 3)
    d2gamma = torch.stack(d2gamma_xyz, dim=-1)             # (B, N, 3)
    return gamma, dgamma, d2gamma


# ---------------------------------------------------------------------------
# Individual losses
# ---------------------------------------------------------------------------

def curve_mse_loss(
    gamma: torch.Tensor,
    gt_points: torch.Tensor,
) -> torch.Tensor:
    """
    Monte-Carlo approximation of ∫ ||γ(t) - γ_gt(t)||² dt.
    """
    return F.mse_loss(gamma, gt_points)



def curve_plane_loss(
    gamma: torch.Tensor,
    gt_points: torch.Tensor,
) -> torch.Tensor:
    """
    Auxiliary plane-aware loss.

    Computes MSE in each anatomical projection separately:
    - XY
    - XZ
    - YZ

    This helps when the model reduces overall 3D error but still underfits one
    or two projections.
    """
    loss_xy = F.mse_loss(gamma[..., [0, 1]], gt_points[..., [0, 1]])
    loss_xz = F.mse_loss(gamma[..., [0, 2]], gt_points[..., [0, 2]])
    loss_yz = F.mse_loss(gamma[..., [1, 2]], gt_points[..., [1, 2]])
    return (loss_xy + loss_xz + loss_yz) / 3.0


def curve_length_loss(
    gamma: torch.Tensor,
    gt_points: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Compare total polyline length of the predicted curve with the total
    polyline length of the GT curve at the same sampled t values.

    Both ``gamma`` and ``gt_points`` are expected to be sampled at the same
    sorted random t values (this is how sample_gt_at_random_t works in
    train_implicit.py), so the polyline approximation of the integral
    ∫_0^1 ||γ'(t)|| dt is consistent between pred and GT.

    Loss is a relative squared deviation, averaged over the batch:
        loss = mean_b ((L_pred_b - L_gt_b) / (L_gt_b + eps))^2

    Args:
        gamma:     (B, N, 3) — predicted curve points
        gt_points: (B, N, 3) — GT curve points at the same parameter values

    Returns:
        scalar loss
    """
    pred_seg = gamma[:, 1:] - gamma[:, :-1]                # (B, N-1, 3)
    gt_seg = gt_points[:, 1:] - gt_points[:, :-1]          # (B, N-1, 3)

    pred_len = torch.norm(pred_seg, dim=-1).sum(dim=1)     # (B,)
    gt_len = torch.norm(gt_seg, dim=-1).sum(dim=1)         # (B,)

    rel = (pred_len - gt_len) / (gt_len + eps)
    return (rel ** 2).mean()

def curve_speed_loss(
    dgamma: torch.Tensor,
) -> torch.Tensor:
    """
    Stable regularizer for approximately arc-length parameterised curves.

    Instead of forcing one global absolute speed for all samples, penalize only
    speed *variation* along the curve. This is much more stable because centerlines
    have different natural total lengths.

    Args:
        dgamma:       (B, N, 3) — first derivative of curve w.r.t. t

    Returns:
        scalar loss
    """
    speed = torch.norm(dgamma, dim=-1)                     # (B, N)
    mean_speed = speed.mean(dim=1, keepdim=True).clamp_min(1e-6)
    speed_norm = speed / mean_speed
    return ((speed_norm - 1.0) ** 2).mean()

def curve_smooth_loss(
    d2gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Penalises curvature: ∫ ||γ''(t)||² dt.

    Args:
        d2gamma: (B, N, 3) — second derivative of curve w.r.t. t

    Returns:
        scalar loss
    """
    return (d2gamma ** 2).mean()


def sdf_loss(
    pred_sdf: torch.Tensor,
    gt_sdf: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    MSE between predicted and ground-truth Signed Distance Function.

    Optionally weighted by a mask (e.g. give higher weight near the surface).

    Args:
        pred_sdf: (B, 1, X, Y, Z)
        gt_sdf:   (B, 1, X, Y, Z)
        mask:     (B, 1, X, Y, Z) optional weight map

        Returns:
        scalar loss
    """
    diff = (pred_sdf - gt_sdf) ** 2
    if mask is not None:
        diff = diff * mask
    return diff.mean()


def sample_volume_at_curve_points(
    volume: torch.Tensor,
    gamma: torch.Tensor,
) -> torch.Tensor:
    """
    Sample a 3D volume at predicted curve points using trilinear interpolation.

    Args:
        volume: (B, 1, X, Y, Z)
        gamma:  (B, N, 3) normalized coords in approximately [-1, 1]
               interpreted as (x, y, z)

    Returns:
        values: (B, N)
    """
    B, _, X, Y, Z = volume.shape
    _, N, _ = gamma.shape

    gx = ((gamma[..., 0] + 1.0) * 0.5) * (X - 1)
    gy = ((gamma[..., 1] + 1.0) * 0.5) * (Y - 1)
    gz = ((gamma[..., 2] + 1.0) * 0.5) * (Z - 1)

    gx = gx.clamp(0, X - 1)
    gy = gy.clamp(0, Y - 1)
    gz = gz.clamp(0, Z - 1)

    x0 = gx.floor().long().clamp(0, X - 1)
    y0 = gy.floor().long().clamp(0, Y - 1)
    z0 = gz.floor().long().clamp(0, Z - 1)
    x1 = (x0 + 1).clamp(0, X - 1)
    y1 = (y0 + 1).clamp(0, Y - 1)
    z1 = (z0 + 1).clamp(0, Z - 1)

    wx = gx - x0.float()
    wy = gy - y0.float()
    wz = gz - z0.float()

    vol = volume[:, 0]
    b_idx = torch.arange(B, device=volume.device).view(B, 1).expand(B, N)

    c000 = vol[b_idx, x0, y0, z0]
    c001 = vol[b_idx, x0, y0, z1]
    c010 = vol[b_idx, x0, y1, z0]
    c011 = vol[b_idx, x0, y1, z1]
    c100 = vol[b_idx, x1, y0, z0]
    c101 = vol[b_idx, x1, y0, z1]
    c110 = vol[b_idx, x1, y1, z0]
    c111 = vol[b_idx, x1, y1, z1]

    c00 = c000 * (1.0 - wz) + c001 * wz
    c01 = c010 * (1.0 - wz) + c011 * wz
    c10 = c100 * (1.0 - wz) + c101 * wz
    c11 = c110 * (1.0 - wz) + c111 * wz

    c0 = c00 * (1.0 - wy) + c01 * wy
    c1 = c10 * (1.0 - wy) + c11 * wy

    values = c0 * (1.0 - wx) + c1 * wx
    return values


def curve_inside_loss(
    gamma: torch.Tensor,
    gt_sdf: torch.Tensor,
) -> torch.Tensor:
    """
    Penalize predicted curve points that fall outside the vessel.

    Assumes GT SDF is negative inside, ~0 on the boundary, positive outside.
    """
    sdf_on_curve = sample_volume_at_curve_points(gt_sdf, gamma)
    return F.relu(sdf_on_curve).mean()


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class ImplicitCurveLoss(nn.Module):
    """
    Weighted combination of all losses.

    Usage during training:
        loss_fn = ImplicitCurveLoss(w_curve=1.0, w_length=0.1,
                                    w_smooth=0.01, w_sdf=1.0)

        # For each branch, call compute_curve_losses with a model_fn closure
        losses = loss_fn(
            model_fn_left, model_fn_right,
            t_left, t_right,
            gt_left, gt_right,
            pred_sdf, gt_sdf,
        )
        losses['total'].backward()
    """

    def __init__(
        self,
        w_curve:  float = 1.0,
        w_length: float = 0.1,
        w_smooth: float = 0.01,
        w_sdf:    float = 1.0,
        w_plane:  float = 0.0,
        w_inside: float = 0.0,
        w_speed: float = 0.0,
        **_legacy_kwargs,  # tolerate removed args (e.g. arc_length_target_speed)
    ):
        super().__init__()
        self.w_speed = w_speed
        self.w_curve  = w_curve
        self.w_length = w_length
        self.w_smooth = w_smooth
        self.w_sdf    = w_sdf
        self.w_plane  = w_plane
        self.w_inside = w_inside

    def compute_curve_losses(
        self,
        model_fn,
        t: torch.Tensor,
        gt_points: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """
        Compute all curve-related losses for one branch.

        Args:
            model_fn:  callable (t, **kwargs) → (B, N, 3)
            t:         (B, N) — sampled parameter values in [0, 1]
            gt_points: (B, N, 3) — GT points at the same t values
            **kwargs:  extra args forwarded to model_fn

        Returns:
            dict with keys: 'curve', 'length', 'smooth'
        """
        zero = torch.zeros(1, device=t.device)[0]

        if self.w_smooth > 0.0:
            gamma, dgamma, d2gamma = _second_derivative(model_fn, t, **kwargs)
            l_smooth = curve_smooth_loss(d2gamma)
        elif self.w_speed > 0.0:
            gamma, dgamma = _first_derivative(model_fn, t, **kwargs)
            l_smooth = zero
        else:
            gamma = model_fn(t, **kwargs)
            dgamma = None
            l_smooth = zero

        if self.w_length > 0.0:
            # Length loss is a polyline-based comparison of total lengths,
            # it does not require autograd over t.
            l_length = curve_length_loss(gamma, gt_points)
        else:
            l_length = zero

        if self.w_speed > 0.0 and dgamma is not None:
            l_speed = curve_speed_loss(dgamma)
        else:
            l_speed = zero

        l_curve = curve_mse_loss(gamma, gt_points)
        l_plane = curve_plane_loss(gamma, gt_points)

        return {
            "curve":  l_curve,
            "plane":  l_plane,
            "length": l_length,
            "smooth": l_smooth,
            "gamma":  gamma,
            "speed": l_speed,
        }

    def forward(
        self,
        model_fn_left,
        model_fn_right,
        t_left:    torch.Tensor,
        t_right:   torch.Tensor,
        gt_left:   torch.Tensor,
        gt_right:  torch.Tensor,
        pred_sdf:  torch.Tensor,
        gt_sdf:    torch.Tensor,
        sdf_weight_mask: torch.Tensor | None = None,
        active_branch: str = "both",
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            model_fn_left / right: callable (t) → (B, N, 3)
            t_left / right:        (B, N)
            gt_left / right:       (B, N, 3)
            pred_sdf:              (B, 1, X, Y, Z)
            gt_sdf:                (B, 1, X, Y, Z)
            sdf_weight_mask:       (B, 1, X, Y, Z) optional
            active_branch:         'both', 'left' or 'right'. Only selected
                                   curve branch contributes gradients to total.

        Returns:
            dict with keys: 'total', 'curve_left', 'curve_right',
                            'length_left', 'length_right',
                            'smooth_left', 'smooth_right', 'sdf'
        """
        if active_branch not in ("both", "left", "right"):
            raise ValueError(f"active_branch must be 'both', 'left' or 'right', got {active_branch!r}")

        zero = torch.zeros(1, device=t_left.device)[0]
        left_losses = None
        right_losses = None

        if active_branch in ("both", "left"):
            left_losses = self.compute_curve_losses(model_fn_left, t_left, gt_left)
        if active_branch in ("both", "right"):
            right_losses = self.compute_curve_losses(model_fn_right, t_right, gt_right)

        if self.w_sdf > 0.0:
            l_sdf = sdf_loss(pred_sdf, gt_sdf, sdf_weight_mask)
        else:
            l_sdf = torch.zeros(1, device=t_left.device)[0]

        if self.w_inside > 0.0 and left_losses is not None:
            l_inside_left = curve_inside_loss(left_losses["gamma"], gt_sdf)
        else:
            l_inside_left = zero

        if self.w_inside > 0.0 and right_losses is not None:
            l_inside_right = curve_inside_loss(right_losses["gamma"], gt_sdf)
        else:
            l_inside_right = zero

        total = self.w_sdf * l_sdf

        if left_losses is not None:
            total = total + (
                self.w_curve  * left_losses["curve"]
                + self.w_plane  * left_losses["plane"]
                + self.w_length * left_losses["length"]
                + self.w_smooth * left_losses["smooth"]
                + self.w_inside * l_inside_left
                + self.w_speed  * left_losses["speed"]
            )

        if right_losses is not None:
            total = total + (
                self.w_curve  * right_losses["curve"]
                + self.w_plane  * right_losses["plane"]
                + self.w_length * right_losses["length"]
                + self.w_smooth * right_losses["smooth"]
                + self.w_inside * l_inside_right
                + self.w_speed  * right_losses["speed"]
            )

        return {
            "total":        total,
            "curve_left":   left_losses["curve"] if left_losses is not None else zero,
            "curve_right":  right_losses["curve"] if right_losses is not None else zero,
            "plane_left":   left_losses["plane"] if left_losses is not None else zero,
            "plane_right":  right_losses["plane"] if right_losses is not None else zero,
            "length_left":  left_losses["length"] if left_losses is not None else zero,
            "length_right": right_losses["length"] if right_losses is not None else zero,
            "smooth_left":  left_losses["smooth"] if left_losses is not None else zero,
            "smooth_right": right_losses["smooth"] if right_losses is not None else zero,
            "speed_left":  left_losses["speed"] if left_losses is not None else zero,
            "speed_right": right_losses["speed"] if right_losses is not None else zero,
            "inside_left":  l_inside_left,
            "inside_right": l_inside_right,
            "sdf":          l_sdf,
        }


# ---------------------------------------------------------------------------
# GT SDF computation utility (numpy, used in dataset preprocessing)
# ---------------------------------------------------------------------------

def compute_gt_sdf(mask: "np.ndarray") -> "np.ndarray":
    """
    Compute a Signed Distance Function from a binary mask.

    Inside  (mask == 1): negative distance to surface
    Outside (mask == 0): positive distance to surface

    Uses scipy's distance_transform_edt for efficiency.

    Args:
        mask: (X, Y, Z) binary uint8 array

    Returns:
        sdf: (X, Y, Z) float32 array
    """
    from scipy.ndimage import distance_transform_edt
    import numpy as np

    dist_inside  = distance_transform_edt(mask == 1).astype(np.float32)
    dist_outside = distance_transform_edt(mask == 0).astype(np.float32)

    sdf = dist_outside - dist_inside
    return sdf


def compute_gt_sdf_surface_weighted_mask(
    sdf: "np.ndarray",
    bandwidth: float = 5.0,
) -> "np.ndarray":
    """
    Create a weight mask that emphasises voxels near the surface (|SDF| < bandwidth).

    Args:
        sdf:       (X, Y, Z) float32
        bandwidth: half-width of the surface band in voxels

    Returns:
        weight: (X, Y, Z) float32 in [0, 1]
    """
    import numpy as np
    weight = np.exp(-0.5 * (sdf / bandwidth) ** 2).astype(np.float32)
    return weight
