from __future__ import annotations

"""
Implicit Neural Parametric Curve model for aorta centerline extraction.

Architecture overview:
  1. 3D U-Net encoder (EfficientNetV2-B0 backbone, converted to 3D)
     - Produces dense 3D feature maps at multiple scales
  2. SDF head: predicts Signed Distance Function from the decoder output
     - Replaces binary segmentation mask
     - Gives richer geometric signal (gradients everywhere, not just on boundary)
  3. Implicit Curve Decoder (per branch: left / right iliac + main aorta)
     - Uses a global pooled bottleneck descriptor from the full 3D volume
     - Positional encoding of parameter t ∈ [0, 1] (NeRF-style Fourier features)
     - Branch embedding (0 = left, 1 = right)
     - Small MLP: [PE(t) | branch_emb | global_3d_context] → (x, y, z)

Losses (defined in losses_implicit.py):
  - L_curve  : MSE between γ(t_i) and GT point at arc-length parameter t_i
  - L_length : ∫|γ'(t)|dt  — penalises non-unit-speed (encourages arc-length param)
  - L_smooth : ∫|γ''(t)|²dt — penalises curvature (smoothness regulariser)
  - L_sdf    : MSE between predicted SDF and GT SDF
"""

import math
import torch
import torch.nn as nn
from nnspt.segmentation.unet import Unet
from aaa.models.layer_convertors import (
    convert_inplace,
    LayerConvertorNNSPT,
    LayerConvertorSm,
)


# ---------------------------------------------------------------------------
# Positional Encoding  (NeRF / Transformer style)
# ---------------------------------------------------------------------------

class FourierPositionalEncoding(nn.Module):
    """
    Maps scalar t ∈ [0, 1] to a high-dimensional Fourier feature vector:
        PE(t) = [t, sin(2^0 π t), cos(2^0 π t), ..., sin(2^(L-1) π t), cos(2^(L-1) π t)]
    Output dimension: 1 + 2*num_freqs
    """

    def __init__(self, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        freqs = 2.0 ** torch.arange(num_freqs).float() * math.pi
        self.register_buffer("freqs", freqs)

    @property
    def out_dim(self) -> int:
        return 1 + 2 * self.num_freqs

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        args = t * self.freqs
        return torch.cat([t, torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# 3D Global Context Sampler
# ---------------------------------------------------------------------------

class GlobalContextSampler(nn.Module):
    """
    Compresses a 3D feature volume (B, C, X, Y, Z) into a single global 3D-aware
    descriptor (B, C').

    Unlike the previous 1D axial profile, this module does not collapse the
    volume to a sequence along one axis and does not assume that t corresponds
    to Z-progression. It aggregates context from the full 3D bottleneck.
    """

    def __init__(self, in_channels: int, out_channels: int, pool_size: int = 1, log_shapes: bool = False):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.pool_size = int(pool_size)
        self.pool = nn.AdaptiveAvgPool3d(self.pool_size)
        self.log_shapes = bool(log_shapes)
        self._shape_logged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, X, Y, Z)
        Returns:
            global_ctx: (B, C' * pool_size^3)
        """
        do_log = self.log_shapes and not self._shape_logged
        if do_log:
            print(f"[GlobalContextSampler] input (bottleneck) shape: {tuple(x.shape)}")
        x = self.proj(x)
        if do_log:
            print(f"[GlobalContextSampler] after self.proj shape:    {tuple(x.shape)}")
        x = self.pool(x)
        if do_log:
            print(f"[GlobalContextSampler] after self.pool shape:    {tuple(x.shape)}  (pool_size={self.pool_size})")
        out = x.flatten(1)
        if do_log:
            print(f"[GlobalContextSampler] flattened output shape:   {tuple(out.shape)}")
            self._shape_logged = True
        return out


# ---------------------------------------------------------------------------
# Local Feature Sampler (point-conditioned features)
# ---------------------------------------------------------------------------

class LocalFeatureSampler(nn.Module):
    """
    Samples per-point local features from a 3D feature volume at arbitrary
    normalised coordinates in [-1, 1]^3.

    Steps:
      1. Project a dense decoder feature map (B, C_in, X, Y, Z) to a smaller
         channel dimension (B, C_local, X, Y, Z) via a 1x1x1 conv + GroupNorm.
      2. For each query point γ(t) ∈ [-1, 1]^3, sample the projected volume
         using trilinear interpolation, producing (B, N, C_local).

    The coordinate convention matches sample_volume_at_curve_points in
    losses_implicit.py: gamma[..., 0] → X axis, gamma[..., 1] → Y axis,
    gamma[..., 2] → Z axis. This is implemented manually (not F.grid_sample)
    to keep ordering explicit and to support second-order autograd if needed.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.GELU(),
        )
        self.out_channels = out_channels

    @staticmethod
    def _trilinear_sample(
        volume: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        volume: (B, C, X, Y, Z)
        gamma:  (B, N, 3) in approx [-1, 1], ordering (x, y, z) → (X, Y, Z)
        returns: (B, N, C)
        """
        B, C, X, Y, Z = volume.shape
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

        wx = (gx - x0.float()).unsqueeze(-1)  # (B, N, 1)
        wy = (gy - y0.float()).unsqueeze(-1)
        wz = (gz - z0.float()).unsqueeze(-1)

        # volume: (B, C, X, Y, Z) → permute to (B, X, Y, Z, C) for gather-like indexing
        vol = volume.permute(0, 2, 3, 4, 1)  # (B, X, Y, Z, C)
        b_idx = torch.arange(B, device=volume.device).view(B, 1).expand(B, N)

        c000 = vol[b_idx, x0, y0, z0]  # (B, N, C)
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

        values = c0 * (1.0 - wx) + c1 * wx  # (B, N, C)
        return values

    def forward(
        self,
        feature_volume: torch.Tensor,
        gamma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            feature_volume: (B, C_in, X, Y, Z)
            gamma:          (B, N, 3) in [-1, 1]
        Returns:
            local_feats: (B, N, C_local)
        """
        feats = self.proj(feature_volume)
        return self._trilinear_sample(feats, gamma)


# ---------------------------------------------------------------------------
# Implicit Curve MLP
# ---------------------------------------------------------------------------

class ImplicitCurveMLP(nn.Module):
    """
    Maps (PE(t), branch_embedding, context) → (x, y, z) in normalised
    coordinates (unbounded; GT is normalised to approximately [-1, 1]).

    Architecture:
        Linear → GELU → ... → Linear(3)
    with a skip connection from the input to the middle layer (NeRF-style).
    No output activation — tanh was removed because GT Z-coordinates can
    slightly exceed ±1 after voxel-space normalisation, causing mode collapse.
    """

    def __init__(
        self,
        pe_dim: int,
        context_dim: int,
        branch_emb_dim: int = 8,
        hidden_dim: int = 256,
        num_layers: int = 6,
        num_branches: int = 2,
        local_feat_dim: int = 0,
    ):
        super().__init__()

        self.branch_emb = nn.Embedding(num_branches, branch_emb_dim)
        self.local_feat_dim = local_feat_dim
        in_dim = pe_dim + branch_emb_dim + context_dim + local_feat_dim

        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i == num_layers // 2:
                layers.append(nn.Linear(hidden_dim + in_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        self.layers = nn.ModuleList(layers)
        self.skip_at = num_layers // 2
        self.num_layers = num_layers
        self.out = nn.Linear(hidden_dim, 3)

    def forward(
        self,
        pe: torch.Tensor,
        ctx: torch.Tensor,
        branch_ids: torch.Tensor,
        local_feats: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = pe.shape

        b_emb = self.branch_emb(branch_ids)
        b_emb = b_emb.unsqueeze(1).expand(-1, N, -1)

        parts = [pe, b_emb, ctx]
        if self.local_feat_dim > 0:
            if local_feats is None:
                local_feats = torch.zeros(
                    B, N, self.local_feat_dim, device=pe.device, dtype=pe.dtype
                )
            parts.append(local_feats)
        x_in = torch.cat(parts, dim=-1)
        x = x_in.reshape(B * N, -1)
        x_in_flat = x

        layer_idx = 0
        for i in range(self.num_layers):
            linear = self.layers[layer_idx]
            act = self.layers[layer_idx + 1]
            layer_idx += 2

            if i == self.skip_at:
                x = torch.cat([x, x_in_flat], dim=-1)

            x = act(linear(x))

        coords = self.out(x)
        coords = coords.view(B, N, 3)
        return coords


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class ImplicitCurveNet(nn.Module):
    """
    Full model:
      - 3D U-Net backbone (EfficientNetV2-B0, depth=5)
      - SDF head (predicts signed distance to aorta surface)
      - Implicit curve decoder for left and right branches

    Forward signature:
        x: (B, C_in, X, Y, Z)  — input CT voxels
        t_left:  (B, N_t) — query parameters for left branch
        t_right: (B, N_t) — query parameters for right branch

    Returns:
        sdf:         (B, 1, X, Y, Z)  — predicted SDF
        curve_left:  (B, N_t, 3)      — left branch points in [-1, 1]
        curve_right: (B, N_t, 3)      — right branch points in [-1, 1]
    """

    def __init__(
        self,
        in_channels: int = 1,
        depth: int = 5,
        pe_num_freqs: int = 10,
        context_channels: int = 64,
        mlp_hidden_dim: int = 256,
        mlp_num_layers: int = 6,
        branch_emb_dim: int = 8,
        local_feat_channels: int = 32,
        num_refine_passes: int = 1,
        refine_scale: float = 0.25,
        coarse_scale: float = 1.0,
        base_xy_slope: float = 0.12,
        base_branch_y_slope: float = 0.06,
        affine_max_shift: float = 0.6,
        affine_max_log_scale: float = 0.35,
        context_pool_size: int = 1,
        log_shapes: bool = False,
    ):
        super().__init__()

        self.log_shapes = bool(log_shapes)
        self._encdec_shape_logged = False

        unet = Unet(
            in_channels=in_channels,
            out_channels=1,
            encoder="timm-efficientnetv2-b0",
            depth=depth,
        )
        convert_inplace(unet, LayerConvertorNNSPT)
        convert_inplace(unet, LayerConvertorSm)

        self.encoder = unet.encoder
        self.decoder = unet.decoder

        decoder_out_channels = 32
        self.sdf_head = nn.Sequential(
            nn.Conv3d(decoder_out_channels, 16, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(16, 1, kernel_size=1),
        )

        # nnspt.Unet.encoder returns features ordered fine→coarse
        # (features[0] = identity/raw input, features[-1] = deepest bottleneck).
        # We want the deepest semantic features.
        bottleneck_channels = self.encoder.out_channels[-3]
        self.context_pool_size = int(context_pool_size)
        self.context_sampler = GlobalContextSampler(
            in_channels=bottleneck_channels,
            out_channels=context_channels,
            pool_size=self.context_pool_size,
            log_shapes=self.log_shapes,
        )
        # GlobalContextSampler flattens pooled volume to (B, context_channels * pool_size^3)
        ctx_dim = context_channels * (self.context_pool_size ** 3)

        self.local_feat_channels = max(0, int(local_feat_channels))
        self.num_refine_passes = max(0, int(num_refine_passes))
        self.refine_scale = float(refine_scale)
        self.coarse_scale = float(coarse_scale)
        self.base_xy_slope = float(base_xy_slope)
        self.base_branch_y_slope = float(base_branch_y_slope)
        self.affine_max_shift = float(affine_max_shift)
        self.affine_max_log_scale = float(affine_max_log_scale)
        self.last_delta_mean = 0.0
        self.last_delta_max = 0.0
        self.last_coarse_delta_mean = 0.0
        self.last_coarse_delta_max = 0.0
        self.last_affine_translation_mean = 0.0
        self.last_affine_translation_max = 0.0
        self.last_affine_scale_mean = 1.0
        self.last_affine_scale_min = 1.0
        self.last_affine_scale_max = 1.0
        if self.local_feat_channels > 0:
            self.local_sampler = LocalFeatureSampler(
                in_channels=decoder_out_channels,
                out_channels=self.local_feat_channels,
            )
        else:
            self.local_sampler = None

        self.pe = FourierPositionalEncoding(num_freqs=pe_num_freqs)
        pe_dim = self.pe.out_dim

        self.coarse_mlp = ImplicitCurveMLP(
            pe_dim=pe_dim,
            context_dim=ctx_dim,
            branch_emb_dim=branch_emb_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            num_branches=2,
            local_feat_dim=0,
        )
        self.refine_mlp = ImplicitCurveMLP(
            pe_dim=pe_dim,
            context_dim=ctx_dim,
            branch_emb_dim=branch_emb_dim,
            hidden_dim=mlp_hidden_dim,
            num_layers=mlp_num_layers,
            num_branches=2,
            local_feat_dim=self.local_feat_channels,
        )

        affine_hidden_dim = max(32, mlp_hidden_dim // 2)
        self.affine_branch_emb = nn.Embedding(2, branch_emb_dim)
        self.affine_head = nn.Sequential(
            nn.Linear(ctx_dim + branch_emb_dim, affine_hidden_dim),
            nn.GELU(),
            nn.Linear(affine_hidden_dim, 6),
        )

        # Important initialisation: both curve heads predict residuals.
        # Zero output heads make the initial curve equal to the simple tilted
        # base line instead of a random Fourier curve/ring.
        nn.init.zeros_(self.coarse_mlp.out.weight)
        nn.init.zeros_(self.coarse_mlp.out.bias)
        nn.init.zeros_(self.refine_mlp.out.weight)
        nn.init.zeros_(self.refine_mlp.out.bias)

        # Affine head predicts [translation_xyz, log_scale_xyz] residuals.
        # Zero output means identity transform: translation=0, scale=1.
        nn.init.zeros_(self.affine_head[-1].weight)
        nn.init.zeros_(self.affine_head[-1].bias)

    def _base_curve(self, t: torch.Tensor, branch_id: int) -> torch.Tensor:
        """
        Branch-aware tilted base curve.

        Both branches meet at the opposite end, t=1: (0, 0, 1).
        Moving towards t=0, they smoothly diverge in the axial plane:
            left  branch_id=0 → positive x/y direction,
            right branch_id=1 → negative x/y direction.

        This gives the zero-initialised residual heads a more anatomical prior:
        not two identical curves on top of each other, but a shared bifurcation
        root with left/right branches gradually separating away from t=1.
        """
        # z = 2.0 * t - 1.0
        # branch_sign = 1.0 if branch_id == 0 else -1.0
        # branch_sign = t.new_tensor(branch_sign)
        # divergence = 1.0 - t
        # x = branch_sign * self.base_xy_slope * divergence
        # y = branch_sign * self.base_branch_y_slope * divergence
        # return torch.stack([x, y, z], dim=-1)
        B, N = t.shape

        # Инициализация всех точек в нуле.
        # После bifurcation-centering это логично:
        # бифуркация находится около (0, 0, 0).
        base_point = torch.zeros(B, N, 3, device=t.device, dtype=t.dtype)

        return base_point

    def _predict_affine(
        self,
        global_ctx: torch.Tensor,
        branch_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predict a constrained branch-specific translation + anisotropic scale.

        Returns:
            translation: (B, 3), bounded to [-affine_max_shift, affine_max_shift]
            scale:       (B, 3), bounded to exp(±affine_max_log_scale)
        """
        branch_emb = self.affine_branch_emb(branch_ids)
        affine_in = torch.cat([global_ctx, branch_emb], dim=-1)
        raw = self.affine_head(affine_in)
        raw_translation, raw_log_scale = raw[:, :3], raw[:, 3:]
        translation = self.affine_max_shift * torch.tanh(raw_translation)
        log_scale = self.affine_max_log_scale * torch.tanh(raw_log_scale)
        scale = torch.exp(log_scale)
        return translation, scale

    def _encode_decode(self, x: torch.Tensor):
        """Run encoder + decoder, return (decoder_out, bottleneck_features).

        nnspt.Unet.encoder produces features ordered from fine to coarse:
            features[0]  — identity-skip of the raw input (1 channel)
            features[-1] — deepest semantic bottleneck (many channels)
        We need the deepest one for global context.
        """
        features = self.encoder(x)
        bottleneck = features[-3]
        decoder_out = self.decoder(*features)
        if self.log_shapes and not self._encdec_shape_logged:
            print(f"[ImplicitCurveNet] encoder input shape:          {tuple(x.shape)}")
            for i, f in enumerate(features):
                print(f"[ImplicitCurveNet] encoder features[{i}] shape:    {tuple(f.shape)}")
            print(f"[ImplicitCurveNet] bottleneck (features[-3]):    {tuple(bottleneck.shape)}")
            print(f"[ImplicitCurveNet] decoder_out shape:            {tuple(decoder_out.shape)}")
            self._encdec_shape_logged = True
        return decoder_out, bottleneck

    def _query_curve(
        self,
        bottleneck: torch.Tensor,
        decoder_out: torch.Tensor,
        t: torch.Tensor,
        branch_id: int,
    ) -> torch.Tensor:
        """
        Query with residual point-conditioned local feature refinement.

        Coarse pass:
            γ_base(t, branch_id) = branch-aware tilted line, mostly along z
            γ_can(t) = γ_base(t, branch_id) + coarse_scale * coarse_mlp(PE(t), global_ctx, branch_emb)
            γ₀(t)    = scale(I, branch) * γ_can(t) + translation(I, branch)
        Refinement pass:
            f_local(t) = LocalFeatureSampler(decoder_out, γ₀(t))
            Δγ(t)      = refine_mlp(PE(t), global_ctx, branch_emb, f_local(t))
            γ(t)       = γ₀(t) + refine_scale * Δγ(t)

        The local-feature coordinates are intentionally not detached: gradients
        can flow through the trilinear sampler into γ₀, so the coarse curve can
        learn where to query image features.

        Args:
            bottleneck:  (B, C_bn, X', Y', Z')
            decoder_out: (B, C_dec, X, Y, Z) — full-resolution decoder features
            t:           (B, N_t) in [0, 1]
            branch_id:   0 or 1
        Returns:
            points: (B, N_t, 3) — final refined curve
        """
        B, N = t.shape

        global_ctx = self.context_sampler(bottleneck)              # (B, C')
        ctx = global_ctx.unsqueeze(1).expand(-1, N, -1)            # (B, N, C')
        pe = self.pe(t)                                            # (B, N, pe_dim)
        branch_ids = torch.full((B,), branch_id, dtype=torch.long, device=t.device)

        base = self._base_curve(t, branch_id)                           # (B, N, 3)
        coarse_delta = self.coarse_mlp(pe, ctx, branch_ids)             # (B, N, 3)
        canonical = base + self.coarse_scale * coarse_delta             # (B, N, 3)
        translation, scale = self._predict_affine(global_ctx, branch_ids)
        coarse = canonical * scale.unsqueeze(1) + translation.unsqueeze(1)
        points = coarse
        delta = None

        if self.local_sampler is not None and self.num_refine_passes > 0:
            query_coords = coarse
            for _ in range(self.num_refine_passes):
                f_local = self.local_sampler(decoder_out, query_coords)      # (B, N, C_local)
                delta = self.refine_mlp(pe, ctx, branch_ids, f_local)        # (B, N, 3)
                points = coarse + self.refine_scale * delta                  # residual refinement
                query_coords = points                                        # optional iterative refinement, no detach

        with torch.no_grad():
            coarse_delta_norm = coarse_delta.norm(dim=-1)
            translation_norm = translation.norm(dim=-1)
            self.last_coarse_delta_mean = float(coarse_delta_norm.mean().detach().cpu())
            self.last_coarse_delta_max = float(coarse_delta_norm.max().detach().cpu())
            self.last_affine_translation_mean = float(translation_norm.mean().detach().cpu())
            self.last_affine_translation_max = float(translation_norm.max().detach().cpu())
            self.last_affine_scale_mean = float(scale.mean().detach().cpu())
            self.last_affine_scale_min = float(scale.min().detach().cpu())
            self.last_affine_scale_max = float(scale.max().detach().cpu())
            if delta is not None:
                delta_norm = delta.norm(dim=-1)
                self.last_delta_mean = float(delta_norm.mean().detach().cpu())
                self.last_delta_max = float(delta_norm.max().detach().cpu())
            else:
                self.last_delta_mean = 0.0
                self.last_delta_max = 0.0

        return points

    def forward(
        self,
        x: torch.Tensor,
        t_left: torch.Tensor,
        t_right: torch.Tensor,
    ):
        decoder_out, bottleneck = self._encode_decode(x)
        sdf = self.sdf_head(decoder_out)
        curve_left = self._query_curve(bottleneck, decoder_out, t_left, branch_id=0)
        curve_right = self._query_curve(bottleneck, decoder_out, t_right, branch_id=1)
        return sdf, curve_left, curve_right

    @torch.no_grad()
    def predict_curve(
        self,
        x: torch.Tensor,
        n_points: int = 256,
        branch_id: int = 0,
    ) -> torch.Tensor:
        assert x.shape[0] == 1, "predict_curve expects batch size 1"
        decoder_out, bottleneck = self._encode_decode(x)
        t = torch.linspace(0, 1, n_points, device=x.device).unsqueeze(0)
        points = self._query_curve(bottleneck, decoder_out, t, branch_id=branch_id)
        return points
