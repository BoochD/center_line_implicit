from __future__ import annotations

"""
Training script for ImplicitCurveNet.

Все параметры задаются в config_implicit.yaml.
Запуск:
    python train_implicit.py
    python train_implicit.py config_implicit.yaml   # явно указать путь к конфигу
"""

import cv2
import copy
import sys
import time
import json
import csv
import torch
import voxelmentations as V
import numpy as np
import nibabel as nib
import torch.nn.functional as F
import torch.optim.swa_utils as tsu

from pathlib import Path
from collections import defaultdict, OrderedDict
from torch.utils.data import Dataset, DataLoader
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

from aaa.utils import load_yaml_config
from aaa.utils import io
from aaa.geometry.misc import reshape

from model_implicit import ImplicitCurveNet
from losses_implicit import ImplicitCurveLoss, compute_gt_sdf, compute_gt_sdf_surface_weighted_mask
from visualize_implicit import compute_metrics, visualize_epoch

# ---------------------------------------------------------------------------
# Global config dict (заполняется из YAML в main())
# ---------------------------------------------------------------------------
config: dict = {}


# ---------------------------------------------------------------------------
# Arc-length parameterisation utilities
# ---------------------------------------------------------------------------

def arc_length_parameterise(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given an ordered (N, 3) array of 3D points, compute a robust arc-length
    parameterisation with strictly increasing t.

    Removes NaN/Inf points and collapses consecutive near-duplicates, because
    repeated points lead to repeated cumulative lengths and break [`interp1d()`](train_implicit.py:91).
    """
    points = np.asarray(points, dtype=np.float32)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected points with shape (N, 3), got {points.shape}")

    # Remove non-finite points first
    finite_mask = np.isfinite(points).all(axis=1)
    points = points[finite_mask]

    if len(points) == 0:
        points = np.zeros((2, 3), dtype=np.float32)
    elif len(points) == 1:
        points = np.concatenate([points, points], axis=0)

    # Remove consecutive duplicates / near-duplicates so t stays strictly increasing
    keep = np.ones(len(points), dtype=bool)
    if len(points) > 1:
        step = np.linalg.norm(np.diff(points, axis=0), axis=1)
        keep[1:] = step > 1e-6
        points = points[keep]

    if len(points) == 0:
        points = np.zeros((2, 3), dtype=np.float32)
    elif len(points) == 1:
        points = np.concatenate([points, points], axis=0)

    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    cum_lengths = np.concatenate([[0.0], np.cumsum(seg_lengths)])
    total_length = float(cum_lengths[-1])

    if total_length < 1e-8:
        t = np.linspace(0.0, 1.0, len(points), dtype=np.float32)
    else:
        t = (cum_lengths / total_length).astype(np.float32)
        # Make t strictly increasing for robust interpolation
        for i in range(1, len(t)):
            if t[i] <= t[i - 1]:
                t[i] = np.nextafter(t[i - 1], np.float32(2.0))
        if t[-1] > 1.0:
            t = t / t[-1]
            t[0] = 0.0
            t[-1] = 1.0

    return t.astype(np.float32), points.astype(np.float32)


def interpolate_curve_at_t(
    t_gt: np.ndarray,
    points_gt: np.ndarray,
    t_query: np.ndarray,
) -> np.ndarray:
    """
    Interpolate GT curve at arbitrary query t values.
    Input is sanitized to keep [`interp1d()`](train_implicit.py:105) stable.
    """
    t_gt = np.asarray(t_gt, dtype=np.float32)
    points_gt = np.asarray(points_gt, dtype=np.float32)
    t_query = np.asarray(t_query, dtype=np.float32)

    finite_mask = np.isfinite(t_gt) & np.isfinite(points_gt).all(axis=1)
    t_gt = t_gt[finite_mask]
    points_gt = points_gt[finite_mask]

    if len(t_gt) < 2:
        base = points_gt[0] if len(points_gt) > 0 else np.zeros(3, dtype=np.float32)
        return np.repeat(base[None], len(t_query), axis=0).astype(np.float32)

    keep = np.ones(len(t_gt), dtype=bool)
    keep[1:] = np.diff(t_gt) > 1e-8
    t_gt = t_gt[keep]
    points_gt = points_gt[keep]

    if len(t_gt) < 2:
        base = points_gt[0]
        return np.repeat(base[None], len(t_query), axis=0).astype(np.float32)

    interp = interp1d(
        t_gt,
        points_gt,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(points_gt[0], points_gt[-1]),
        assume_sorted=True,
    )
    out = interp(t_query).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ImplicitCurveDataset(Dataset):
    """
    Loads custom aorta data (imgs, masks, left/right knots).

    Preprocessing (done once at __init__):
      - Crop, reshape, normalise spacing
      - Compute GT SDF from mask
      - Arc-length parameterise left and right centerlines

    __getitem__ returns everything needed for one training step.
    """

    def __init__(
        self,
        datapath: Path,
        names: list[str],
        channels: dict,
        mode: str = "train",
        augment: bool = True,
    ):
        assert mode in ("train", "val", "test")
        self.mode = mode
        self.augment = augment and (mode == "train")
        self.channels = channels
        self.keys = list(names)

        self.imgs: dict[str, np.ndarray] = {}
        self.masks: dict[str, np.ndarray] = {}
        self.sdfs: dict[str, np.ndarray] = {}
        self.sdf_weights: dict[str, np.ndarray] = {}
        self.t_left: dict[str, np.ndarray] = {}
        self.pts_left: dict[str, np.ndarray] = {}
        self.t_right: dict[str, np.ndarray] = {}
        self.pts_right: dict[str, np.ndarray] = {}
        self.spacings: dict[str, np.ndarray] = {}
        self.shapes: list = []

        for name in names:
            print(f"[dataset] loading {name}")
            self._load_sample(datapath, name)

    def _load_sample(self, datapath: Path, name: str):
        imgpath = datapath / "custom" / "imgs" / (name + ".nii.gz")
        maskpath = datapath / "custom" / "masks" / (name + ".nii.gz")
        knotpath_l = datapath / "custom" / "knots" / (name + "_l.json")
        knotpath_r = datapath / "custom" / "knots" / (name + "_r.json")

        # ---- Load raw data ----
        img_nib = nib.load(imgpath)
        image = img_nib.dataobj[:]
        mask = nib.load(maskpath).dataobj[:]

        SIZE = image.shape[0]
        HHSIZE = SIZE // 4

        image = image[HHSIZE:SIZE - HHSIZE, HHSIZE:SIZE - HHSIZE, :]
        mask = mask[HHSIZE:SIZE - HHSIZE, HHSIZE:SIZE - HHSIZE, :]

        # ---- Spacing / reshape ----
        spacing = np.abs(img_nib.affine.diagonal()[:3]).astype(np.float32)
        nspacing = np.array([2.0, 2.0, 2.0], dtype=np.float32)
        rshape = (2 * np.array(image.shape) * spacing / nspacing) // 3
        rshape = rshape.astype(int)
        rspacing = np.array(image.shape) * spacing / rshape

        self.spacings[name] = rspacing

        image = reshape(image, image.shape, rshape, cv2.INTER_LINEAR)
        mask = reshape(mask, mask.shape, rshape, cv2.INTER_NEAREST)
        mask = (mask == 1).astype(np.uint8)

        # ---- Normalise HU ----
        image = io.split_images(image[:, :, :, None], self.channels)  # (X, Y, Z, C)
        image = np.moveaxis(image, -1, 0).astype(np.float32)  # (C, X, Y, Z)

        self.imgs[name] = image
        self.masks[name] = mask

        # ---- GT SDF ----
        sdf = compute_gt_sdf(mask)
        self.sdfs[name] = sdf[None]  # (1, X, Y, Z)
        self.sdf_weights[name] = compute_gt_sdf_surface_weighted_mask(sdf, bandwidth=5.0)[None]

        # ---- Load and parameterise knots ----
        def load_knots(path: Path) -> np.ndarray:
            with open(path) as f:
                pts = np.array(json.load(f)["knots"])
            # Axis permutation: stored as (z, x, y) → (x, y, z)
            pts = pts[:, [1, 2, 0]]
            pts[:, 0] = np.clip(pts[:, 0], HHSIZE, SIZE - HHSIZE - 1)
            pts[:, 1] = np.clip(pts[:, 1], HHSIZE, SIZE - HHSIZE - 1)
            pts -= np.array([HHSIZE, HHSIZE, 0])
            pts = pts * spacing / rspacing
            return pts.astype(np.float32)

        pts_l = load_knots(knotpath_l)
        pts_r = load_knots(knotpath_r)

        # Normalise to [-1, 1] (same as model output space)
        shape_arr = np.array(mask.shape, dtype=np.float32)

        # Отладка: проверяем диапазоны до нормализации
        print(
            f"  [load_knots] {name}: shape={mask.shape} pts_l range=[{pts_l.min():.1f}, {pts_l.max():.1f}] "
            f"per_axis={pts_l.min(axis=0).tolist()} .. {pts_l.max(axis=0).tolist()}"
        )

        pts_l_norm = 2.0 * (pts_l / (shape_arr - 1)) - 1.0
        pts_r_norm = 2.0 * (pts_r / (shape_arr - 1)) - 1.0

        # Canonical centering by bifurcation point.
        # Предполагаем, что левая и правая ветви сходятся в последней точке.
        bif_point = 0.5 * (pts_l_norm[-1] + pts_r_norm[-1])

        pts_l_norm = pts_l_norm - bif_point
        pts_r_norm = pts_r_norm - bif_point

        print(
            f"  [canonical] {name}: bif_point={bif_point.tolist()} "
            f"left_range={pts_l_norm.min(axis=0).tolist()}..{pts_l_norm.max(axis=0).tolist()} "
            f"right_range={pts_r_norm.min(axis=0).tolist()}..{pts_r_norm.max(axis=0).tolist()}"
        )
        t_l, pts_l_norm = arc_length_parameterise(pts_l_norm)
        t_r, pts_r_norm = arc_length_parameterise(pts_r_norm)

        self.t_left[name] = t_l
        self.pts_left[name] = pts_l_norm
        self.t_right[name] = t_r
        self.pts_right[name] = pts_r_norm

        self.shapes.append(mask.shape)

    def __len__(self) -> int:
        return len(self.keys)

    def _build_augmentation(self):
        """
        3D аугментации через voxelmentations.
        Применяются одновременно к изображению, маске и точкам кривых,
        поэтому геометрические трансформации корректно сдвигают GT.
        """
        return V.Sequential([
            # Интенсивностные (не меняют геометрию → точки не трогают)
            V.GaussNoise(variance=0.05, p=0.3),
            V.GaussBlur(p=0.25),
            V.IntensityShift(shift_limit=0.15, p=0.8),
            # Геометрические (меняют и изображение, и маску, и точки)
            # V.AxialPlaneAffine(
            #     angle_limit=75,
            #     scale_limit=0.25,
            #     shift_limit=0.25,
            #     p=1.0,
            #     fill_value=-1000,
            # ),
            # V.Flip(p=0.5),
        ])

    def __getitem__(self, idx):
        """
        Returns a dict with all tensors needed for one training step.
        The t values are stored as numpy arrays here; the collate_fn
        will convert them to tensors.

        В режиме train применяются 3D аугментации через voxelmentations.
        Аугментации применяются к (image, mask, points) одновременно,
        поэтому GT кривые корректно трансформируются вместе с объёмом.
        """
        key = self.keys[idx]

        image = self.imgs[key].copy()  # (C, X, Y, Z)
        mask = self.masks[key].copy()  # (X, Y, Z)
        pts_left = self.pts_left[key].copy()  # (N, 3) в [-1, 1]
        pts_right = self.pts_right[key].copy()  # (M, 3) в [-1, 1]

        if self.augment:
            aug = self._build_augmentation()

            # voxelmentations ожидает (X, Y, Z, C) для изображения
            image_hwzc = np.moveaxis(image, 0, -1).astype(np.float32)  # (X, Y, Z, C)

            # Точки в voxelmentations передаются в пространстве вокселей [0, shape-1].
            # Наши точки в [-1, 1] → конвертируем туда и обратно.
            shape_arr = np.array(mask.shape, dtype=np.float32)

            def norm_to_vox(pts_norm):
                return (pts_norm * 0.5 + 0.5) * (shape_arr - 1)

            def vox_to_norm(pts_vox):
                return 2.0 * (pts_vox / (shape_arr - 1)) - 1.0

            pts_l_vox = norm_to_vox(pts_left)
            pts_r_vox = norm_to_vox(pts_right)

            # Добавляем фиктивный 4-й столбец (флаг валидности), как в старом коде
            pts_l_vox4 = np.concatenate([pts_l_vox, np.ones((len(pts_l_vox), 1))], axis=-1)
            pts_r_vox4 = np.concatenate([pts_r_vox, np.ones((len(pts_r_vox), 1))], axis=-1)
            all_pts = np.vstack([pts_l_vox4, pts_r_vox4])
            n_left = len(pts_l_vox4)

            try:
                auged = aug(voxel=image_hwzc, mask=mask, points=all_pts)
                image_hwzc = auged["voxel"]
                mask = auged["mask"]
                all_pts_aug = auged["points"]

                pts_l_vox_aug = all_pts_aug[:n_left, :3]
                pts_r_vox_aug = all_pts_aug[n_left:, :3]

                pts_left = vox_to_norm(pts_l_vox_aug)
                pts_right = vox_to_norm(pts_r_vox_aug)
            except Exception as e:
                # Если аугментация упала (редкий edge case) — используем оригинал
                print(f"  [aug] предупреждение: {e}")

            image = np.moveaxis(image_hwzc, -1, 0).astype(np.float32)  # (C, X, Y, Z)

            # Пересчитываем arc-length параметризацию для аугментированных точек
            t_left, pts_left = arc_length_parameterise(pts_left)
            t_right, pts_right = arc_length_parameterise(pts_right)

            # Пересчитываем SDF для аугментированной маски
            from losses_implicit import compute_gt_sdf, compute_gt_sdf_surface_weighted_mask
            sdf = compute_gt_sdf(mask)
            sdf_vol = sdf[None]
            sdf_weight = compute_gt_sdf_surface_weighted_mask(sdf, bandwidth=5.0)[None]
        else:
            t_left = self.t_left[key]
            t_right = self.t_right[key]
            sdf_vol = self.sdfs[key]
            sdf_weight = self.sdf_weights[key]

        return {
            "key": key,
            "image": image,  # (C, X, Y, Z)
            "mask": mask,  # (X, Y, Z)
            "sdf": sdf_vol,  # (1, X, Y, Z)
            "sdf_weight": sdf_weight,  # (1, X, Y, Z)
            "t_left": t_left,
            "pts_left": pts_left,
            "t_right": t_right,
            "pts_right": pts_right,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Stack tensors with per-batch zero-padding to the maximal spatial shape,
    keep variable-length GT curves as lists.
    """
    out = {}
    out["key"] = [b["key"] for b in batch]

    image_shapes = [tuple(b["image"].shape) for b in batch]
    mask_shapes = [tuple(b["mask"].shape) for b in batch]
    sdf_shapes = [tuple(b["sdf"].shape) for b in batch]

    max_c = max(shape[0] for shape in image_shapes)
    max_x = max(shape[1] for shape in image_shapes)
    max_y = max(shape[2] for shape in image_shapes)
    max_z = max(shape[3] for shape in image_shapes)

    padded_images = []
    padded_masks = []
    padded_sdfs = []
    padded_sdf_weights = []

    for b in batch:
        img = b["image"]
        mask = b["mask"]
        sdf = b["sdf"]
        sdf_weight = b["sdf_weight"]

        img_pad = np.zeros((max_c, max_x, max_y, max_z), dtype=img.dtype)
        mask_pad = np.zeros((max_x, max_y, max_z), dtype=mask.dtype)
        sdf_pad = np.zeros((1, max_x, max_y, max_z), dtype=sdf.dtype)
        sdf_weight_pad = np.zeros((1, max_x, max_y, max_z), dtype=sdf_weight.dtype)

        c, x, y, z = img.shape
        img_pad[:c, :x, :y, :z] = img
        mask_pad[:x, :y, :z] = mask
        sdf_pad[:, :x, :y, :z] = sdf
        sdf_weight_pad[:, :x, :y, :z] = sdf_weight

        padded_images.append(img_pad)
        padded_masks.append(mask_pad)
        padded_sdfs.append(sdf_pad)
        padded_sdf_weights.append(sdf_weight_pad)

    out["image"] = torch.tensor(np.stack(padded_images, axis=0), dtype=torch.float32)
    out["mask"] = torch.tensor(np.stack(padded_masks, axis=0), dtype=torch.uint8)
    out["sdf"] = torch.tensor(np.stack(padded_sdfs, axis=0), dtype=torch.float32)
    out["sdf_weight"] = torch.tensor(np.stack(padded_sdf_weights, axis=0), dtype=torch.float32)
    out["t_left"] = [b["t_left"] for b in batch]
    out["pts_left"] = [b["pts_left"] for b in batch]
    out["t_right"] = [b["t_right"] for b in batch]
    out["pts_right"] = [b["pts_right"] for b in batch]
    out["orig_image_shapes"] = image_shapes
    out["orig_mask_shapes"] = mask_shapes
    out["orig_sdf_shapes"] = sdf_shapes
    return out


# ---------------------------------------------------------------------------
# Curve GT sampling
# ---------------------------------------------------------------------------

def sample_gt_at_random_t(
    t_gt_list: list[np.ndarray],
    pts_gt_list: list[np.ndarray],
    n_query: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each sample in the batch, sample n_query random t values uniformly
    in [0, 1] and interpolate the GT curve at those positions.

    Returns:
        t_batch:   (B, n_query)  float32 tensor
        pts_batch: (B, n_query, 3) float32 tensor
    """
    B = len(t_gt_list)
    t_batch = np.zeros((B, n_query), dtype=np.float32)
    pts_batch = np.zeros((B, n_query, 3), dtype=np.float32)

    for b in range(B):
        t_q = np.sort(np.random.uniform(0.0, 1.0, n_query).astype(np.float32))
        pts_q = interpolate_curve_at_t(t_gt_list[b], pts_gt_list[b], t_q)
        t_batch[b] = t_q
        pts_batch[b] = pts_q

    t_tensor = torch.tensor(t_batch, dtype=torch.float32, device=device)
    pts_tensor = torch.tensor(pts_batch, dtype=torch.float32, device=device)
    return t_tensor, pts_tensor


# ---------------------------------------------------------------------------
# Validation / train-set metrics
# ---------------------------------------------------------------------------

def compute_val_metrics(
    model: ImplicitCurveNet,
    dataset: ImplicitCurveDataset,
    device: torch.device,
    n_eval_points: int = 256,
) -> dict[str, float]:
    """
    Полная валидация на всём датасете (по одному пациенту).

    Метрики (из visualize_implicit.compute_metrics):
      - msd_left / right          : Mean Symmetric Distance (непрерывная)
      - hausdorff_left / right    : Расстояние Хаусдорфа
      - endpoint_error_left/right : Ошибка концов кривой (t=0 и t=1)
      - speed_variance_left/right : Дисперсия скорости (равномерность параметризации)
      - curvature_left / right    : Средняя кривизна
      - msd_mean / hausdorff_mean / endpoint_error_mean : агрегированные
    """
    model.eval()
    all_metrics = defaultdict(list)

    with torch.no_grad():
        for key in dataset.keys:
            img = torch.tensor(dataset.imgs[key], dtype=torch.float32, device=device).unsqueeze(0)

            pts_left = model.predict_curve(img, n_points=n_eval_points, branch_id=0)[0].cpu().numpy()
            pts_right = model.predict_curve(img, n_points=n_eval_points, branch_id=1)[0].cpu().numpy()

            # Отладка: проверяем предсказания и GT
            if key == dataset.keys[0]:
                print(f"  [DEBUG] {key}: pts_left range=[{pts_left.min():.3f}, {pts_left.max():.3f}] shape={pts_left.shape}")
                print(f"  [DEBUG] {key}: pts_right range=[{pts_right.min():.3f}, {pts_right.max():.3f}] shape={pts_right.shape}")
                print(f"  [DEBUG] {key}: t_gt_left range=[{dataset.t_left[key].min():.3f}, {dataset.t_left[key].max():.3f}] len={len(dataset.t_left[key])}")
                print(f"  [DEBUG] {key}: pts_gt_left range=[{dataset.pts_left[key].min():.3f}, {dataset.pts_left[key].max():.3f}]")

            m = compute_metrics(
                pred_pts_left=pts_left,
                pred_pts_right=pts_right,
                t_gt_left=dataset.t_left[key],
                pts_gt_left=dataset.pts_left[key],
                t_gt_right=dataset.t_right[key],
                pts_gt_right=dataset.pts_right[key],
                n_dense=n_eval_points,
            )

            for k, v in m.items():
                all_metrics[k].append(v)

    return {k: float(np.mean(v)) for k, v in all_metrics.items()}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: ImplicitCurveNet,
    averaged_model: tsu.AveragedModel,
    loss_fn: ImplicitCurveLoss,
    opt: torch.optim.Optimizer,
    dataloader: DataLoader,
    device: torch.device,
    n_query: int,
    accum_steps: int = 1,
    target_branch: str = "both",
) -> dict[str, float]:
    model.train()
    running = defaultdict(float)
    n_steps = 0

    opt.zero_grad(set_to_none=True)

    import time as _time
    _t_data, _t_enc, _t_loss, _t_back = 0.0, 0.0, 0.0, 0.0

    for step_idx, batch in enumerate(dataloader):
        _t0 = _time.perf_counter()
        images = batch["image"].to(device)

        # Sample random t values and interpolate GT for this batch
        t_left, gt_left = sample_gt_at_random_t(batch["t_left"], batch["pts_left"], n_query, device)
        t_right, gt_right = sample_gt_at_random_t(batch["t_right"], batch["pts_right"], n_query, device)
        _t_data += _time.perf_counter() - _t0

        # Single encoder pass — cache bottleneck for MLP queries
        _t1 = _time.perf_counter()
        decoder_out, bottleneck = model._encode_decode(images)

        # SDF head — только если нужен лосс
        if loss_fn.w_sdf > 0.0:
            pred_sdf = model.sdf_head(decoder_out)
            gt_sdf = batch["sdf"].to(device)
            sdf_weight = batch["sdf_weight"].to(device)
        else:
            pred_sdf = decoder_out.new_zeros(1)  # заглушка, не используется
            gt_sdf = decoder_out.new_zeros(1)
            sdf_weight = None
        _t_enc += _time.perf_counter() - _t1

        def make_model_fn(branch_id: int):
            def _fn(t: torch.Tensor) -> torch.Tensor:
                return model._query_curve(bottleneck, decoder_out, t, branch_id=branch_id)
            return _fn

        model_fn_left = make_model_fn(0)
        model_fn_right = make_model_fn(1)

        _t2 = _time.perf_counter()
        losses = loss_fn(
            model_fn_left,
            model_fn_right,
            t_left,
            t_right,
            gt_left,
            gt_right,
            pred_sdf,
            gt_sdf,
            sdf_weight_mask=sdf_weight,
            active_branch=target_branch,
        )

        _t_loss += _time.perf_counter() - _t2

        _t3 = _time.perf_counter()
        loss = losses["total"] / accum_steps
        loss.backward()
        _t_back += _time.perf_counter() - _t3

        # Отладка — первый шаг каждой эпохи
        if step_idx == 0:
            with torch.no_grad():
                pred_sample = model_fn_left(t_left)  # (B, N, 3)
                gt_sample = gt_left  # (B, N, 3)
                print(
                    f"  [TRAIN DBG] pred range=[{pred_sample.min():.3f}, {pred_sample.max():.3f}]  "
                    f"gt range=[{gt_sample.min():.3f}, {gt_sample.max():.3f}]  "
                    f"mse={((pred_sample - gt_sample) ** 2).mean():.4f}"
                )
            coarse_out_grad = model.coarse_mlp.out.weight.grad
            refine_out_grad = model.refine_mlp.out.weight.grad
            if coarse_out_grad is not None or refine_out_grad is not None:
                coarse_norm = coarse_out_grad.norm().item() if coarse_out_grad is not None else 0.0
                refine_norm = refine_out_grad.norm().item() if refine_out_grad is not None else 0.0
                print(
                    f"  [GRAD] coarse_out={coarse_norm:.4f} | refine_out={refine_norm:.4f} | "
                    f"coarse_delta={model.last_coarse_delta_mean:.4f}/{model.last_coarse_delta_max:.4f} | "
                    f"refine_delta={model.last_delta_mean:.4f}/{model.last_delta_max:.4f}"
                )

        if (step_idx + 1) % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            averaged_model.update_parameters(model)
            opt.zero_grad(set_to_none=True)

        for k, v in losses.items():
            running[k] += v.item()
        running["coarse_delta_mean"] += float(getattr(model, "last_coarse_delta_mean", 0.0))
        running["coarse_delta_max"] += float(getattr(model, "last_coarse_delta_max", 0.0))
        running["delta_mean"] += float(getattr(model, "last_delta_mean", 0.0))
        running["delta_max"] += float(getattr(model, "last_delta_max", 0.0))
        n_steps += 1

    print(f"  [timing] data={_t_data:.2f}s  enc={_t_enc:.2f}s  loss={_t_loss:.2f}s  back={_t_back:.2f}s  steps={n_steps}")
    return {k: v / max(n_steps, 1) for k, v in running.items()}


# ---------------------------------------------------------------------------
# Main fit function
# ---------------------------------------------------------------------------

def _save_history_plots(history: list[dict], out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]
    if not epochs:
        return

    def _plot(keys, title, fname):
        plt.figure(figsize=(9, 5))
        for key in keys:
            vals = [row.get(key, float("nan")) for row in history]
            plt.plot(epochs, vals, label=key)
        plt.xlabel("epoch")
        plt.ylabel("value")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / fname, dpi=140)
        plt.close()

    _plot(["train_total", "train_curve_left", "train_curve_right", "train_sdf"], "Train losses", "train_losses.png")
    _plot(["train_msd_mean", "train_hausdorff_mean", "train_endpoint_error_mean"], "Train main metrics", "train_main_metrics.png")
    _plot(["val_msd_mean", "val_hausdorff_mean", "val_endpoint_error_mean"], "Validation main metrics", "val_main_metrics.png")
    _plot(["val_msd_xy_mean", "val_msd_xz_mean", "val_msd_yz_mean"], "Validation MSD by plane", "val_plane_msd.png")
    _plot(["val_speed_variance_left", "val_speed_variance_right"], "Validation speed variance", "val_speed_variance.png")



def fit(model: ImplicitCurveNet, data: dict):
    device = config["DEVICE"]
    model.to(device)

    # EMA averaged model — decay=0.9 (быстрее адаптируется при малом датасете)
    ema_fn = lambda ema, cur, n: 0.9 * ema + 0.1 * cur
    averaged_model = tsu.AveragedModel(model, avg_fn=ema_fn)
    averaged_model.to(device)

    loss_fn = ImplicitCurveLoss(
        w_curve=config["W_CURVE"],
        w_length=config["W_LENGTH"],
        w_smooth=config["W_SMOOTH"],
        w_sdf=config["W_SDF"],
        w_plane=config["W_PLANE"],
        w_inside=config.get("W_INSIDE", 0.0),
        w_speed=config.get("W_SPEED", 0.0),
    )

    opt = torch.optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], eps=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=config.get("LR_SCHED_FACTOR", 0.5),
        patience=config.get("LR_SCHED_PATIENCE", 10),
        min_lr=config.get("LR_SCHED_MIN_LR", 1e-6),
    )

    train_loader = DataLoader(
        data["train"],
        batch_size=config["BATCH_SIZE"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False,
    )

    best_score = float("inf")
    best_state = copy.deepcopy(averaged_model.state_dict())
    epochs_no_improve = 0

    vis_dir = Path(config.get("VIS_DIR", "vis"))
    history_dir = vis_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []

    for epoch in range(config["EPOCHS"]):
        t0 = time.perf_counter()

        # ---- Train ----
        train_losses = train_one_epoch(
            model, averaged_model, loss_fn, opt, train_loader, device,
            n_query=config["NPOINTS_QUERY"],
            accum_steps=config["ACCUMULATION_STEPS"],
            target_branch=config.get("TARGET_BRANCH", "both"),
        )

        train_metrics = compute_val_metrics(
            averaged_model.module, data["train"], device,
            n_eval_points=256,
        )

        # ---- Validate ----
        val_metrics = compute_val_metrics(
            averaged_model.module, data["val"], device,
            n_eval_points=256,
        )

        # Primary metric: MSD (Mean Symmetric Distance), среднее по обеим ветвям
        score = val_metrics.get("msd_mean", 1e9)

        scheduler.step(score)
        current_lr = opt.param_groups[0]["lr"]

        elapsed = time.perf_counter() - t0

        # ---- Консольный лог ----
        print(
            f"\nEpoch {epoch + 1:03d}/{config['EPOCHS']} | "
            f"branch={config.get('TARGET_BRANCH', 'both')} | "
            f"time={elapsed:.1f}s | lr={current_lr:.2e}"
        )
        print(
            f"  LOSS  total={train_losses['total']:.4f} | "
            f"curve={train_losses['curve_left']:.4f}/{train_losses['curve_right']:.4f} | "
            f"plane={train_losses['plane_left']:.4f}/{train_losses['plane_right']:.4f} | "
            f"len={train_losses['length_left']:.4f}/{train_losses['length_right']:.4f} | "
            f"speed={train_losses['speed_left']:.4f}/{train_losses['speed_right']:.4f} | "
            f"smooth={train_losses['smooth_left']:.8f}/{train_losses['smooth_right']:.8f} | "
            f"inside={train_losses.get('inside_left', 0.0):.4f}/{train_losses.get('inside_right', 0.0):.4f} | "
            f"sdf={train_losses['sdf']:.4f} | "
            f"coarse_delta={train_losses.get('coarse_delta_mean', 0.0):.4f}/{train_losses.get('coarse_delta_max', 0.0):.4f} | "
            f"refine_delta={train_losses.get('delta_mean', 0.0):.4f}/{train_losses.get('delta_max', 0.0):.4f}"
        )
        print(
            f"  TRAIN MSD={train_metrics.get('msd_mean', 0):.4f} | "
            f"HD={train_metrics.get('hausdorff_mean', 0):.4f} | "
            f"EP_err={train_metrics.get('endpoint_error_mean', 0):.4f} | "
            f"speed_var_L={train_metrics.get('speed_variance_left', 0):.4f} "
            f"speed_var_R={train_metrics.get('speed_variance_right', 0):.4f}"
        )
        print(
            f"        planes: MSD_XY={train_metrics.get('msd_xy_mean', 0):.4f} | "
            f"MSD_XZ={train_metrics.get('msd_xz_mean', 0):.4f} | "
            f"MSD_YZ={train_metrics.get('msd_yz_mean', 0):.4f}"
        )
        print(
            f"  VAL   MSD={val_metrics.get('msd_mean', 0):.4f} | "
            f"HD={val_metrics.get('hausdorff_mean', 0):.4f} | "
            f"EP_err={val_metrics.get('endpoint_error_mean', 0):.4f} | "
            f"speed_var_L={val_metrics.get('speed_variance_left', 0):.4f} "
            f"speed_var_R={val_metrics.get('speed_variance_right', 0):.4f}"
        )
        print(
            f"        planes: MSD_XY={val_metrics.get('msd_xy_mean', 0):.4f} | "
            f"MSD_XZ={val_metrics.get('msd_xz_mean', 0):.4f} | "
            f"MSD_YZ={val_metrics.get('msd_yz_mean', 0):.4f}"
        )

        # ---- Визуализация ----
        vis_every = int(config.get("VIS_EVERY", 1))
        should_visualize = vis_every > 0 and ((epoch + 1) % vis_every == 0 or epoch == 0)
        if should_visualize:
            visualize_epoch(
                model=averaged_model.module,
                dataset=data["train"],
                device=device,
                epoch=epoch + 1,
                vis_dir=vis_dir,
                n_samples=config.get("VIS_SAMPLES", 2),
                n_points=256,
                split_name="train",
                make_html=False,
            )
            visualize_epoch(
                model=averaged_model.module,
                dataset=data["val"],
                device=device,
                epoch=epoch + 1,
                vis_dir=vis_dir,
                n_samples=config.get("VIS_SAMPLES", 2),
                n_points=256,
                split_name="val",
                make_html=True,
            )

        # ---- Сохранение истории эпох ----
        row = {"epoch": epoch + 1, "lr": float(current_lr)}
        row.update({f"train_{k}": float(v) for k, v in train_losses.items()})
        row.update({f"train_{k}": float(v) for k, v in train_metrics.items()})
        row.update({f"val_{k}": float(v) for k, v in val_metrics.items()})
        history.append(row)

        json_path = history_dir / "history.json"
        csv_path = history_dir / "history.csv"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

        fieldnames = sorted({k for r in history for k in r.keys()}, key=lambda x: (x != "epoch", x))
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in history:
                writer.writerow(r)

        _save_history_plots(history, history_dir)

        # ---- ClearML logging ----
        if config.get("TASK") is not None:
            logger = config["TASK"].get_logger()
            for k, v in train_losses.items():
                logger.report_scalar("train_loss", k, iteration=epoch, value=v)
            for k, v in train_metrics.items():
                logger.report_scalar("train_metrics", k, iteration=epoch, value=v)
            for k, v in val_metrics.items():
                logger.report_scalar("val_metrics", k, iteration=epoch, value=v)
            logger.report_scalar("val_metrics", "score (MSD mean)", iteration=epoch, value=score)
            # Загружаем PNG в ClearML только в эпохи визуализации
            if should_visualize:
                for split_name in ("train", "val"):
                    ds = data[split_name]
                    for key in ds.keys[:config.get("VIS_SAMPLES", 2)]:
                        png_path = vis_dir / split_name / "png" / f"epoch_{epoch + 1:03d}_{key}.png"
                        if png_path.exists() and config.get("TASK") is not None:
                            config["TASK"].get_logger().report_image(
                                title=f"{split_name} curves",
                                series=key,
                                iteration=epoch,
                                local_path=str(png_path),
                            )

        # ---- Checkpoint ----
        if score < best_score:
            best_score = score
            best_state = copy.deepcopy(averaged_model.state_dict())
            epochs_no_improve = 0
            torch.save(averaged_model.module.state_dict(), config["MODELNAME"])
            print(f"  ✓ New best model saved (score={best_score:.4f})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= config["STOP_EPOCHS"]:
            print(f"Early stopping after {epoch + 1} epochs.")
            break

    # Restore best
    averaged_model.load_state_dict(best_state)
    model.load_state_dict(averaged_model.module.state_dict())


# ---------------------------------------------------------------------------
# Entry point — читает всё из config_implicit.yaml
# ---------------------------------------------------------------------------

def main():
    # ---- Путь к конфигу ----
    cfg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config_implicit.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Конфиг не найден: {cfg_path}")

    cfg = load_yaml_config(cfg_path)
    print(f"[config] загружен из {cfg_path}")

    # ---- Заполняем глобальный config ----
    config["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr = cfg["training"]
    config["EPOCHS"] = tr["epochs"]
    config["STOP_EPOCHS"] = tr["stop_epochs"]
    config["BATCH_SIZE"] = tr["batch_size"]
    config["ACCUMULATION_STEPS"] = tr["accumulation_steps"]
    config["LEARNING_RATE"] = tr["learning_rate"]
    config["NPOINTS_QUERY"] = tr["npoints_query"]
    config["MODELNAME"] = tr["modelname"]
    config["LR_SCHED_FACTOR"] = tr.get("lr_sched_factor", 0.5)
    config["LR_SCHED_PATIENCE"] = tr.get("lr_sched_patience", 10)
    config["LR_SCHED_MIN_LR"] = tr.get("lr_sched_min_lr", 1e-6)
    config["TARGET_BRANCH"] = tr.get("target_branch", "both")
    if config["TARGET_BRANCH"] not in ("both", "left", "right"):
        raise ValueError(
            "training.target_branch must be one of: 'both', 'left', 'right', "
            f"got {config['TARGET_BRANCH']!r}"
        )

    ls = cfg["losses"]
    config["W_CURVE"] = ls["w_curve"]
    config["W_PLANE"] = ls.get("w_plane", 0.0)
    config["W_SPEED"] = ls.get("w_speed", 0.0)
    config["W_LENGTH"] = ls["w_length"]
    config["W_SMOOTH"] = ls["w_smooth"]
    config["W_SDF"] = ls["w_sdf"]
    config["W_INSIDE"] = ls.get("w_inside", 0.0)

    vis = cfg.get("visualization", {})
    config["VIS_DIR"] = vis.get("vis_dir", "vis")
    config["VIS_SAMPLES"] = vis.get("vis_samples", 2)
    config["VIS_EVERY"] = int(vis.get("vis_every", 1))

    config["TASK"] = None

    # ---- ClearML ----
    cm = cfg.get("clearml", {})
    if cm.get("enabled", False):
        from clearml import Task
        Task.set_credentials(
            api_host=cm["api_host"],
            web_host=cm["web_host"],
            files_host=cm["files_host"],
            key=cm["key"],
            secret=cm["secret"],
        )
        task = Task.init(
            project_name=cm["project_name"],
            task_name=cm["task_name"],
        )
        config["TASK"] = task
        print(f"[clearml] задача '{cm['task_name']}' в проекте '{cm['project_name']}' запущена")

    # ---- Данные ----
    split = load_yaml_config(Path(cfg["split_options_path"]))
    train_names = [str(n) for n in split["train"]]
    val_names = [str(n) for n in split["val"]]

    channels = {"all": {"MIN_HU": -200, "MAX_HU": 1200}}
    datapath = Path(cfg["datapath"])

    print("Загрузка train датасета...")
    train_ds = ImplicitCurveDataset(datapath, train_names, channels, mode="train")

    print("Загрузка val датасета...")
    val_ds = ImplicitCurveDataset(datapath, val_names, channels, mode="val")

    data = {"train": train_ds, "val": val_ds}

    # ---- Модель ----
    mc = cfg["model"]
    model = ImplicitCurveNet(
        in_channels=mc["in_channels"],
        depth=mc["depth"],
        pe_num_freqs=mc["pe_num_freqs"],
        context_channels=mc["context_channels"],
        mlp_hidden_dim=mc["mlp_hidden_dim"],
        mlp_num_layers=mc["mlp_num_layers"],
        branch_emb_dim=mc.get("branch_emb_dim", 8),
        local_feat_channels=mc.get("local_feat_channels", 32),
        num_refine_passes=mc.get("num_refine_passes", 1),
        refine_scale=mc.get("refine_scale", 0.25),
        coarse_scale=mc.get("coarse_scale", 1.0),
        base_xy_slope=mc.get("base_xy_slope", 0.12),
        base_branch_y_slope=mc.get("base_branch_y_slope", 0.06),
        affine_max_shift=mc.get("affine_max_shift", 0.6),
        affine_max_log_scale=mc.get("affine_max_log_scale", 0.35),
        context_pool_size=mc.get("context_pool_size", 1),
        log_shapes=mc.get("log_shapes", False),
    )

    print(f"Параметров модели: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Устройство: {config['DEVICE']}")

    # ---- Обучение ----
    fit(model, data)

    print("Готово.")


if __name__ == "__main__":
    main()
