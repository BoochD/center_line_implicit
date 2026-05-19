from __future__ import annotations

"""
Визуализация и метрики для ImplicitCurveNet.

Сохраняет PNG-картинки с предсказанными и GT кривыми в папку vis/.
Также считает метрики качества и при наличии plotly может сохранять
интерактивные HTML-сцены с 3D-маской и кривыми.
"""

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # без GUI
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pathlib import Path
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from skimage import measure


# ---------------------------------------------------------------------------
# Метрики
# ---------------------------------------------------------------------------

def _sanitize_curve_points(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] not in (2, 3):
        return np.zeros((2, 3), dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return np.zeros((2, pts.shape[1]), dtype=np.float32)
    if len(pts) == 1:
        return np.repeat(pts, 2, axis=0)
    return pts


def _dense_sample_curve(t_gt, pts_gt, n=512):
    """Плотно сэмплировать GT кривую устойчиво к дубликатам и NaN."""
    t_gt = np.asarray(t_gt, dtype=np.float32)
    pts_gt = np.asarray(pts_gt, dtype=np.float32)
    finite_mask = np.isfinite(t_gt) & np.isfinite(pts_gt).all(axis=1)
    t_gt = t_gt[finite_mask]
    pts_gt = pts_gt[finite_mask]

    if len(t_gt) < 2:
        base = pts_gt[0] if len(pts_gt) > 0 else np.zeros(3, dtype=np.float32)
        return np.repeat(base[None], n, axis=0)

    keep = np.ones(len(t_gt), dtype=bool)
    keep[1:] = np.diff(t_gt) > 1e-8
    t_gt = t_gt[keep]
    pts_gt = pts_gt[keep]

    if len(t_gt) < 2:
        base = pts_gt[0]
        return np.repeat(base[None], n, axis=0)

    t_dense = np.linspace(0.0, 1.0, n, dtype=np.float32)
    interp = interp1d(
        t_gt,
        pts_gt,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value=(pts_gt[0], pts_gt[-1]),
        assume_sorted=True,
    )
    out = interp(t_dense).astype(np.float32)
    out[~np.isfinite(out)] = 0.0
    return out


def mean_symmetric_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    pred_pts = _sanitize_curve_points(pred_pts)
    gt_pts = _sanitize_curve_points(gt_pts)
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred, _ = tree_gt.query(pred_pts)
    d_gt, _ = tree_pred.query(gt_pts)
    val = 0.5 * (np.mean(d_pred) + np.mean(d_gt))
    return float(val) if np.isfinite(val) else 1e6


def hausdorff_distance(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    pred_pts = _sanitize_curve_points(pred_pts)
    gt_pts = _sanitize_curve_points(gt_pts)
    tree_gt = cKDTree(gt_pts)
    tree_pred = cKDTree(pred_pts)
    d_pred, _ = tree_gt.query(pred_pts)
    d_gt, _ = tree_pred.query(gt_pts)
    val = max(np.max(d_pred), np.max(d_gt))
    return float(val) if np.isfinite(val) else 1e6


def endpoint_error(pred_pts: np.ndarray, gt_pts: np.ndarray) -> float:
    pred_pts = _sanitize_curve_points(pred_pts)
    gt_pts = _sanitize_curve_points(gt_pts)
    p_start, p_end = pred_pts[0], pred_pts[-1]
    g_start, g_end = gt_pts[0], gt_pts[-1]
    d_same = np.linalg.norm(p_start - g_start) + np.linalg.norm(p_end - g_end)
    d_flip = np.linalg.norm(p_start - g_end) + np.linalg.norm(p_end - g_start)
    val = min(d_same, d_flip)
    return float(val) if np.isfinite(val) else 1e6


def speed_variance(pred_pts: np.ndarray) -> float:
    pred_pts = _sanitize_curve_points(pred_pts)
    diffs = np.diff(pred_pts, axis=0)
    speeds = np.linalg.norm(diffs, axis=1)
    mean_speed = np.mean(speeds)
    if mean_speed < 1e-8 or not np.isfinite(mean_speed):
        return 1e3
    val = np.std(speeds) / mean_speed
    return float(val) if np.isfinite(val) else 1e3


def mean_curvature(pred_pts: np.ndarray) -> float:
    if len(pred_pts) < 3:
        return 0.0
    d1 = np.diff(pred_pts, axis=0)
    d2 = np.diff(d1, axis=0)
    return float(np.mean(np.linalg.norm(d2, axis=1)))


def compute_metrics(
    pred_pts_left: np.ndarray,
    pred_pts_right: np.ndarray,
    t_gt_left: np.ndarray,
    pts_gt_left: np.ndarray,
    t_gt_right: np.ndarray,
    pts_gt_right: np.ndarray,
    n_dense: int = 512,
) -> dict[str, float]:
    gt_left_dense = _dense_sample_curve(t_gt_left, pts_gt_left, n_dense)
    gt_right_dense = _dense_sample_curve(t_gt_right, pts_gt_right, n_dense)

    metrics = {}
    plane_axes = {
        "xy": [0, 1],
        "xz": [0, 2],
        "yz": [1, 2],
    }

    for side, pred, gt_dense in [
        ("left", pred_pts_left, gt_left_dense),
        ("right", pred_pts_right, gt_right_dense),
    ]:
        metrics[f"msd_{side}"] = mean_symmetric_distance(pred, gt_dense)
        metrics[f"hausdorff_{side}"] = hausdorff_distance(pred, gt_dense)
        metrics[f"endpoint_error_{side}"] = endpoint_error(pred, gt_dense)
        metrics[f"speed_variance_{side}"] = speed_variance(pred)
        metrics[f"curvature_{side}"] = mean_curvature(pred)

        for plane, axes in plane_axes.items():
            pred_2d = pred[:, axes]
            gt_2d = gt_dense[:, axes]
            metrics[f"msd_{plane}_{side}"] = mean_symmetric_distance(pred_2d, gt_2d)
            metrics[f"hausdorff_{plane}_{side}"] = hausdorff_distance(pred_2d, gt_2d)

    metrics["msd_mean"] = (metrics["msd_left"] + metrics["msd_right"]) / 2
    metrics["hausdorff_mean"] = (metrics["hausdorff_left"] + metrics["hausdorff_right"]) / 2
    metrics["endpoint_error_mean"] = (metrics["endpoint_error_left"] + metrics["endpoint_error_right"]) / 2

    for plane in plane_axes:
        metrics[f"msd_{plane}_mean"] = (metrics[f"msd_{plane}_left"] + metrics[f"msd_{plane}_right"]) / 2
        metrics[f"hausdorff_{plane}_mean"] = (metrics[f"hausdorff_{plane}_left"] + metrics[f"hausdorff_{plane}_right"]) / 2

    return metrics


# ---------------------------------------------------------------------------
# Координатные преобразования
# ---------------------------------------------------------------------------

def _norm_to_voxel_coords(points_norm: np.ndarray, volume_shape: tuple[int, int, int]) -> np.ndarray:
    shape_arr = np.asarray(volume_shape, dtype=np.float32)
    pts = np.asarray(points_norm, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0, 3)
    return (pts * 0.5 + 0.5) * (shape_arr - 1.0)


def _mask_mesh_from_binary(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    mask = np.asarray(mask)
    if mask.ndim != 3 or mask.max() <= 0:
        return None, None

    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    verts, faces, _, _ = measure.marching_cubes(padded, level=0.5)
    verts -= 1.0
    return verts.astype(np.float32), faces.astype(np.int32)


# ---------------------------------------------------------------------------
# Визуализация — PNG
# ---------------------------------------------------------------------------

def save_curve_plots(
    pred_pts_left: np.ndarray,
    pred_pts_right: np.ndarray,
    t_gt_left: np.ndarray,
    pts_gt_left: np.ndarray,
    t_gt_right: np.ndarray,
    pts_gt_right: np.ndarray,
    save_path: Path,
    sample_name: str = "",
    epoch: int = 0,
    n_dense: int = 256,
    mask_slice: np.ndarray | None = None,
    title_prefix: str = "",
):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gt_left_dense = _dense_sample_curve(t_gt_left, pts_gt_left, n_dense)
    gt_right_dense = _dense_sample_curve(t_gt_right, pts_gt_right, n_dense)

    fig = plt.figure(figsize=(18, 14))
    prefix = f"{title_prefix} | " if title_prefix else ""
    fig.suptitle(
        f"{prefix}Epoch {epoch:03d} | {sample_name}\n"
        f"Синий=pred_left, Красный=pred_right | "
        f"Голубой=gt_left, Розовый=gt_right",
        fontsize=11,
    )

    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.3)
    views = [
        ("XY (аксиальный)", 0, 1),
        ("XZ (сагиттальный)", 0, 2),
        ("YZ (корональный)", 1, 2),
    ]

    for col, (title, ax_h, ax_v) in enumerate(views):
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(title, fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        ax.plot(
            gt_left_dense[:, ax_h], gt_left_dense[:, ax_v],
            color="deepskyblue", lw=1.5, alpha=0.8, ls="--", label="GT left",
        )
        ax.plot(
            gt_right_dense[:, ax_h], gt_right_dense[:, ax_v],
            color="lightcoral", lw=1.5, alpha=0.8, ls="--", label="GT right",
        )

        ax.plot(
            pred_pts_left[:, ax_h], pred_pts_left[:, ax_v],
            color="blue", lw=2.0, label="Pred left",
        )
        ax.plot(
            pred_pts_right[:, ax_h], pred_pts_right[:, ax_v],
            color="red", lw=2.0, label="Pred right",
        )

        ax.scatter(pred_pts_left[0, ax_h], pred_pts_left[0, ax_v], color="blue", s=60, zorder=5, marker="o")
        ax.scatter(pred_pts_left[-1, ax_h], pred_pts_left[-1, ax_v], color="blue", s=60, zorder=5, marker="s")
        ax.scatter(pred_pts_right[0, ax_h], pred_pts_right[0, ax_v], color="red", s=60, zorder=5, marker="o")
        ax.scatter(pred_pts_right[-1, ax_h], pred_pts_right[-1, ax_v], color="red", s=60, zorder=5, marker="s")

        if col == 0:
            ax.legend(fontsize=7, loc="upper right")

    ax3d = fig.add_subplot(gs[1, :2], projection="3d")
    ax3d.set_title("3D вид", fontsize=9)
    ax3d.plot(
        gt_left_dense[:, 0], gt_left_dense[:, 1], gt_left_dense[:, 2],
        color="deepskyblue", lw=1.5, alpha=0.8, ls="--", label="GT left",
    )
    ax3d.plot(
        gt_right_dense[:, 0], gt_right_dense[:, 1], gt_right_dense[:, 2],
        color="lightcoral", lw=1.5, alpha=0.8, ls="--", label="GT right",
    )
    ax3d.plot(
        pred_pts_left[:, 0], pred_pts_left[:, 1], pred_pts_left[:, 2],
        color="blue", lw=2.0, label="Pred left",
    )
    ax3d.plot(
        pred_pts_right[:, 0], pred_pts_right[:, 1], pred_pts_right[:, 2],
        color="red", lw=2.0, label="Pred right",
    )
    ax3d.set_xlabel("X", fontsize=7)
    ax3d.set_ylabel("Y", fontsize=7)
    ax3d.set_zlabel("Z", fontsize=7)
    ax3d.legend(fontsize=7)

    metrics = compute_metrics(
        pred_pts_left, pred_pts_right,
        t_gt_left, pts_gt_left,
        t_gt_right, pts_gt_right,
        n_dense=n_dense,
    )

    ax_tbl = fig.add_subplot(gs[1, 2])
    ax_tbl.axis("off")
    rows = [
        ["Метрика", "Left", "Right"],
        ["MSD", f"{metrics['msd_left']:.4f}", f"{metrics['msd_right']:.4f}"],
        ["HD", f"{metrics['hausdorff_left']:.4f}", f"{metrics['hausdorff_right']:.4f}"],
        ["EP err", f"{metrics['endpoint_error_left']:.4f}", f"{metrics['endpoint_error_right']:.4f}"],
        ["Speed var", f"{metrics['speed_variance_left']:.4f}", f"{metrics['speed_variance_right']:.4f}"],
        ["Curvature", f"{metrics['curvature_left']:.4f}", f"{metrics['curvature_right']:.4f}"],
        ["MSD XY", f"{metrics['msd_xy_left']:.4f}", f"{metrics['msd_xy_right']:.4f}"],
        ["MSD XZ", f"{metrics['msd_xz_left']:.4f}", f"{metrics['msd_xz_right']:.4f}"],
        ["MSD YZ", f"{metrics['msd_yz_left']:.4f}", f"{metrics['msd_yz_right']:.4f}"],
        ["MSD mean", f"{metrics['msd_mean']:.4f}", ""],
        ["HD mean", f"{metrics['hausdorff_mean']:.4f}", ""],
    ]

    tbl = ax_tbl.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.5)
    ax_tbl.set_title("Метрики", fontsize=9)

    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return metrics


# ---------------------------------------------------------------------------
# Визуализация — HTML
# ---------------------------------------------------------------------------

def save_curve_scene_html(
    pred_pts_left: np.ndarray,
    pred_pts_right: np.ndarray,
    t_gt_left: np.ndarray,
    pts_gt_left: np.ndarray,
    t_gt_right: np.ndarray,
    pts_gt_right: np.ndarray,
    mask: np.ndarray,
    save_path: Path,
    sample_name: str = "",
    epoch: int = 0,
    n_dense: int = 256,
    title_prefix: str = "val",
) -> Path | None:
    try:
        import plotly.graph_objects as go
    except ImportError:
        meta = {
            "status": "plotly_not_installed",
            "message": "Install plotly to enable interactive HTML visualization",
            "target_html": str(save_path),
            "sample_name": sample_name,
            "epoch": int(epoch),
        }
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fallback_path = save_path.with_suffix(".plotly_missing.json")
        fallback_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return None

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    gt_left_dense = _dense_sample_curve(t_gt_left, pts_gt_left, n_dense)
    gt_right_dense = _dense_sample_curve(t_gt_right, pts_gt_right, n_dense)

    pred_left_vox = _norm_to_voxel_coords(pred_pts_left, mask.shape)
    pred_right_vox = _norm_to_voxel_coords(pred_pts_right, mask.shape)
    gt_left_vox = _norm_to_voxel_coords(gt_left_dense, mask.shape)
    gt_right_vox = _norm_to_voxel_coords(gt_right_dense, mask.shape)

    verts, faces = _mask_mesh_from_binary(mask)

    fig = go.Figure()
    if verts is not None and faces is not None and len(verts) > 0 and len(faces) > 0:
        fig.add_trace(go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color="lightgray",
            opacity=0.18,
            name="GT vessel mask",
            visible=True,
            hoverinfo="skip",
        ))

    def _add_curve(points, name, color, dash):
        fig.add_trace(go.Scatter3d(
            x=points[:, 0],
            y=points[:, 1],
            z=points[:, 2],
            mode="lines",
            name=name,
            line=dict(color=color, width=6 if dash == "solid" else 4, dash=dash),
        ))

    _add_curve(gt_left_vox, "GT left", "deepskyblue", "dash")
    _add_curve(gt_right_vox, "GT right", "lightcoral", "dash")
    _add_curve(pred_left_vox, "Pred left", "blue", "solid")
    _add_curve(pred_right_vox, "Pred right", "red", "solid")

    fig.update_layout(
        title=f"{title_prefix.upper()} | Epoch {epoch:03d} | {sample_name}",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255,255,255,0.75)",
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.write_html(str(save_path), include_plotlyjs="cdn")
    return save_path


# ---------------------------------------------------------------------------
# Вспомогательная функция для вызова из train_implicit.py
# ---------------------------------------------------------------------------

def visualize_epoch(
    model,
    dataset,
    device,
    epoch: int,
    vis_dir: Path,
    n_samples: int = 3,
    n_points: int = 256,
    split_name: str = "val",
    make_html: bool = False,
):
    import torch

    vis_dir = Path(vis_dir)
    split_dir = vis_dir / split_name
    png_dir = split_dir / "png"
    html_dir = split_dir / "html"
    png_dir.mkdir(parents=True, exist_ok=True)
    if make_html:
        html_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    keys_to_vis = dataset.keys[:n_samples]
    all_metrics = []

    with torch.no_grad():
        for key in keys_to_vis:
            img = torch.tensor(dataset.imgs[key], dtype=torch.float32, device=device).unsqueeze(0)
            pts_left = model.predict_curve(img, n_points=n_points, branch_id=0)[0].cpu().numpy()
            pts_right = model.predict_curve(img, n_points=n_points, branch_id=1)[0].cpu().numpy()

            save_path = png_dir / f"epoch_{epoch:03d}_{key}.png"
            metrics = save_curve_plots(
                pred_pts_left=pts_left,
                pred_pts_right=pts_right,
                t_gt_left=dataset.t_left[key],
                pts_gt_left=dataset.pts_left[key],
                t_gt_right=dataset.t_right[key],
                pts_gt_right=dataset.pts_right[key],
                save_path=save_path,
                sample_name=key,
                epoch=epoch,
                n_dense=256,
                title_prefix=split_name,
            )

            html_path = None
            if make_html:
                html_path = save_curve_scene_html(
                    pred_pts_left=pts_left,
                    pred_pts_right=pts_right,
                    t_gt_left=dataset.t_left[key],
                    pts_gt_left=dataset.pts_left[key],
                    t_gt_right=dataset.t_right[key],
                    pts_gt_right=dataset.pts_right[key],
                    mask=dataset.masks[key],
                    save_path=html_dir / f"epoch_{epoch:03d}_{key}.html",
                    sample_name=key,
                    epoch=epoch,
                    n_dense=256,
                    title_prefix=split_name,
                )

            all_metrics.append(metrics)
            extra = f" | html={html_path}" if html_path is not None else ""
            print(f"  [vis:{split_name}] сохранено {save_path} | MSD={metrics['msd_mean']:.4f}{extra}")

    return all_metrics
