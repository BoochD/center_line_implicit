from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import nibabel as nib
import numpy as np
import yaml

from aaa.geometry.misc import reshape


AXES = {
    "XY": (0, 1),
    "XZ": (0, 2),
    "YZ": (1, 2),
}


COLORS = {
    "left": "blue",
    "right": "red",
    "bif": "black",
}


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def load_preprocessed_shape_and_spacing(datapath: Path, name: str) -> tuple[tuple[int, int, int], np.ndarray, int, int, np.ndarray]:
    """
    Repeat the preprocessing geometry from train_implicit.ImplicitCurveDataset._load_sample.

    This intentionally follows the training code step-by-step up to the point
    where points are normalised:
        image/mask load -> central crop -> rshape/rspacing -> reshape(mask).
    """
    imgpath = datapath / "custom" / "imgs" / f"{name}.nii.gz"
    maskpath = datapath / "custom" / "masks" / f"{name}.nii.gz"

    img_nib = nib.load(str(imgpath))
    image = img_nib.dataobj[:]
    mask = nib.load(str(maskpath)).dataobj[:]

    size = image.shape[0]
    hhsize = size // 4

    image = image[hhsize:size - hhsize, hhsize:size - hhsize, :]
    mask = mask[hhsize:size - hhsize, hhsize:size - hhsize, :]

    spacing = np.abs(img_nib.affine.diagonal()[:3]).astype(np.float32)
    nspacing = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    rshape = (2 * np.array(image.shape) * spacing / nspacing) // 3
    rshape = rshape.astype(int)
    rspacing = np.array(image.shape) * spacing / rshape

    mask = reshape(mask, mask.shape, rshape, cv2.INTER_NEAREST)
    mask = (mask == 1).astype(np.uint8)

    return tuple(int(v) for v in mask.shape), spacing, int(size), int(hhsize), rspacing


def load_knots_like_training(datapath: Path, name: str, side: str, spacing: np.ndarray, rspacing: np.ndarray, size: int, hhsize: int) -> np.ndarray:
    path = datapath / "custom" / "knots" / f"{name}_{side}.json"
    with open(path, "r", encoding="utf-8") as f:
        pts = np.array(json.load(f)["knots"], dtype=np.float32)

    # Stored as (z, x, y) -> model/training order (x, y, z).
    pts = pts[:, [1, 2, 0]]
    pts[:, 0] = np.clip(pts[:, 0], hhsize, size - hhsize - 1)
    pts[:, 1] = np.clip(pts[:, 1], hhsize, size - hhsize - 1)
    pts -= np.array([hhsize, hhsize, 0], dtype=np.float32)
    pts = pts * spacing / rspacing
    return pts.astype(np.float32)


def normalize_pair(pts_l: np.ndarray, pts_r: np.ndarray, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    shape_arr = np.array(shape, dtype=np.float32)

    pts_l_norm = 2.0 * (pts_l / (shape_arr - 1.0)) - 1.0
    pts_r_norm = 2.0 * (pts_r / (shape_arr - 1.0)) - 1.0

    bif_point = 0.5 * (pts_l_norm[-1] + pts_r_norm[-1])
    pts_l_bif = pts_l_norm - bif_point
    pts_r_bif = pts_r_norm - bif_point

    return {
        "left_standard": pts_l_norm,
        "right_standard": pts_r_norm,
        "left_bif": pts_l_bif,
        "right_bif": pts_r_bif,
        "bif_point_standard": bif_point,
        "bif_point_bif": np.zeros(3, dtype=np.float32),
    }


def axis_limits(arrays: list[np.ndarray], axis_a: int, axis_b: int, pad: float = 0.08) -> tuple[tuple[float, float], tuple[float, float]]:
    pts = np.concatenate(arrays, axis=0)
    mins = pts[:, [axis_a, axis_b]].min(axis=0)
    maxs = pts[:, [axis_a, axis_b]].max(axis=0)
    span = np.maximum(maxs - mins, 1e-3)
    mins = mins - pad * span
    maxs = maxs + pad * span
    return (float(mins[0]), float(maxs[0])), (float(mins[1]), float(maxs[1]))


def plot_case_png(name: str, norm: dict[str, np.ndarray], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    modes = ["standard", "bif"]

    for row, mode in enumerate(modes):
        left = norm[f"left_{mode}"]
        right = norm[f"right_{mode}"]
        bif = norm[f"bif_point_{mode}"]

        for col, (plane, (a, b)) in enumerate(AXES.items()):
            ax = axes[row, col]
            ax.plot(left[:, a], left[:, b], color=COLORS["left"], marker="o", markersize=2, linewidth=1.5)
            ax.plot(right[:, a], right[:, b], color=COLORS["right"], marker="o", markersize=2, linewidth=1.5)
            ax.scatter([bif[a]], [bif[b]], color=COLORS["bif"], s=80, marker="x", linewidths=2.5)
            ax.scatter([left[-1, a]], [left[-1, b]], color=COLORS["left"], s=45, marker="s")
            ax.scatter([right[-1, a]], [right[-1, b]], color=COLORS["right"], s=45, marker="s")

            xlim, ylim = axis_limits([left, right, bif[None]], a, b)
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(True, alpha=0.3)
            ax.set_title(plane)
            ax.set_xlabel("XYZ"[a])
            ax.set_ylabel("XYZ"[b])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_case_html(name: str, norm: dict[str, np.ndarray], out_path: Path) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except Exception as e:
        print(f"[html] plotly unavailable, skip HTML for {name}: {e}")
        return

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("standard volume normalization", "bifurcation-centered normalization"),
    )

    for col, mode in enumerate(["standard", "bif"], start=1):
        for side, color in [("left", "blue"), ("right", "red")]:
            pts = norm[f"{side}_{mode}"]
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0],
                    y=pts[:, 1],
                    z=pts[:, 2],
                    mode="lines+markers",
                    marker={"size": 3, "color": color},
                    line={"width": 5, "color": color},
                    name=f"{mode} {side}",
                    showlegend=True,
                ),
                row=1,
                col=col,
            )
        bif = norm[f"bif_point_{mode}"]
        fig.add_trace(
            go.Scatter3d(
                x=[bif[0]],
                y=[bif[1]],
                z=[bif[2]],
                mode="markers",
                marker={"size": 8, "color": "red", "symbol": "x"},
                name=f"{mode} bif",
                showlegend=True,
            ),
            row=1,
            col=col,
        )

    fig.update_layout(title=f"GT normalization comparison: {name}", width=1400, height=700)
    for scene_name in ["scene", "scene2"]:
        fig.update_layout(**{
            scene_name: {
                "xaxis_title": "X",
                "yaxis_title": "Y",
                "zaxis_title": "Z",
                "aspectmode": "data",
            }
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def print_stats(name: str, norm: dict[str, np.ndarray], shape: tuple[int, int, int]) -> None:
    print(f"\n[{name}] preprocessed shape={shape}")
    for mode in ["standard", "bif"]:
        left = norm[f"left_{mode}"]
        right = norm[f"right_{mode}"]
        both = np.concatenate([left, right], axis=0)
        print(f"  {mode}:")
        print(f"    bif_point={norm[f'bif_point_{mode}'].round(4).tolist()}")
        print(f"    left  min={left.min(axis=0).round(4).tolist()} max={left.max(axis=0).round(4).tolist()} last={left[-1].round(4).tolist()}")
        print(f"    right min={right.min(axis=0).round(4).tolist()} max={right.max(axis=0).round(4).tolist()} last={right[-1].round(4).tolist()}")
        print(f"    both  min={both.min(axis=0).round(4).tolist()} max={both.max(axis=0).round(4).tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare standard GT normalization with bifurcation-centered normalization.")
    parser.add_argument("--datapath", type=Path, default=Path("cdata"), help="Dataset root with custom/imgs, custom/masks and custom/knots")
    parser.add_argument("--split-options", type=Path, default=Path("afolds/foldx.yaml"), help="YAML split file with train/val/test lists")
    parser.add_argument("--out-dir", type=Path, default=Path("vis/bifurcation_norm_compare"))
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--num", type=int, default=2)
    parser.add_argument("--names", nargs="*", default=None, help="Explicit case names, e.g. --names 0017 0018")
    args = parser.parse_args()

    split = load_yaml(args.split_options)

    if args.names:
        names = [str(x) for x in args.names]
    else:
        names = [str(x) for x in split.get(args.split, [])[: args.num]]

    if not names:
        raise RuntimeError(f"No cases selected for split={args.split!r}; check {args.split_options}")

    print(f"[datapath] {args.datapath}")
    print(f"[split_options] {args.split_options}")
    print(f"[cases] {names}")
    print(f"[out] {args.out_dir}")

    for name in names:
        shape, spacing, size, hhsize, rspacing = load_preprocessed_shape_and_spacing(args.datapath, name)
        pts_l = load_knots_like_training(args.datapath, name, "l", spacing, rspacing, size, hhsize)
        pts_r = load_knots_like_training(args.datapath, name, "r", spacing, rspacing, size, hhsize)
        norm = normalize_pair(pts_l, pts_r, shape)

        print_stats(name, norm, shape)
        plot_case_png(name, norm, args.out_dir / f"{name}_normalization_compare.png")
        save_case_html(name, norm, args.out_dir / f"{name}_normalization_compare.html")

    print("\nDone.")


if __name__ == "__main__":
    main()
