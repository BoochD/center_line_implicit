from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from model_implicit import ImplicitCurveNet
from train_implicit import ImplicitCurveDataset, load_yaml_config
from visualize_implicit import mean_symmetric_distance, hausdorff_distance, endpoint_error


def build_model_from_cfg(cfg: dict) -> ImplicitCurveNet:
    mc = cfg["model"]
    return ImplicitCurveNet(
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
        log_shapes=False,
    )


def load_model(cfg: dict, checkpoint_path: Path, device: torch.device) -> ImplicitCurveNet:
    model = build_model_from_cfg(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)

    # Saved checkpoints are usually plain model.state_dict(). If someone saved a
    # wrapper dict, support common keys as well.
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    # Strip common prefixes from DataParallel/AveragedModel-style checkpoints.
    if isinstance(state, dict):
        cleaned = {}
        for key, value in state.items():
            new_key = key
            for prefix in ("module.", "module.module."):
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
            cleaned[new_key] = value
        state = cleaned

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[load] missing keys: {len(missing)}")
        for key in missing[:10]:
            print(f"  missing: {key}")
    if unexpected:
        print(f"[load] unexpected keys: {len(unexpected)}")
        for key in unexpected[:10]:
            print(f"  unexpected: {key}")

    model.eval()
    return model


def make_dataset(cfg: dict, split_name: str) -> ImplicitCurveDataset:
    split = load_yaml_config(Path(cfg["split_options_path"]))
    names = [str(n) for n in split[split_name]]
    channels = {"all": {"MIN_HU": -200, "MAX_HU": 1200}}
    return ImplicitCurveDataset(
        Path(cfg["datapath"]),
        names,
        channels,
        mode="val" if split_name == "val" else "test",
        augment=False,
    )


def predict_pair(
    model: ImplicitCurveNet,
    image: torch.Tensor,
    n_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        left = model.predict_curve(image, n_points=n_points, branch_id=0)[0].detach().cpu().numpy()
        right = model.predict_curve(image, n_points=n_points, branch_id=1)[0].detach().cpu().numpy()
    return left, right


def curve_delta_report(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    point_l2 = np.linalg.norm(diff, axis=-1)
    return {
        "mean_point_l2": float(np.mean(point_l2)),
        "max_point_l2": float(np.max(point_l2)),
        "msd": float(mean_symmetric_distance(a, b)),
        "hd": float(hausdorff_distance(a, b)),
        "endpoint_delta": float(endpoint_error(a, b)),
    }


def print_delta(title: str, delta: dict[str, float]) -> None:
    print(
        f"    {title:<15} "
        f"point_l2_mean={delta['mean_point_l2']:.6f} | "
        f"point_l2_max={delta['max_point_l2']:.6f} | "
        f"MSD={delta['msd']:.6f} | "
        f"HD={delta['hd']:.6f} | "
        f"EP={delta['endpoint_delta']:.6f}"
    )


def run_case(
    model: ImplicitCurveNet,
    dataset: ImplicitCurveDataset,
    key: str,
    device: torch.device,
    n_points: int,
    noise_seed: int,
    noise_mode: str,
) -> None:
    image_np = dataset.imgs[key].astype(np.float32)
    image = torch.tensor(image_np, dtype=torch.float32, device=device).unsqueeze(0)

    zero = torch.zeros_like(image)

    gen = torch.Generator(device=device)
    gen.manual_seed(noise_seed)
    noise_raw = torch.randn(image.shape, generator=gen, device=device, dtype=image.dtype)
    if noise_mode == "standard":
        noise = noise_raw
    elif noise_mode == "matched":
        noise = noise_raw * image.std().clamp_min(1e-6) + image.mean()
    else:
        raise ValueError(f"unknown noise_mode={noise_mode!r}")

    normal_left, normal_right = predict_pair(model, image, n_points)
    zero_left, zero_right = predict_pair(model, zero, n_points)
    noise_left, noise_right = predict_pair(model, noise, n_points)

    gt_left = dataset.pts_left[key]
    gt_right = dataset.pts_right[key]

    print(f"\n[case {key}] image range=[{image.min().item():.4f}, {image.max().item():.4f}] mean={image.mean().item():.4f} std={image.std().item():.4f}")
    print(f"  normal-vs-GT left : MSD={mean_symmetric_distance(normal_left, gt_left):.6f}")
    print(f"  normal-vs-GT right: MSD={mean_symmetric_distance(normal_right, gt_right):.6f}")

    print("  LEFT input sensitivity:")
    print_delta("normal-zero", curve_delta_report(normal_left, zero_left))
    print_delta("normal-noise", curve_delta_report(normal_left, noise_left))
    print_delta("zero-noise", curve_delta_report(zero_left, noise_left))

    print("  RIGHT input sensitivity:")
    print_delta("normal-zero", curve_delta_report(normal_right, zero_right))
    print_delta("normal-noise", curve_delta_report(normal_right, noise_right))
    print_delta("zero-noise", curve_delta_report(zero_right, noise_right))

    print("  Prediction ranges:")
    print(f"    normal left [{normal_left.min():.4f}, {normal_left.max():.4f}] | right [{normal_right.min():.4f}, {normal_right.max():.4f}]")
    print(f"    zero   left [{zero_left.min():.4f}, {zero_left.max():.4f}] | right [{zero_right.min():.4f}, {zero_right.max():.4f}]")
    print(f"    noise  left [{noise_left.min():.4f}, {noise_left.max():.4f}] | right [{noise_right.min():.4f}, {noise_right.max():.4f}]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero/noise image sensitivity test for ImplicitCurveNet.")
    parser.add_argument("--config", default="config_implicit.yaml", help="Path to config yaml.")
    parser.add_argument("--checkpoint", default=None, help="Path to model checkpoint. Defaults to training.modelname from config.")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="Dataset split to test.")
    parser.add_argument("--keys", nargs="*", default=None, help="Specific case keys. If omitted, uses first --num-cases keys from split.")
    parser.add_argument("--num-cases", type=int, default=3, help="How many cases to test if --keys is omitted.")
    parser.add_argument("--n-points", type=int, default=256, help="Number of curve points to predict.")
    parser.add_argument("--noise-seed", type=int, default=42, help="Random seed for noise image.")
    parser.add_argument("--noise-mode", default="matched", choices=("matched", "standard"), help="matched: noise has same mean/std as image; standard: N(0,1).")
    parser.add_argument("--device", default=None, help="cuda/cpu. Defaults to cuda if available.")
    args = parser.parse_args()

    cfg = load_yaml_config(Path(args.config))
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = Path(args.checkpoint or cfg["training"]["modelname"])

    print(f"[config] {args.config}")
    print(f"[checkpoint] {checkpoint_path}")
    print(f"[device] {device}")

    dataset = make_dataset(cfg, args.split)
    model = load_model(cfg, checkpoint_path, device)

    if args.keys:
        keys = [str(k) for k in args.keys]
    else:
        keys = dataset.keys[:args.num_cases]

    print(f"[split] {args.split} | testing keys: {keys}")
    print("\nInterpretation:")
    print("  If normal-zero / normal-noise MSD is near 0, curve branch is almost image-agnostic.")
    print("  If those deltas are comparable to normal-vs-GT MSD, prediction strongly depends on image.")

    for idx, key in enumerate(keys):
        run_case(
            model=model,
            dataset=dataset,
            key=key,
            device=device,
            n_points=args.n_points,
            noise_seed=args.noise_seed + idx,
            noise_mode=args.noise_mode,
        )


if __name__ == "__main__":
    main()
