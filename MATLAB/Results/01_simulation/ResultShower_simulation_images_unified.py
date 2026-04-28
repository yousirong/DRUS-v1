#!/usr/bin/env python3
"""
Unified Python port of ResultShower_simulation_images.m.

Supports:
- phantomIdx=1 (kidney)
- phantomIdx=2 (fetus)
- phantomIdx=3 (simuComplex)

Differences from MATLAB:
- Uses grayscale colormap/colorbar for all phantoms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import matplotlib
import numpy as np
from scipy.io import loadmat


PHANTOM_NAME = {1: "kidney", 2: "fetus", 3: "simuComplex"}
MODELS = ["DAS", "Hty_50it", "CHty_50it"]
MODEL_LABELS_RIGHT = ["WDRUS", "DRUS", "Baseline"]
GAMMA = [0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
DEFAULT_NOISE_LEVELS = [1, 3, 6]  # MATLAB 1-based indices
IMG_SIZE = 256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified simulation figure plotter (kidney/fetus/simuComplex)."
    )
    parser.add_argument(
        "--sim-root",
        type=Path,
        default=Path(__file__).resolve().parent / "SimulationResults",
        help="Root folder containing SimulationResults/{1,2,3}.",
    )
    parser.add_argument(
        "--phantom-idx",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="Phantom index: 1=kidney, 2=fetus, 3=simuComplex.",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="1,3,6",
        help="Comma-separated 1-based noise-level indices (default: 1,3,6).",
    )
    parser.add_argument(
        "--dynamic-range",
        type=float,
        default=60.0,
        help="Dynamic range used in MATLAB script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output figure path. Default: SimulationResults/{idx}/ResultShower_simulation_<name>_gray_python.png",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output DPI.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive window (default: save only).",
    )
    return parser.parse_args()


def parse_noise_levels(noise_levels_str: str) -> list[int]:
    parts = [p.strip() for p in noise_levels_str.split(",") if p.strip()]
    levels = [int(p) for p in parts]
    if not levels:
        raise ValueError("At least one noise level is required.")
    if any(level < 1 for level in levels):
        raise ValueError("Noise levels must be positive 1-based indices.")
    if any(level > len(GAMMA) for level in levels):
        raise ValueError(f"Noise level index must be <= {len(GAMMA)}.")
    return levels


def load_mat_variable(path: Path, var_name: str) -> np.ndarray:
    """
    Load a MATLAB variable from classic MAT or v7.3 HDF5 MAT.
    """
    try:
        data = loadmat(path)
        if var_name in data:
            return np.asarray(data[var_name])
    except NotImplementedError:
        pass

    with h5py.File(path, "r") as f:
        if var_name not in f:
            raise KeyError(f"Variable '{var_name}' not found in {path}")
        return np.asarray(f[var_name])


def reshape_matlab(vec: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    return np.reshape(vec, shape, order="F")


def normalize_abs(img: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    peak = float(np.max(np.abs(img)))
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros_like(img, dtype=np.float32)
    return (img / max(peak, eps)).astype(np.float32)


def to_display_01(x: np.ndarray, dynamic_range: float, eps: float = 1e-12) -> np.ndarray:
    # MATLAB: temp = 20*log10(abs(x)) ./ dynamicRange + 1; temp(temp<0)=0;
    val = 20.0 * np.log10(np.maximum(np.abs(x), eps)) / dynamic_range + 1.0
    return np.clip(val, 0.0, None).astype(np.float32)


def load_ground_truth(sim_dir: Path, phantom_idx: int, dynamic_range: float) -> np.ndarray:
    x_orig = load_mat_variable(sim_dir / f"o_orig_{phantom_idx}.mat", "x_orig")
    x_linear = reshape_matlab(np.ravel(x_orig), (IMG_SIZE, IMG_SIZE))

    eps = 1e-12
    peak = float(np.max(np.abs(x_linear)))
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.float32)

    x_db = 20.0 * np.log10(np.maximum(np.abs(x_linear), eps) / max(peak, eps)) + dynamic_range
    x_db = np.clip(x_db, 0.0, None)
    return (x_db / dynamic_range).astype(np.float32)


def load_das_image(sim_dir: Path, noise_level: int) -> np.ndarray:
    o_hty = load_mat_variable(sim_dir / "us_DAS" / f"{noise_level}_-1.mat", "o_Hty")
    flat = np.ravel(o_hty)
    if flat.size < IMG_SIZE * IMG_SIZE:
        raise ValueError(f"Unexpected o_Hty size for noise level {noise_level}: {flat.size}")
    return reshape_matlab(flat[: IMG_SIZE * IMG_SIZE], (IMG_SIZE, IMG_SIZE))


def load_model_image(sim_dir: Path, model_name: str, noise_level: int) -> np.ndarray:
    if model_name == "DAS":
        return load_das_image(sim_dir, noise_level)

    x = load_mat_variable(sim_dir / f"us_{model_name}" / f"{noise_level}_-1.mat", "x")
    if x.ndim != 3:
        raise ValueError(f"Expected 3D x for {model_name} noise {noise_level}, got {x.shape}")

    if x.shape[0] == 3:
        return np.mean(x, axis=0)
    if x.shape[-1] == 3:
        return np.mean(x, axis=-1)
    return np.mean(x, axis=int(np.argmin(x.shape)))


def plot_grid(
    gt_display: np.ndarray,
    model_displays: np.ndarray,
    noise_levels: Iterable[int],
    output_path: Path,
    dpi: int,
    dynamic_range: float,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    noise_levels = list(noise_levels)
    num_cols = len(noise_levels) + 1  # Ground truth + selected gamma columns
    num_rows = len(MODELS)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=((num_cols * 220 + 100) / 100.0, (num_rows * 220) / 100.0),
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.005, right=0.84, bottom=0.01, top=0.908, wspace=0.0, hspace=0.0)

    for r in range(num_rows):
        for c in range(num_cols):
            axes[r, c].set_axis_off()

    # first tile = GT, rows 2/3 first column blank.
    gt_view = gt_display * dynamic_range - dynamic_range
    mappable = axes[0, 0].imshow(
        gt_view, cmap="gray", vmin=-dynamic_range, vmax=0.0, aspect="equal"
    )
    for model_idx in range(num_rows):
        for lvl_idx in range(len(noise_levels)):
            disp = model_displays[model_idx, lvl_idx] * dynamic_range - dynamic_range
            axes[model_idx, lvl_idx + 1].imshow(
                disp, cmap="gray", vmin=-dynamic_range, vmax=0.0, aspect="equal"
            )

    cax = fig.add_axes([0.13, 0.10, 0.02, 0.52])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.ax.tick_params(labelsize=20)

    top_names = ["Ground Truth"] + [f"$\\gamma$={GAMMA[level - 1]:g}" for level in noise_levels]
    for i, text in enumerate(top_names):
        x0 = i * 0.21 + 0.022
        fig.text(x0 + 0.1, 0.965, text, ha="center", va="center", fontsize=20)

    for i, text in enumerate(MODEL_LABELS_RIGHT):
        y0 = i * 0.35 + 0.1
        fig.text(0.9, y0 + 0.025, text, ha="center", va="center", fontsize=20)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.show:
        matplotlib.use("Agg")
    if args.dynamic_range <= 0:
        raise ValueError("--dynamic-range must be > 0.")

    noise_levels = parse_noise_levels(args.noise_levels)
    sim_dir = (args.sim_root / str(args.phantom_idx)).resolve()
    if not sim_dir.exists():
        raise FileNotFoundError(f"Simulation directory not found: {sim_dir}")

    if args.output is None:
        output_path = sim_dir / f"ResultShower_simulation_{PHANTOM_NAME[args.phantom_idx]}_gray_python.png"
    else:
        output_path = args.output.resolve()

    gt_display = load_ground_truth(
        sim_dir=sim_dir,
        phantom_idx=args.phantom_idx,
        dynamic_range=args.dynamic_range,
    )

    model_displays = np.zeros((len(MODELS), len(noise_levels), IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for model_idx, model_name in enumerate(MODELS):
        for lvl_idx, noise_level in enumerate(noise_levels):
            x = load_model_image(sim_dir, model_name, noise_level)
            x = normalize_abs(x)
            model_displays[model_idx, lvl_idx] = to_display_01(x, dynamic_range=args.dynamic_range)

    plot_grid(
        gt_display=gt_display,
        model_displays=model_displays,
        noise_levels=noise_levels,
        output_path=output_path,
        dpi=args.dpi,
        dynamic_range=args.dynamic_range,
        show=args.show,
    )

    print(f"[OK] Saved figure: {output_path}")


if __name__ == "__main__":
    main()
