#!/usr/bin/env python3
"""
Python port of ResultShower_picmus_images.m (Fig.3 PICMUS image grid).

This script reads:
- DAS reconstructions from HDF5
- DRUS/WDRUS reconstructions from MAT files

and saves a 4x7 figure matching the MATLAB layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import h5py
import matplotlib
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import maximum_filter1d


MODELS = [
    "DAS1",
    "DAS11",
    "DAS75",
    "BH50",
    "CBH50",
    "BH50_fineTune",
    "CBH50_fineTune",
]

MODEL_LABELS = ["DAS1", "DAS11", "DAS75", "DRUS", "WDRUS", "DRUS", "WDRUS"]
GROUP_LABELS = ["Before fine-tuning", "After fine-tuning"]
ROW_LABELS_BOTTOM_TO_TOP = ["EC", "ER", "SC", "SR"]

DAS_PLANE_INDEX = {"DAS1": 0, "DAS11": 2, "DAS75": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PICMUS image grid (Python port of MATLAB ResultShower_picmus_images.m)."
    )
    parser.add_argument(
        "--picmus-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory containing DAS/, BH/, CBH/ folders (default: script directory).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "ResultShower_picmus_images_python.png",
        help="Output figure path.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure DPI.",
    )
    parser.add_argument(
        "--envelope-size",
        type=int,
        default=1,
        help=(
            "Peak-envelope window size for non-DAS rows (MATLAB uses envelope(...,1,'peak')). "
            "Use 1 to match near-identity behavior."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive figure window (default: save only).",
    )
    return parser.parse_args()


def to_zxpw(data_3d: np.ndarray) -> np.ndarray:
    """
    Normalize a 3D array to [z, x, pw] layout.
    """
    data_3d = np.asarray(data_3d)
    if data_3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {data_3d.shape}")

    # Common PICMUS storage: [pw, z, x] with pw being small (e.g., 4).
    if data_3d.shape[0] <= 32 and data_3d.shape[1] == data_3d.shape[2]:
        return np.transpose(data_3d, (1, 2, 0))

    # Already [z, x, pw].
    if data_3d.shape[2] <= 32 and data_3d.shape[0] == data_3d.shape[1]:
        return data_3d

    # Fallback: move smallest axis to pw.
    pw_axis = int(np.argmin(data_3d.shape))
    return np.moveaxis(data_3d, pw_axis, -1)


def load_scan_axes(picmus_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Try loading scan axes from DAS HDF5. Fallback to MATLAB hardcoded axes.
    """
    das_path = picmus_dir / "DAS" / "1.hdf5"
    if das_path.exists():
        with h5py.File(das_path, "r") as f:
            x_key = "/US/US_DATASET0000/scan/x_axis"
            z_key = "/US/US_DATASET0000/scan/z_axis"
            if x_key in f and z_key in f:
                return np.asarray(f[x_key]), np.asarray(f[z_key])

    x_axis = np.linspace(-0.018, 0.018, 256)
    z_axis = np.linspace(0.01, 0.046, 256)
    return x_axis, z_axis


def load_das_plane(h5_path: Path, plane_idx: int) -> np.ndarray:
    with h5py.File(h5_path, "r") as f:
        real_key = "/US/US_DATASET0000/data/real"
        imag_key = "/US/US_DATASET0000/data/imag"

        if real_key in f and imag_key in f:
            real = np.asarray(f[real_key])
            imag = np.asarray(f[imag_key])
            data = real + 1j * imag
        else:
            raise KeyError(
                f"Could not find expected DAS keys in {h5_path}: "
                f"'{real_key}' and '{imag_key}'."
            )

    data_zxpw = to_zxpw(data)
    if plane_idx >= data_zxpw.shape[2]:
        raise IndexError(
            f"Plane index {plane_idx} out of range for {h5_path} with shape {data_zxpw.shape}."
        )
    return data_zxpw[:, :, plane_idx]


def load_avg_from_mat(mat_path: Path) -> np.ndarray:
    data = loadmat(mat_path)
    if "x" not in data:
        raise KeyError(f"Variable 'x' not found in {mat_path}")

    x = np.asarray(data["x"])
    if x.ndim != 3:
        raise ValueError(f"Expected 3D 'x' in {mat_path}, got shape {x.shape}")

    if x.shape[0] == 3:
        return np.mean(x, axis=0)
    if x.shape[-1] == 3:
        return np.mean(x, axis=-1)

    # Fallback: average over smallest axis.
    return np.mean(x, axis=int(np.argmin(x.shape)))


def log_compress(img: np.ndarray) -> np.ndarray:
    env = np.abs(np.asarray(img))
    eps = 1e-12
    peak = float(np.max(env))
    if peak <= 0 or not np.isfinite(peak):
        return np.full(env.shape, -60.0, dtype=np.float32)
    return (20.0 * np.log10(np.maximum(env, eps) / max(peak, eps))).astype(np.float32)


def apply_peak_envelope(log_img: np.ndarray, window_size: int) -> np.ndarray:
    """
    Approximate MATLAB envelope(...,1,'peak') in 2D visualization path.
    With window_size=1 (default), this is identity.
    """
    if window_size <= 1:
        return log_img
    return maximum_filter1d(log_img, size=window_size, axis=0, mode="nearest")


def should_use_envelope(model_idx: int, row_idx: int) -> bool:
    # MATLAB condition:
    # if ((~any(i == [1,2,3])) && (~any(j == [1,3])))
    # with 1-based indices.
    return (model_idx not in {0, 1, 2}) and (row_idx not in {0, 2})


def load_all_images(picmus_dir: Path) -> List[List[np.ndarray]]:
    num_phan = 4
    all_images: List[List[np.ndarray]] = [[None for _ in MODELS] for _ in range(num_phan)]  # type: ignore[list-item]

    for model_idx, model_name in enumerate(MODELS):
        for phan_idx in range(1, num_phan + 1):
            if model_name in DAS_PLANE_INDEX:
                h5_path = picmus_dir / "DAS" / f"{phan_idx}.hdf5"
                img = load_das_plane(h5_path, DAS_PLANE_INDEX[model_name])
            elif model_name in {"BH50", "BH50_fineTune"}:
                mat_path = picmus_dir / "BH" / "Results" / model_name / f"{phan_idx}_-1.mat"
                img = load_avg_from_mat(mat_path)
            elif model_name in {"CBH50", "CBH50_fineTune"}:
                mat_path = picmus_dir / "CBH" / "Results" / model_name / f"{phan_idx}_-1.mat"
                img = load_avg_from_mat(mat_path)
            else:
                raise ValueError(f"Unsupported model: {model_name}")

            all_images[phan_idx - 1][model_idx] = np.asarray(img)

    return all_images


def plot_figure(
    all_images: List[List[np.ndarray]],
    x_axis: np.ndarray,
    z_axis: np.ndarray,
    output_path: Path,
    dpi: int,
    envelope_size: int,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    num_rows = 4
    num_cols = len(MODELS)

    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 2.0, num_rows * 1.9),
        constrained_layout=False,
    )
    plt.subplots_adjust(left=0.035, right=0.925, bottom=0.0, top=0.925, wspace=0.0, hspace=0.0)

    extent = [float(np.min(x_axis) * 1e3), float(np.max(x_axis) * 1e3), float(np.max(z_axis) * 1e3), float(np.min(z_axis) * 1e3)]

    mappable = None
    for row in range(num_rows):
        for col in range(num_cols):
            ax = axes[row, col]
            disp = log_compress(all_images[row][col])
            if should_use_envelope(col, row):
                disp = apply_peak_envelope(disp, envelope_size)

            mappable = ax.imshow(
                disp,
                cmap="gray",
                vmin=-60,
                vmax=0,
                extent=extent,
                aspect="equal",
            )
            ax.set_axis_off()

    # Group labels (top)
    group_boxes = [(0.45, 0.96, 0.2), (0.705, 0.96, 0.2)]
    for (x0, y0, width), text in zip(group_boxes, GROUP_LABELS):
        fig.text(x0 + width / 2.0, y0, text, ha="center", va="bottom", fontsize=25)

    # Model labels (top row)
    for idx, text in enumerate(MODEL_LABELS):
        x0 = idx * 0.126 + 0.028
        fig.text(x0 + 0.075, 0.92, text, ha="center", va="bottom", fontsize=25)

    # Row labels (left side, bottom -> top to mirror MATLAB annotations)
    for idx, text in enumerate(ROW_LABELS_BOTTOM_TO_TOP):
        y0 = idx * 0.24 + 0.1
        fig.text(0.015, y0 + 0.025, text, ha="center", va="center", fontsize=25)

    if mappable is not None:
        cax = fig.add_axes([0.935, 0.045, 0.015, 0.88])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.ax.tick_params(labelsize=20)

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

    picmus_dir = args.picmus_dir.resolve()
    output_path = args.output.resolve()

    all_images = load_all_images(picmus_dir)
    x_axis, z_axis = load_scan_axes(picmus_dir)

    plot_figure(
        all_images=all_images,
        x_axis=x_axis,
        z_axis=z_axis,
        output_path=output_path,
        dpi=args.dpi,
        envelope_size=args.envelope_size,
        show=args.show,
    )

    print(f"[OK] Saved figure: {output_path}")


if __name__ == "__main__":
    main()

