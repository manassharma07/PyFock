#!/usr/bin/env python3
"""Plot PyFock CAO ERI memory requirements as a function of system size.

The script reads the ``Raw Data`` sheet in the Excel workbook stored beside
this file and writes publication-ready PNG and vector PDF figures.

Dependencies
------------
numpy, pandas, matplotlib, openpyxl

Examples
--------
Use the workbook and output directory beside this script:

    python plot_eri_memory_requirements.py

Specify a different workbook or output directory:

    python plot_eri_memory_requirements.py --workbook results.xlsx \
        --output-dir figures
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "pyfock-eri-memory-plot-cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, ScalarFormatter  # noqa: E402


DEFAULT_WORKBOOK = "PyFock_CAO_1e-9_ERI_Memory_Requirements.xlsx"
DEFAULT_OUTPUT_STEM = "eri_total_memory_scaling_CAO_1e-9"
RAW_SHEET = "Raw Data"

DATA_COLOR = "#4477AA"
FIT_COLOR = "#CC6677"
REFERENCE_COLOR = "#9AA5AE"
GRID_COLOR = "#D9E1E8"
TEXT_COLOR = "#202124"


def parse_arguments() -> argparse.Namespace:
    """Return command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-quality plot of stored CAO ERI memory "
            "requirements from the local benchmark workbook."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=script_dir / DEFAULT_WORKBOOK,
        help="Benchmark workbook (default: workbook beside this script).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Directory for PNG and PDF outputs (default: script directory).",
    )
    parser.add_argument(
        "--output-stem",
        default=DEFAULT_OUTPUT_STEM,
        help=f"Output filename without extension (default: {DEFAULT_OUTPUT_STEM}).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="PNG resolution in dots per inch (default: 600).",
    )
    return parser.parse_args()


def load_memory_data(workbook_path: Path) -> pd.DataFrame:
    """Read, validate, and return the ERI-memory benchmark data."""
    if not workbook_path.is_file():
        raise FileNotFoundError(f"Benchmark workbook not found: {workbook_path}")

    data = pd.read_excel(
        workbook_path,
        sheet_name=RAW_SHEET,
        header=3,
        engine="openpyxl",
    )
    data.columns = [str(column).strip() for column in data.columns]

    required_columns = {
        "System",
        "H2O molecules",
        "2c2e ERI (GB)",
        "Screened 3c2e ERI (GB)",
    }
    missing = sorted(required_columns.difference(data.columns))
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")

    data = data.dropna(subset=["System", "H2O molecules"]).copy()
    numeric_columns = [
        "H2O molecules",
        "2c2e ERI (GB)",
        "Screened 3c2e ERI (GB)",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data["H2O molecules"] = data["H2O molecules"].astype(int)
    data["ERI total calculated (GB)"] = (
        data["2c2e ERI (GB)"] + data["Screened 3c2e ERI (GB)"]
    )

    if "ERI total (GB)" in data.columns:
        workbook_totals = pd.to_numeric(data["ERI total (GB)"], errors="coerce")
        available = workbook_totals.notna()
        if available.any() and not np.allclose(
            workbook_totals.loc[available],
            data.loc[available, "ERI total calculated (GB)"],
            rtol=1e-10,
            atol=1e-12,
        ):
            raise ValueError(
                "Workbook ERI totals do not equal 2c2e + screened 3c2e storage."
            )

    if data["H2O molecules"].duplicated().any():
        raise ValueError("Duplicate water-cluster sizes were found.")
    if (data[numeric_columns[1:]] <= 0).any().any():
        raise ValueError("All ERI storage values must be positive.")

    return data.sort_values("H2O molecules").reset_index(drop=True)


def fit_power_law(
    cluster_sizes: np.ndarray, memory_gb: np.ndarray
) -> tuple[float, float, float]:
    """Fit memory = prefactor * cluster_size**exponent in log space."""
    log_sizes = np.log(cluster_sizes)
    log_memory = np.log(memory_gb)
    exponent, log_prefactor = np.polyfit(log_sizes, log_memory, 1)
    predicted = log_prefactor + exponent * log_sizes
    residual_sum = np.sum((log_memory - predicted) ** 2)
    total_sum = np.sum((log_memory - log_memory.mean()) ** 2)
    r_squared = 1.0 - residual_sum / total_sum
    return float(np.exp(log_prefactor)), float(exponent), float(r_squared)


def publication_style() -> None:
    """Apply restrained journal-figure defaults."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.labelweight": "bold",
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.linewidth": 0.9,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "legend.fontsize": 10.5,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def format_memory_label(value: float) -> str:
    """Return a compact data-label string in GB."""
    if value < 0.1:
        return f"{value:.3f}"
    if value < 1:
        return f"{value:.3f}"
    if value < 10:
        return f"{value:.2f}"
    return f"{value:.1f}"


def plot_memory_scaling(data: pd.DataFrame) -> tuple[plt.Figure, dict[str, float]]:
    """Build the publication-quality ERI-memory scaling plot."""
    publication_style()

    sizes = data["H2O molecules"].to_numpy(dtype=float)
    memory = data["ERI total calculated (GB)"].to_numpy(dtype=float)
    prefactor, exponent, r_squared = fit_power_law(sizes, memory)

    figure, axis = plt.subplots(figsize=(7.2, 4.8))
    figure.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.22)

    fit_sizes = np.geomspace(sizes.min() * 0.9, sizes.max() * 1.1, 300)
    fit_memory = prefactor * fit_sizes**exponent
    fit_label = (
        rf"Power-law fit: $M={prefactor * 1e4:.2f}\times10^{{-4}}"
        rf"n^{{{exponent:.2f}}}$ ($R^2={r_squared:.3f}$)"
    )

    axis.plot(
        fit_sizes,
        fit_memory,
        color=FIT_COLOR,
        linewidth=1.8,
        linestyle="--",
        label=fit_label,
        zorder=2,
    )
    axis.plot(
        sizes,
        memory,
        color=DATA_COLOR,
        linewidth=2.2,
        marker="o",
        markersize=7.5,
        markerfacecolor="white",
        markeredgecolor=DATA_COLOR,
        markeredgewidth=2.0,
        label="Measured ERI total",
        zorder=4,
    )

    for capacity in (16, 32, 64):
        axis.axhline(
            capacity,
            color=REFERENCE_COLOR,
            linewidth=0.9,
            linestyle=":",
            zorder=1,
        )

    label_offsets = [
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
        (0, 10),
    ]
    for size, value, offset in zip(sizes, memory, label_offsets):
        axis.annotate(
            format_memory_label(value),
            xy=(size, value),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "center",
            va="top" if offset[1] < 0 else "bottom",
            fontsize=9.5,
            fontweight="bold",
            color=DATA_COLOR,
            zorder=5,
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(4.3, 165)
    axis.set_ylim(0.014, 100)
    axis.set_xticks(sizes)
    axis.xaxis.set_major_formatter(ScalarFormatter())
    axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axis.set_yticks([0.02, 0.1, 1, 10, 16, 32, 64, 100])
    axis.yaxis.set_major_formatter(
        FuncFormatter(
            lambda value, _: (
                f"{value:g} GB" if value in {16, 32, 64} else f"{value:g}"
            )
        )
    )

    axis.set_xlabel(
        r"Water-cluster size, $n$ in $(\mathrm{H_2O})_n$",
        labelpad=8,
    )
    axis.set_ylabel("Stored ERI memory (GB)")
    axis.set_title("ERI storage requirements", pad=12)
    axis.grid(
        which="major",
        axis="both",
        color=GRID_COLOR,
        linewidth=0.8,
        alpha=0.9,
        zorder=0,
    )
    axis.grid(
        which="minor",
        axis="y",
        color=GRID_COLOR,
        linewidth=0.5,
        alpha=0.45,
        zorder=0,
    )
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(which="both", direction="out", length=4)
    axis.legend(loc="upper left", frameon=False)

    figure.text(
        0.5,
        0.035,
        (
            "ERI total = 2c2e + screened 3c2e storage; "
            "CAO, def2-SVP/def2-universal-jfit; decimal GB."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#5F6368",
    )

    return figure, {
        "prefactor": prefactor,
        "exponent": exponent,
        "r_squared": r_squared,
    }


def main() -> None:
    """Read the workbook, generate the figure, and save PNG/PDF outputs."""
    args = parse_arguments()
    data = load_memory_data(args.workbook.resolve())
    figure, fit = plot_memory_scaling(data)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    png_path = output_dir / f"{args.output_stem}.png"
    pdf_path = output_dir / f"{args.output_stem}.pdf"

    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)

    print(f"Loaded {len(data)} water clusters from: {args.workbook.resolve()}")
    print(
        "Power-law fit: "
        f"M = {fit['prefactor']:.6f} n^{fit['exponent']:.4f}, "
        f"R^2 = {fit['r_squared']:.6f}"
    )
    for cluster_size, memory_gb in zip(
        data["H2O molecules"],
        data["ERI total calculated (GB)"],
    ):
        print(
            f"(H2O){int(cluster_size)}: "
            f"{float(memory_gb):.9f} GB"
        )
    print(f"Saved PNG: {png_path}")
    print(f"Saved PDF: {pdf_path}")


if __name__ == "__main__":
    main()
