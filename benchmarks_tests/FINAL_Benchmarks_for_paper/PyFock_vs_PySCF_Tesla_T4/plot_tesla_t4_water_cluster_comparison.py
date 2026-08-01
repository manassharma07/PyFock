#!/usr/bin/env python3
"""Plot the Kaggle Tesla T4 water-cluster timing benchmarks.

The script reads the parsed Excel workbook in this directory, excludes the
(H2O)5 cold-start point, and compares PyFock with GPU4PySCF after removing the
GPU4PySCF grid-generation time. Both replicate runs are shown as faint lines,
while the two-run means and run-to-run ranges are emphasized.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "tesla_t4_water_cluster_benchmark_parsed.xlsx"
DEFAULT_OUTPUT = SCRIPT_DIR / "tesla_t4_water_cluster_timing_comparison"

WATER_COLUMN = "Water molecules"
BASIS_COLUMN = "Basis functions"
RUN_COLUMN = "Run"

METHODS = {
    "GPU4PySCF": {
        "total": "GPU4PySCF wall time excl. grid (s)",
        "iteration": "GPU4PySCF wall time excl. grid / iteration (s)",
        "color": "#0072B2",
        "marker": "o",
    },
    "PyFock": {
        "total": "PyFock total wall time (s)",
        "iteration": "PyFock wall time / iteration (s)",
        "color": "#D55E00",
        "marker": "s",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-quality Tesla T4 timing plots from the parsed "
            "water-cluster benchmark workbook."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input workbook (default: {DEFAULT_INPUT.name})",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=(
            "Output path without an extension. Both PNG and PDF are written "
            f"(default: {DEFAULT_OUTPUT.name})."
        ),
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="Use linear axes instead of the default log-log axes.",
    )
    return parser.parse_args()


def load_data(workbook_path: Path) -> pd.DataFrame:
    data = pd.read_excel(workbook_path, sheet_name="Raw Data")
    required = {
        RUN_COLUMN,
        WATER_COLUMN,
        BASIS_COLUMN,
        *(method[column] for method in METHODS.values() for column in ("total", "iteration")),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")

    data = data.loc[data[WATER_COLUMN] != 5].copy()
    numeric_columns = [
        RUN_COLUMN,
        WATER_COLUMN,
        BASIS_COLUMN,
        *(method[column] for method in METHODS.values() for column in ("total", "iteration")),
    ]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    data.sort_values([WATER_COLUMN, RUN_COLUMN], inplace=True)

    counts = data.groupby(WATER_COLUMN)[RUN_COLUMN].nunique()
    if not (counts == 2).all():
        raise ValueError(
            "Expected two distinct runs for every retained water cluster; "
            f"found {counts.to_dict()}."
        )
    return data


def plot_metric(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric_key: str,
    ylabel: str,
    panel_label: str,
    use_log_axes: bool,
) -> None:
    clusters = (
        data[[WATER_COLUMN, BASIS_COLUMN]]
        .drop_duplicates()
        .sort_values(BASIS_COLUMN)
    )
    basis_functions = clusters[BASIS_COLUMN].to_numpy()
    water_counts = clusters[WATER_COLUMN].to_numpy()

    for method_name, method in METHODS.items():
        timing_column = method[metric_key]
        color = method["color"]
        marker = method["marker"]

        for run_number, run_data in data.groupby(RUN_COLUMN):
            run_data = run_data.sort_values(BASIS_COLUMN)
            ax.plot(
                run_data[BASIS_COLUMN],
                run_data[timing_column],
                color=color,
                linewidth=0.9,
                alpha=0.24,
                linestyle="--",
                marker=marker,
                markersize=3.5,
                markerfacecolor="white",
                markeredgewidth=0.7,
                label=f"{method_name}, run {int(run_number)}",
                zorder=2,
            )

        grouped = data.groupby(BASIS_COLUMN)[timing_column]
        mean = grouped.mean().reindex(basis_functions).to_numpy()
        minimum = grouped.min().reindex(basis_functions).to_numpy()
        maximum = grouped.max().reindex(basis_functions).to_numpy()
        error = np.vstack((mean - minimum, maximum - mean))

        ax.errorbar(
            basis_functions,
            mean,
            yerr=error,
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.3,
            capsize=3,
            capthick=0.9,
            label=f"{method_name}, two-run mean",
            zorder=4,
        )

    if use_log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")

    ax.set_xticks(basis_functions)
    ax.set_xticklabels(
        [f"{basis:,}\n$(\\mathrm{{H_2O}})_{{{water}}}$"
         for basis, water in zip(basis_functions, water_counts)]
    )
    ax.tick_params(axis="x", which="minor", labelbottom=False)
    ax.set_xlabel("Number of basis functions and water cluster")
    ax.set_ylabel(ylabel)
    ax.grid(which="major", color="#CBD5E1", linewidth=0.6, alpha=0.72)
    ax.grid(which="minor", color="#E2E8F0", linewidth=0.35, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.015,
        0.98,
        panel_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        fontweight="bold",
    )


def main() -> None:
    args = parse_args()
    data = load_data(args.input)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.labelsize": 8.5,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axes = plt.subplots(1, 2, figsize=(7.2, 3.25))
    plot_metric(
        axes[0],
        data,
        metric_key="total",
        ylabel="Grid-excluded wall time (s)",
        panel_label="(a) Total SCF wall time",
        use_log_axes=not args.linear,
    )
    plot_metric(
        axes[1],
        data,
        metric_key="iteration",
        ylabel="Grid-excluded wall time per SCF iteration (s)",
        panel_label="(b) Wall time per SCF iteration",
        use_log_axes=not args.linear,
    )

    handles, labels = axes[0].get_legend_handles_labels()
    mean_indices = [index for index, label in enumerate(labels) if "mean" in label]
    figure.legend(
        [handles[index] for index in mean_indices],
        [labels[index] for index in mean_indices],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
        handlelength=2.4,
        columnspacing=2.0,
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.995,
        bottom=0.235,
        top=0.84,
        wspace=0.31,
    )

    output_prefix = args.output_prefix.resolve()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(figure)

    print(f"Read: {args.input.resolve()}")
    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
