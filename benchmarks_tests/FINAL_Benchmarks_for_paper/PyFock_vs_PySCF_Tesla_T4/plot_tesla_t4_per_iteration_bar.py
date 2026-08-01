#!/usr/bin/env python3
"""Create a grouped bar plot of grid-excluded SCF iteration timings.

The input is the parsed Tesla T4 benchmark workbook in this directory.
The (H2O)5 cold-start benchmark is excluded. Bars show two-run means.
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
DEFAULT_OUTPUT = SCRIPT_DIR / "tesla_t4_per_iteration_bar"

RUN_COLUMN = "Run"
WATER_COLUMN = "Water molecules"
BASIS_COLUMN = "Basis functions"
METHODS = {
    "GPU4PySCF": {
        "column": "GPU4PySCF wall time excl. grid / iteration (s)",
        "color": "#0072B2",
    },
    "PyFock GPU": {
        "column": "PyFock wall time / iteration (s)",
        "color": "#D55E00",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot two-run mean wall times per SCF iteration for GPU4PySCF "
            "and PyFock on the Kaggle Tesla T4."
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
    return parser.parse_args()


def load_data(workbook_path: Path) -> pd.DataFrame:
    data = pd.read_excel(workbook_path, sheet_name="Raw Data")
    required = {
        RUN_COLUMN,
        WATER_COLUMN,
        BASIS_COLUMN,
        *(method["column"] for method in METHODS.values()),
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Workbook is missing required columns: {missing}")

    data = data.loc[data[WATER_COLUMN] != 5].copy()
    numeric_columns = [
        RUN_COLUMN,
        WATER_COLUMN,
        BASIS_COLUMN,
        *(method["column"] for method in METHODS.values()),
    ]
    data[numeric_columns] = data[numeric_columns].apply(
        pd.to_numeric, errors="raise"
    )
    data.sort_values([WATER_COLUMN, RUN_COLUMN], inplace=True)

    run_counts = data.groupby(WATER_COLUMN)[RUN_COLUMN].nunique()
    if not (run_counts == 2).all():
        raise ValueError(
            "Expected two runs for every retained cluster; "
            f"found {run_counts.to_dict()}."
        )
    return data


def main() -> None:
    args = parse_args()
    data = load_data(args.input)
    clusters = (
        data[[WATER_COLUMN, BASIS_COLUMN]]
        .drop_duplicates()
        .sort_values(WATER_COLUMN)
    )
    water_counts = clusters[WATER_COLUMN].to_numpy()
    basis_functions = clusters[BASIS_COLUMN].to_numpy()
    positions = np.arange(len(clusters), dtype=float)
    width = 0.34

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    figure, axis = plt.subplots(figsize=(6.4, 3.45))
    offsets = [-width / 2, width / 2]

    for offset, (method_name, method) in zip(offsets, METHODS.items()):
        timing_column = method["column"]
        grouped = data.groupby(WATER_COLUMN)[timing_column]
        means = grouped.mean().reindex(water_counts).to_numpy()
        bar_positions = positions + offset

        bars = axis.bar(
            bar_positions,
            means,
            width=width,
            color=method["color"],
            edgecolor="#1F2937",
            linewidth=0.65,
            label=method_name,
            zorder=2,
        )

        for bar, mean in zip(bars, means):
            axis.annotate(
                f"{mean:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
                color="#111827",
            )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"$\\mathbf{{(H_2O)_{{{water}}}}}$\n{basis:,} functions"
            for water, basis in zip(water_counts, basis_functions)
        ],
        fontweight="bold",
    )
    axis.set_ylabel("Wall time per SCF iteration (S)", fontweight="bold")
    axis.set_ylim(0, 1.17 * data[[method["column"] for method in METHODS.values()]].max().max())
    axis.grid(axis="y", color="#CBD5E1", linewidth=0.6, alpha=0.75, zorder=0)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(
        loc="upper left",
        frameon=False,
        ncol=2,
        handlelength=1.8,
        columnspacing=1.6,
    )
    figure.subplots_adjust(left=0.105, right=0.99, bottom=0.21, top=0.96)

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
