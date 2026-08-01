#!/usr/bin/env python3
"""Plot the effect of dynamic precision on PyFock XC timings for P100.

The script reads the ``Raw Data`` sheet of the benchmark workbook in this
directory, aggregates the three repeated runs, and writes publication-ready
PNG and vector PDF figures.

Dependencies
------------
numpy, pandas, matplotlib, openpyxl

Examples
--------
Run with the workbook and output directory inferred from this script:

    python plot_dynamic_precision_xc.py

Use a different workbook or output directory:

    python plot_dynamic_precision_xc.py --workbook results.xlsx --output-dir figures
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

_CACHE_ROOT = Path(tempfile.gettempdir()) / "pyfock-plot-cache"
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "xdg").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

GPU_LABEL = "NVIDIA Tesla P100"
DEFAULT_WORKBOOK = "PyFock_P100_Dynamic_Precision_Benchmarks_with_water_47.xlsx"
DEFAULT_OUTPUT_STEM = "dynamic_precision_xc_timings_P100"
RAW_SHEET = "Raw Data"

BASELINE_COLOR = "#4477AA"
DYNAMIC_COLOR = "#EE6677"
EFFECTIVE_COLOR = "#228833"
GRID_COLOR = "#D9E1E8"
TEXT_COLOR = "#202124"


def parse_arguments() -> argparse.Namespace:
    """Return command-line arguments."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=(
            "Create a publication-quality XC timing figure from the local "
            "PyFock dynamic-precision benchmark workbook."
        )
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=script_dir / DEFAULT_WORKBOOK,
        help="Benchmark .xlsx file (default: workbook beside this script).",
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


def load_raw_data(workbook_path: Path) -> pd.DataFrame:
    """Read and validate the workbook's flat benchmark table."""
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
        "Run",
        "System",
        "Dynamic precision",
        "XC time (s)",
        "SCF iterations",
    }
    missing = sorted(required_columns.difference(data.columns))
    if missing:
        raise ValueError(
            f"{workbook_path.name!r} is missing required columns: {missing}"
        )

    data = data.loc[:, list(required_columns)].copy()
    data = data.dropna(subset=["Run", "System", "Dynamic precision"])
    data["Run"] = pd.to_numeric(data["Run"], errors="raise").astype(int)
    data["XC time (s)"] = pd.to_numeric(data["XC time (s)"], errors="raise")
    data["SCF iterations"] = pd.to_numeric(
        data["SCF iterations"], errors="raise"
    ).astype(int)
    data["System"] = data["System"].astype(str).str.strip()
    data["Mode"] = (
        data["Dynamic precision"]
        .astype(str)
        .str.strip()
        .str.casefold()
        .map({"off": "Off", "on": "On"})
    )

    if data["Mode"].isna().any():
        invalid = sorted(data.loc[data["Mode"].isna(), "Dynamic precision"].unique())
        raise ValueError(f"Unrecognized dynamic-precision labels: {invalid}")
    if (data["XC time (s)"] <= 0).any():
        raise ValueError("All XC timings must be positive.")
    if (data["SCF iterations"] <= 0).any():
        raise ValueError("All SCF iteration counts must be positive.")

    extracted_counts = data["System"].str.extract(r"(\d+)", expand=False)
    if extracted_counts.isna().any():
        invalid = sorted(data.loc[extracted_counts.isna(), "System"].unique())
        raise ValueError(f"Could not infer H2O count from systems: {invalid}")
    data["H2O count"] = extracted_counts.astype(int)
    data["XC time / iteration (s)"] = (
        data["XC time (s)"] / data["SCF iterations"]
    )

    duplicate_keys = data.duplicated(["Run", "System", "Mode"], keep=False)
    if duplicate_keys.any():
        duplicates = data.loc[duplicate_keys, ["Run", "System", "Mode"]]
        raise ValueError(f"Duplicate run/system/mode records found:\n{duplicates}")

    mode_counts = data.groupby(["System", "Mode"]).size().unstack(fill_value=0)
    if set(mode_counts.columns) != {"Off", "On"} or (mode_counts < 2).any().any():
        raise ValueError(
            "Every system must contain repeated measurements for both Off and On."
        )

    return data.sort_values(["H2O count", "Run", "Mode"]).reset_index(drop=True)


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """Return convergence-adjusted XC time per baseline SCF iteration."""
    paired = data.pivot(
        index=["Run", "H2O count", "System"],
        columns="Mode",
        values=["XC time (s)", "SCF iterations"],
    )
    if paired.isna().any().any():
        raise ValueError("Each run/system must have both Off and On measurements.")

    paired_metrics = paired.index.to_frame(index=False)
    baseline_iterations = paired[("SCF iterations", "Off")].to_numpy()
    paired_metrics["Off effective XC time / iteration (s)"] = (
        paired[("XC time (s)", "Off")].to_numpy() / baseline_iterations
    )
    paired_metrics["On effective XC time / iteration (s)"] = (
        paired[("XC time (s)", "On")].to_numpy() / baseline_iterations
    )
    paired_metrics["Effective speedup"] = (
        paired_metrics["Off effective XC time / iteration (s)"]
        / paired_metrics["On effective XC time / iteration (s)"]
    )
    paired_metrics["Iteration-count factor"] = (
        paired[("SCF iterations", "On")].to_numpy() / baseline_iterations
    )

    direct_effective = (
        paired[("XC time (s)", "Off")].to_numpy()
        / paired[("XC time (s)", "On")].to_numpy()
    )
    if not np.allclose(
        paired_metrics["Effective speedup"], direct_effective, rtol=1e-12, atol=1e-12
    ):
        raise RuntimeError("Effective-speedup consistency check failed.")

    summary = (
        paired_metrics.groupby(["H2O count", "System"], observed=True)
        .agg(
            off_mean=("Off effective XC time / iteration (s)", "mean"),
            off_std=("Off effective XC time / iteration (s)", "std"),
            on_mean=("On effective XC time / iteration (s)", "mean"),
            on_std=("On effective XC time / iteration (s)", "std"),
            iteration_factor_mean=("Iteration-count factor", "mean"),
            effective_speedup_mean=("Effective speedup", "mean"),
        )
        .reset_index()
        .sort_values("H2O count")
    )

    return summary


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
            "axes.linewidth": 0.8,
            "axes.edgecolor": TEXT_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "legend.fontsize": 11,
            "text.color": TEXT_COLOR,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def plot_figure(summary: pd.DataFrame) -> plt.Figure:
    """Build a simple grouped bar chart of effective XC time per iteration."""
    publication_style()

    summary = summary.sort_values("H2O count").reset_index(drop=True)
    counts = summary["H2O count"].to_numpy()
    system_labels = [rf"$(\mathrm{{H_2O}})_{{{count}}}$" for count in counts]
    positions = np.arange(len(counts), dtype=float)
    width = 0.34
    fig, axis = plt.subplots(figsize=(7.2, 4.5), constrained_layout=True)

    off_bars = axis.bar(
        positions - width / 2,
        summary["off_mean"],
        width,
        yerr=summary["off_std"].fillna(0),
        color=BASELINE_COLOR,
        edgecolor="white",
        linewidth=0.7,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
        label="Dynamic precision off",
        zorder=3,
    )
    on_bars = axis.bar(
        positions + width / 2,
        summary["on_mean"],
        width,
        yerr=summary["on_std"].fillna(0),
        color=DYNAMIC_COLOR,
        edgecolor="white",
        linewidth=0.7,
        capsize=3,
        error_kw={"elinewidth": 0.9, "capthick": 0.9},
        label="Dynamic precision on",
        zorder=3,
    )

    upper = max(
        np.max(summary["off_mean"] + summary["off_std"].fillna(0)),
        np.max(summary["on_mean"] + summary["on_std"].fillna(0)),
    )
    axis.set_ylim(0, upper * 1.18)
    axis.set_xticks(positions, system_labels)
    axis.set_xlabel("Water-cluster size", fontsize=13, fontweight="bold")
    axis.set_ylabel(
        "Effective XC time per iteration (s)",
        fontsize=13,
        fontweight="bold",
    )
    axis.set_title(
        f"Dynamic-precision XC timings — {GPU_LABEL}",
        fontsize=14,
        fontweight="bold",
        pad=10,
    )
    axis.legend(loc="upper left", frameon=False)
    axis.grid(True, axis="y", color=GRID_COLOR, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(axis="x", length=0)
    for tick_label in axis.get_xticklabels():
        tick_label.set_fontweight("bold")

    def format_bar_value(value: float) -> str:
        return f"{value:.3f}" if value < 1.0 else f"{value:.2f}"

    for bars, errors in (
        (off_bars, summary["off_std"].fillna(0)),
        (on_bars, summary["on_std"].fillna(0)),
    ):
        for bar, error in zip(bars, errors):
            axis.annotate(
                format_bar_value(bar.get_height()),
                xy=(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + error,
                ),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10.5,
            )

    for position, speedup in zip(positions, summary["effective_speedup_mean"]):
        group_top = max(
            summary.loc[position, "off_mean"]
            + (summary.loc[position, "off_std"] if pd.notna(summary.loc[position, "off_std"]) else 0),
            summary.loc[position, "on_mean"]
            + (summary.loc[position, "on_std"] if pd.notna(summary.loc[position, "on_std"]) else 0),
        )
        axis.annotate(
            f"{speedup:.2f}×",
            xy=(position, group_top),
            xytext=(0, 18),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=12,
            color=EFFECTIVE_COLOR,
            fontweight="semibold",
        )

    fig.text(
        0.5,
        -0.02,
        (
            "Effective XC time/iteration = total XC time ÷ baseline "
            "(dynamic-off) SCF iterations; mean ± 1 SD over three paired runs."
        ),
        ha="center",
        va="top",
        fontsize=9,
        color="#4D5156",
    )

    return fig


def print_summary(summary: pd.DataFrame) -> None:
    """Print the plotted values for an auditable command-line record."""
    columns = [
        "System",
        "off_mean",
        "on_mean",
        "iteration_factor_mean",
        "effective_speedup_mean",
    ]
    print(
        summary[columns].to_string(
            index=False,
            formatters={
                "off_mean": "{:.4f}".format,
                "on_mean": "{:.4f}".format,
                "iteration_factor_mean": "{:.3f}".format,
                "effective_speedup_mean": "{:.3f}".format,
            },
        )
    )


def main() -> None:
    """Parse the workbook and write PNG/PDF figures."""
    args = parse_arguments()
    workbook_path = args.workbook.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_raw_data(workbook_path)
    summary = summarize(data)
    figure = plot_figure(summary)

    png_path = output_dir / f"{args.output_stem}.png"
    pdf_path = output_dir / f"{args.output_stem}.pdf"
    figure.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    print_summary(summary)
    print(f"\nSaved: {png_path}")
    print(f"Saved: {pdf_path}")

    plt.close(figure)


if __name__ == "__main__":
    main()
