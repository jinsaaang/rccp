#!/usr/bin/env python3
"""Generate analysis figures from included alpha=0.1 figure-value JSON files."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, ScalarFormatter


ROOT = Path(__file__).resolve().parents[1]
VALUE_DIR = ROOT / "data" / "figure_values"
OUT_DIR = ROOT / "outputs" / "figures"
TABLE_DIR = ROOT / "outputs"

DATASETS = ["air", "solar"]
DATASET_LABELS = {
    "air": "Air",
    "solar": "Solar",
    "wind": "Wind",
    "electricity": "Electricity",
}
METHODS = ["scp", "enbpi", "spic", "nextcp", "hopcpt", "rescp", "rccp"]
RAW_METHODS = ["conf_default", "enbpi", "spic", "nextcp", "hopcpt", "rescp", "rccp"]
RAW_TO_FIG_METHOD = {"conf_default": "scp", **{m: m for m in METHODS if m != "scp"}}
SEEDS = {0, 1, 2, 3, 4}
METHOD_LABELS = {
    "scp": "SCP",
    "enbpi": "EnbPI",
    "spic": "SPCI",
    "nextcp": "NexCP",
    "hopcpt": "HopCPT",
    "rescp": "ResCP",
    "rccp": "RCCP (Ours)",
}
COLORS = {
    "scp": "#6b7280",
    "enbpi": "#4e79a7",
    "spic": "#f28e2b",
    "nextcp": "#59a14f",
    "hopcpt": "#b07aa1",
    "rescp": "#9c755f",
    "rccp": "#d62728",
}
TRADEOFF_COLORS = {
    "scp": "#828A90",
    "enbpi": "#D5C8E8",
    "spic": "#C8CFEE",
    "nextcp": "#8FBFFF",
    "hopcpt": "#F6830D",
    "rescp": "#D53A3A",
    "rccp": "#0C6657",
}
TRADEOFF_MARKERS = {
    "scp": "o",
    "enbpi": "s",
    "spic": "^",
    "nextcp": "v",
    "hopcpt": "P",
    "rescp": "X",
    "rccp": "D",
}
TRADEOFF_PLOT_ORDER = ["rccp", "hopcpt", "nextcp", "enbpi", "rescp", "scp", "spic"]
TRADEOFF_LEGEND_ORDER = ["hopcpt", "nextcp", "enbpi", "rescp", "scp", "spic", "rccp"]
TRADEOFF_MARKER_SIZE = 8.5 * math.sqrt(3.0) * 1.5


def load_json(name: str) -> dict:
    with (VALUE_DIR / name).open() as f:
        return json.load(f)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 14,
            "axes.titlesize": 17,
            "axes.labelsize": 15,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def savefig(fig: plt.Figure, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    if path.suffix.lower() == ".pdf":
        fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {path}")


def rows_by_dataset(data: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in data:
        if row.get("dataset") in DATASETS and row.get("method") in METHODS:
            grouped[row["dataset"]].append(row)
    return grouped


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def make_winkler_decomposition() -> None:
    data = load_json("winkler_decomposition.json")["data"]
    grouped = rows_by_dataset(data)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.3), sharey=False)
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        rows = {row["method"]: row for row in grouped[dataset]}
        x = list(range(len(METHODS)))
        widths = [rows[m]["width"] for m in METHODS]
        penalties = [rows[m]["miss_penalty"] for m in METHODS]
        ax.bar(x, widths, color="#9ecae1", edgecolor="white", linewidth=0.6, label="Width")
        ax.bar(
            x,
            penalties,
            bottom=widths,
            color="#fdae6b",
            edgecolor="white",
            linewidth=0.6,
            label="Miss penalty",
        )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right")
        ax.set_ylabel("Winkler score")
        ax.grid(axis="y", color="#d1d5db", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)
    savefig(fig, "winkler_decomposition.pdf")


def make_tradeoff_summary() -> None:
    rows = load_json("tradeoff_summary.json")["data"]
    points = {
        row["method"]: {
            "abs_delta": float(row["abs_delta_cov_pct"]),
            "winkler": float(row["winkler"]),
            "time": float(row["calibration_time_s"]),
        }
        for row in rows
    }

    fig, axes = plt.subplots(1, 2, figsize=(17.6, 5.8), sharey=True)
    plot_tradeoff_axis(axes[0], points, "abs_delta", r"$|\Delta$Cov$|$ (%)")
    plot_tradeoff_axis(axes[1], points, "time", "Calibration time (s)", xscale="log")
    axes[0].set_ylabel("Winkler")
    axes[1].tick_params(axis="y", labelleft=False)
    fig.legend(
        handles=tradeoff_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(TRADEOFF_LEGEND_ORDER),
        frameon=True,
        fancybox=False,
        edgecolor="#9CA3AF",
        facecolor="white",
        framealpha=1.0,
        columnspacing=1.0,
        handletextpad=0.3,
    )
    fig.subplots_adjust(top=0.80, bottom=0.15, left=0.07, right=0.985, wspace=0.08)
    savefig(fig, "tradeoff_winkler_delta_time.pdf")


def plot_tradeoff_axis(
    ax: plt.Axes,
    points: dict[str, dict[str, float]],
    x_key: str,
    xlabel: str,
    *,
    xscale: str | None = None,
) -> None:
    for method in TRADEOFF_PLOT_ORDER:
        ax.plot(
            points[method][x_key],
            points[method]["winkler"],
            marker=TRADEOFF_MARKERS[method],
            color=TRADEOFF_COLORS[method],
            markerfacecolor=TRADEOFF_COLORS[method],
            markeredgecolor=TRADEOFF_COLORS[method],
            linestyle="None",
            markersize=TRADEOFF_MARKER_SIZE,
            alpha=0.95,
            zorder=3,
        )
    x_values = [points[method][x_key] for method in METHODS]
    y_values = [points[method]["winkler"] for method in METHODS]
    if xscale:
        ax.set_xscale(xscale)
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(min(x_values) / 1.7, max(x_values) * 1.7)
    else:
        padding = (max(x_values) - min(x_values)) * 0.22
        ax.set_xlim(min(x_values) - padding, max(x_values) + padding)
    padding_y = (max(y_values) - min(y_values)) * 0.18
    ax.set_ylim(min(y_values) - padding_y, max(y_values) + padding_y)
    ax.set_xlabel(xlabel)
    ax.grid(True, which="major", color="#D5D9E0", linewidth=0.7, alpha=0.75)
    ax.set_axisbelow(True)


def tradeoff_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker=TRADEOFF_MARKERS[method],
            color=TRADEOFF_COLORS[method],
            markerfacecolor=TRADEOFF_COLORS[method],
            markeredgecolor=TRADEOFF_COLORS[method],
            linestyle="None",
            markersize=12,
            label=METHOD_LABELS[method],
        )
        for method in TRADEOFF_LEGEND_ORDER
    ]


def make_error_deciles() -> None:
    width_rows = load_json("error_decile_width_all_methods.json")["data"]
    cov_rows = load_json("error_decile_coverage_all_methods.json")["data"]
    widths: dict[tuple[str, str], list[dict]] = defaultdict(list)
    covs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in width_rows:
        if row["dataset"] in DATASETS and row["method"] in METHODS:
            widths[(row["dataset"], row["method"])].append(row)
    for row in cov_rows:
        if row["dataset"] in DATASETS and row["method"] in METHODS:
            covs[(row["dataset"], row["method"])].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(15.8, 9.2), sharex=True)
    for row_idx, dataset in enumerate(DATASETS):
        ax_w = axes[row_idx, 0]
        ax_c = axes[row_idx, 1]
        for method in METHODS:
            wrows = sorted(widths[(dataset, method)], key=lambda r: r["error_decile"])
            crows = sorted(covs[(dataset, method)], key=lambda r: r["error_decile"])
            deciles = [r["error_decile"] for r in wrows]
            vals = [r["width"] for r in wrows]
            if method == "scp" and vals:
                vals = [sum(vals) / len(vals)] * len(vals)
            ax_w.plot(
                deciles,
                vals,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
            ax_c.plot(
                [r["error_decile"] for r in crows],
                [r["coverage"] for r in crows],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax_w.set_ylabel(f"{DATASET_LABELS[dataset]}\nPI-Width")
        ax_c.set_ylabel("Coverage")
        ax_c.axhline(0.9, color="#111827", linestyle="--", linewidth=1.2, alpha=0.8)
        for ax in (ax_w, ax_c):
            ax.set_xticks(range(1, 11))
            ax.grid(color="#d1d5db", linewidth=0.7, alpha=0.8)
            ax.set_axisbelow(True)
    axes[0, 0].set_title("Width by realized-error decile")
    axes[0, 1].set_title("Coverage by realized-error decile")
    axes[-1, 0].set_xlabel("Realized-error decile")
    axes[-1, 1].set_xlabel("Realized-error decile")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=1.8, w_pad=2.2)
    savefig(fig, "error_decile_width_coverage.pdf")


def make_miss_distance_cdf() -> None:
    data = load_json("miss_distance_cdf_all_methods.json")["data"]
    grouped = rows_by_dataset(data)
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.3), sharey=True)
    axes = axes.ravel()
    for ax, dataset in zip(axes, DATASETS):
        rows = {row["method"]: row for row in grouped[dataset]}
        for method in METHODS:
            row = rows[method]
            ax.plot(
                row["x_grid"],
                row["cdf_miss_events_only"],
                linewidth=2.2,
                color=COLORS[method],
                label=METHOD_LABELS[method],
            )
        ax.set_title(DATASET_LABELS[dataset])
        ax.set_xlabel("Miss distance")
        ax.set_ylabel("CDF over misses")
        ax.set_ylim(0, 1.02)
        ax.grid(color="#d1d5db", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.0)
    savefig(fig, "miss_distance_cdf.pdf")


def main() -> None:
    setup_style()
    make_tradeoff_summary()
    make_winkler_decomposition()
    make_error_deciles()
    make_miss_distance_cdf()


if __name__ == "__main__":
    main()
