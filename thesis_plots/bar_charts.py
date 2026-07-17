from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from graph_tool_v2_panels import (
    BenchmarkData,
    Dimension,
    Metric,
    PlotSpec,
    Plotter,
    TrimSpec,
    make_algorithm_stats_table,
    runtime_ms,
    runtime_power,
    energy_j,
    edp,
    add_relative_to_group_best,
    add_relative_to_group_x
)

def select_best_per_group(
    df: pd.DataFrame,
    group_by: Sequence[str],
    value_col: str,
    *,
    mode: Literal["max", "min"] = "max",
) -> pd.DataFrame:
    """Keep one row per group, chosen by max/min value_col.

    Use this after aggregating by operating_frequency_mhz. For this plot, it
    removes the one-bar-per-frequency explosion and leaves the best frequency
    point for each (Banks, algorithm_family, dev_backend) combination.
    """
    missing = [c for c in [*group_by, value_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Cannot select best rows; missing columns: {missing}")
    if mode not in {"max", "min"}:
        raise ValueError("mode must be 'max' or 'min'")

    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])
    if work.empty:
        return work.reset_index(drop=True)

    selected_indices: list[int] = []
    for _, group in work.groupby(list(group_by), dropna=False, sort=False):
        col_of_interest = group[value_col]
        idx = col_of_interest.idxmax() if mode == "max" else col_of_interest.idxmin()
        selected_indices.append(idx)

    return work.loc[selected_indices].reset_index(drop=True)


def build_plot_spec(agg, algorithm = None):
    if algorithm:
        data = agg.loc[agg["Algorithm"].eq(algorithm)].copy()
    else:
        data = agg

    
    bar_df = select_best_per_group(
        data,
        group_by=["Algorithm", "variant"],
        value_col="energy_j",
        mode="min",
    )

    return bar_df.copy()






def plot_bar_charts(agg, plotter, out_dir: Path = None):

    bar_df = build_plot_spec(agg)

    plotter.plot(bar_df, PlotSpec(
        kind="bar",
        x="variant",
        xlabel=None,
        y="energy_j",
        ylabel="Kernel Energy Consumption (J)",
        yscale="log",
        ybase=2,

        bar_group_by="Algorithm",
        bar_subgroup_by="frequency_component",

        bar_show_subgroup_labels=False,
        bar_show_group_labels=True,

        bar_group_gap=0.8,
        bar_subgroup_gap=0.5,
        bar_width=0.72,

        bar_value_fontsize=12,
        bar_subgroup_label_fontsize=10,

        # You can probably reduce this now because one label row is gone.
        bar_multilevel_bottom=0.30,

        category_orders={
            "Algorithm": ["BFS", "FFT", "KMeans", "SRAD", "SPMV"],
            "frequency_component": ["CPU", "GPU"],
            "variant": ["clvk\N{RIGHTWARDS ARROW}RPI",
                "clvk\N{RIGHTWARDS ARROW}OPI", 
                "OpenCL\N{RIGHTWARDS ARROW}OPI",
                "clvk\N{RIGHTWARDS ARROW}ONS",  
                "CUDA\N{RIGHTWARDS ARROW}ONS",
                "Serial\N{RIGHTWARDS ARROW}RPI", 
                "OpenMP\N{RIGHTWARDS ARROW}RPI",
                "PoCL\N{RIGHTWARDS ARROW}RPI"],
        },

        figsize=(50, 10),
        
        title="",
        output= (out_dir / "energy/energy_by_algorithm_variant.png" if out_dir else "energy/energy_by_algorithm_variant.png"),

        hue_by="Algorithm",
        shade_by="variant",
        pattern_by="variant",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
            "SPMV":   "#eb7323",
        },

        shade_values = {
            "clvk\u2192OPI":      -0.45,  # dark
            "OpenCL\u2192OPI":     0.20,  # light-ish, separated from clvk OPI

            "clvk\u2192RPI":      -0.25,  # medium-dark
            "PoCL\u2192RPI":       0.05,  # near neutral
            "Serial\u2192RPI":     0.40,  # very light
            "OpenMP\u2192RPI":    -0.55,  # very dark, far from PoCL

            "clvk\u2192ONS":      0.30,  # light
            "CUDA\u2192ONS":     -0.70,  # darkest, strongly separated
        },

        bar_edgecolor="black",
        bar_linewidth=0.4,

        legend="none",
        legend_fontsize=10,
    ))

    panel_plot_spec = PlotSpec(
        kind="bar",
        x="variant",
        xlabel=None,
        y="energy_j",
        ylabel="Kernel Energy Consumption (J)",
        yscale="log",

        bar_group_by="Algorithm",
        bar_subgroup_by="frequency_component",

        bar_show_subgroup_labels=False,
        bar_show_group_labels=True,

        bar_group_gap=0.8,
        bar_subgroup_gap=0.5,
        bar_width=0.72,

        bar_value_fontsize=12,
        bar_subgroup_label_fontsize=10,

        # You can probably reduce this now because one label row is gone.
        bar_multilevel_bottom=0.30,

        category_orders={
            "frequency_component": ["CPU", "GPU"],
            "variant": ["clvk\N{RIGHTWARDS ARROW}RPI",
                "clvk\N{RIGHTWARDS ARROW}OPI", 
                "OpenCL\N{RIGHTWARDS ARROW}OPI",
                "clvk\N{RIGHTWARDS ARROW}ONS",  
                "CUDA\N{RIGHTWARDS ARROW}ONS",
                "Serial\N{RIGHTWARDS ARROW}RPI", 
                "OpenMP\N{RIGHTWARDS ARROW}RPI",
                "PoCL\N{RIGHTWARDS ARROW}RPI"],
        },

        figsize=(10, 10),
        
        title="",
        output="",

        hue_by="Algorithm",
        shade_by="variant",
        pattern_by="variant",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
            "SPMV":   "#eb7323",
        },

        shade_values = {
            "clvk\u2192OPI":      -0.45,  # dark
            "OpenCL\u2192OPI":     0.20,  # light-ish, separated from clvk OPI

            "clvk\u2192RPI":      -0.25,  # medium-dark
            "PoCL\u2192RPI":       0.05,  # near neutral
            "Serial\u2192RPI":     0.40,  # very light
            "OpenMP\u2192RPI":    -0.55,  # very dark, far from PoCL

            "clvk\u2192ONS":      0.30,  # light
            "CUDA\u2192ONS":     -0.70,  # darkest, strongly separated
        },

        bar_edgecolor="black",
        bar_linewidth=0.4,

        legend="none",
        legend_fontsize=10,
    )

    plotter.plot_panels(
        [
            (build_plot_spec(agg, "BFS"), panel_plot_spec),
            (build_plot_spec(agg, "FFT"), panel_plot_spec),
            (build_plot_spec(agg, "KMeans"), panel_plot_spec),
            (build_plot_spec(agg, "SPMV"), panel_plot_spec),
            (build_plot_spec(agg, "SRAD"), panel_plot_spec),
        ], 
        output = (out_dir / "energy/paneled_energy_per_algorithm" if out_dir else "energy/paneled_energy_per_algorithm"),
        figsize=(10,30),
        sharey=False,
        legend="none",
        ncols=1,
        dpi=400,
    )