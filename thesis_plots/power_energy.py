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



def plot_energy_by_power(agg, plotter, out_dir: Path = None):
    plotter.plot(agg, PlotSpec(
        kind="scatter",
        x="run_power_w",
        xlabel="Average Power (W)",
        y="energy_j",
        yscale="log",
        ylabel="Kernel Energy Consumption (J)",

        figsize=(6,4),

        title="Kernel Energy Consumption vs Average Power Draw by Algorithm and Variant",
        title_wrap=50,
        output= (out_dir / "energy/energy_by_power.png" if out_dir else "energy/energy_by_power.png"),

        series_by=["Algorithm", "variant"],
        
        hue_by="Algorithm",
        shade_by="variant",
        marker_by="variant",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
            "SPMV": "#eb7323", 
        },

        shade_values={
            "clvk": -0.40,
            "PoCL": -0.15,
            "OpenMP": 0.10,
            "Serial": 0.35,
        },

        marker_values={
            "Serial": "s",
            "OpenMP": "^",
            "PoCL": "D",
            "clvk": "o"
        },

        legend="outside_right",
        legend_fontsize=6,
    ))