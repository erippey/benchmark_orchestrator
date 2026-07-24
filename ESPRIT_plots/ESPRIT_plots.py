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


def normalize_frequency_mhz(values: pd.Series) -> pd.Series:
    """Normalize mixed MHz/kHz/Hz frequency columns into MHz.

    Handles values like:
      * 960          -> 960 MHz, Raspberry Pi v3d/arm columns
      * 1497600      -> 1497.6 MHz, Linux cpufreq-style kHz columns
      * 306000000    -> 306 MHz, Jetson GPU Hz columns
    """
    s = pd.to_numeric(values, errors="coerce").astype(float)
    return pd.Series(
        np.select(
            [s > 10_000_000, s > 10_000, s > 0],
            [s / 1_000_000.0, s / 1_000.0, s],
            default=np.nan,
        ),
        index=values.index,
        dtype=float,
    )


def first_positive_frequency_mhz(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    """Pick the first present, positive frequency column and normalize it to MHz."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for col in columns:
        if col not in df.columns:
            continue
        candidate = normalize_frequency_mhz(df[col])
        out = out.mask(out.isna() & candidate.notna() & (candidate > 0), candidate)
    return out


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



# Derived columns/metrics to add to BenchmarkData. Order matters here:
# operating_frequency_mhz depends on gpu_freq_mhz and cpu_freq_mhz.
DERIVED = [
    Metric(
        "cpu_freq_mhz",
        "CPU Frequency",
        "MHz",
        lambda df: first_positive_frequency_mhz(
            df,
            [
                "arm_freq",
                "arm_freq_min",
                "cpu_freq",
                "cpu_freq_min",
                "policy0_freq",
                "policy4_freq",
                "policy6_freq",
            ],
        ),
        higher_is_better=False,
    ),
    Metric(
        "percent_deadlines_missed",
        "Percent of Deadlines Missed",
        "%",
        lambda df: (df["deadlines_missed"] / df["invocations"])
    ),
    runtime_ms("average_execution_ms", "average_execution_ms", "Average Execution"),
    runtime_ms("maximum_start_lateness_ms", "maximum_start_lateness_ms", "Maximum Start Lateness"),
    runtime_power(),
    runtime_power("idle_power_w", "idle_power_w", "Idle Power"),
]


def main() -> None:
    csv_path = Path("../aggregated_csv/temp_esprit.csv")
    out_dir = Path("graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = BenchmarkData.from_csv(csv_path).with_metrics(DERIVED)

    # Use raw rows here. Once every bucket has enough repeats, you can replace
    # `plot_data` with a trimmed version that drops fastest/slowest runs per bucket.
    plot_data = data


    agg = plot_data.aggregate(
            ["Device", "cpu_freq_mhz", "Algorithm", "Threads"],
            ["average_execution_ms", "maximum_start_lateness_ms", "run_power_w", "percent_deadlines_missed"],
            aggregator="mean",
            include_std=True,
    )

    #make_algorithm_stats_table(agg, "CPU").to_csv(out_dir / "cpu_algorithm_stats.csv", index=False)



    dimensions = {
        "cpu_freq_mhz": Dimension("cpu_freq_mhz", "CPU Frequency", "MHz"),
        "Device": Dimension("Device", "Device"),
        "average_execution_ms": Dimension("average_execution_ms", "Average Execution Time"),
        "maximum_start_lateness_ms": Dimension("average_start_lateness_ms", "Average Start Lateness"),
        "run_power_w": Dimension("runtime_power_w", "Runtime Power"),
        "algorithm": Dimension("algorithm", "Algorithm"),
    }
    metrics = {m.name: m for m in DERIVED}
    plotter = Plotter(dimensions, metrics)


    plotter.plot(agg, PlotSpec(
        kind="line",
        x="cpu_freq_mhz",
        xlabel="CPU Frequency (MHz)",
        y="run_power_w",
        ylabel="Average Power Consumption (W)",

        figsize=(8,4),

        title="CPU Frequency vs Rasperry Pi 5 Average Power Draw",
        output= out_dir / "power_by_cpu_freq.png",

        series_by=["Algorithm", "Threads"],
        
        hue_by="Algorithm",
        shade_by="Threads",
        marker_by="Threads",

        base_colors={
            "ESPRIT 1":    "#4c78a8",
            "ESPRIT 2":    "#54a24b",
            "ESPRIT 3": "#9c6ade", 
        },

        shade_values={
            1: -0.15,
            4: 0.10,
        },

        marker_values={
            1: "s",
            4: "^",
        },

        legend="outside_right",
        legend_fontsize=8,
    ))



    plotter.plot(agg, PlotSpec(
        kind="line",
        x="cpu_freq_mhz",
        xlabel="CPU Frequency (MHz)",
        y="percent_deadlines_missed",
        ylabel="Percent of Deadlines Missed",

        figsize=(8,4),

        title="CPU Frequency vs Percent of Deadlines Missed",
        output= out_dir / "deadlines_missed_by_cpu_freq.png",

        series_by=["Algorithm", "Threads"],
        
        hue_by="Algorithm",
        shade_by="Threads",
        marker_by="Threads",

        base_colors={
            "ESPRIT 1":    "#4c78a8",
            "ESPRIT 2":    "#54a24b",
            "ESPRIT 3": "#9c6ade", 
        },

        shade_values={
            1: -0.15,
            4: 0.10,
        },

        marker_values={
            1: "s",
            4: "^",
        },

        legend="outside_right",
        legend_fontsize=8,
    ))



if __name__ == "__main__":
    main()
