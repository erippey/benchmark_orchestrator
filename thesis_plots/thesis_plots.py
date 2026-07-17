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

from bar_charts import plot_bar_charts
from power_energy import plot_energy_by_power

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





def implementation_component(df: pd.DataFrame) -> pd.Series:
    """Return which component should define the operating frequency: GPU or CPU."""
    backend = df.get("Backend", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    platform = df.get("Platform", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()

    cpu_impl = (
        backend.str.contains("fftw|openmp|cpu|serial|pocl", regex=True, na=False)
        | platform.str.contains("openmp|fftw|cpu|serial|pocl", regex=True, na=False)
    )
    gpu_impl = (
        backend.str.contains("clfft|cufft|cuda|opencl|gpu|vulkan|clvk", regex=True, na=False)
        | platform.str.contains("cuda|opencl|vulkan|clvk|gpu", regex=True, na=False)
    )

    # CPU wins ties so an FFTW row that happens to carry stale GPU metadata
    # does not get assigned the GPU frequency.
    return pd.Series(
        np.select([cpu_impl, gpu_impl], ["CPU", "GPU"], default="unknown"),
        index=df.index,
    )

def variant(df: pd.DataFrame) -> pd.Series:
    threads = df.get("Threads", pd.Series("", index=df.index)).fillna(0).astype(int)
    platform = df.get("Platform", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    device = df.get("Device", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()

    clvk_opi =  (
        platform.str.contains("clvk", regex=True, na=False)
        & device.str.contains("orangepi5ultra", regex=True, na=False)
    )

    clvk_rpi = (
        platform.str.contains("opencl|clvk", regex=True, na=False)
        & device.str.contains("raspberrypicomputemodule5", regex=True, na=False)
    )

    clvk_nano = (
        platform.str.contains("opencl|clvk", regex=True, na=False)
        & device.str.contains("jetsonorinnanosuper", regex=True, na=False)
    )

    opencl_opi =  (
        platform.str.contains("opencl", regex=True, na=False)
        & device.str.contains("orangepi5ultra", regex=True, na=False)
    )

    pocl = (
        platform.str.contains("pocl", regex=True, na=False)
    )

    cuda = (
        platform.str.contains("cuda", regex=True, na=False)
    )

    serial = (
        platform.str.contains("serial", regex=True, na=False)
        | (platform.str.contains("serial" , regex=True, na=False) & threads > 1)
    )

    openmp = (
        platform.str.contains("openmp", regex=True, na=False)
    )


    return pd.Series(
        np.select([clvk_opi, clvk_rpi, clvk_nano, opencl_opi, pocl, cuda, serial, openmp], 
        ["clvk\N{RIGHTWARDS ARROW}OPI", "clvk\N{RIGHTWARDS ARROW}RPI", 
         "clvk\N{RIGHTWARDS ARROW}ONS", "OpenCL\N{RIGHTWARDS ARROW}OPI", 
         "PoCL\N{RIGHTWARDS ARROW}RPI", "CUDA\N{RIGHTWARDS ARROW}ONS", "Serial\N{RIGHTWARDS ARROW}RPI", 
         "OpenMP\N{RIGHTWARDS ARROW}RPI"], default="unknown"),
        index=df.index,
    )



def pull_important_frequency(df: pd.DataFrame) -> pd.Series:
    """Pick the frequency that matches the implementation component.

    GPU implementations use gpu_freq_mhz. CPU implementations use cpu_freq_mhz.
    Missing values fall back to the other component so partially populated CSVs
    still plot instead of silently disappearing.
    """
    component = implementation_component(df)
    gpu = pd.to_numeric(df.get("gpu_freq_mhz", pd.Series(np.nan, index=df.index)), errors="coerce")
    cpu = pd.to_numeric(df.get("cpu_freq_mhz", pd.Series(np.nan, index=df.index)), errors="coerce")

    out = pd.Series(np.nan, index=df.index, dtype=float)
    out = out.mask(component.eq("GPU"), gpu)
    out = out.mask(component.eq("CPU"), cpu)

    # Fallbacks for rows where the preferred component is missing.
    out = out.mask(out.isna() & gpu.notna(), gpu)
    out = out.mask(out.isna() & cpu.notna(), cpu)
    return out

def get_test_name(df: pd.DataFrame) -> pd.Series:
    return df["run_dir"].str.extract(
        r"bench_logs\\[^\\]+\\([^\\]+)"
    )[0]



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
        "gpu_freq_mhz",
        "GPU Frequency",
        "MHz",
        lambda df: first_positive_frequency_mhz(
            df,
            ["v3d_freq_min", "v3d_freq", "gpu_freq_min", "gpu_freq"],
        ),
        higher_is_better=False,
    ),
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
    Metric("test_name", "Test Name", "", get_test_name),
    Metric("frequency_component", "Frequency Component", "", implementation_component),
    Metric("variant", "Variant", "", variant),
    Metric("operating_frequency_mhz", "Operating Frequency", "MHz", pull_important_frequency),
    runtime_ms("Kernel Runtime", "kernel_runtime", "Kernel Runtime"),
    runtime_ms("Region of Interest", "roi_runtime", "Region of Interest"),
    runtime_power(),
    runtime_power("idle_power_w", "idle_power_w", "Idle Power"),
    energy_j("Kernel Runtime", "run_power_w"),
    edp("Kernel Runtime", "energy_j"),
]


def main() -> None:
    csv_path = Path("../aggregated_csv/thesis_results.csv")
    out_dir = Path("graphs/thesis/")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = BenchmarkData.from_csv(csv_path).with_metrics(DERIVED)

    # Use raw rows here. Once every bucket has enough repeats, you can replace
    # `plot_data` with a trimmed version that drops fastest/slowest runs per bucket.
    plot_data = data


    # First aggregate repeated benchmark runs at each operating frequency.
    dev_data = {}

    for device in ["GPU", "CPU"]:
        dev_data[device] = plot_data.where(frequency_component=device).aggregate(
            ["Device", "frequency_component", "operating_frequency_mhz", "Algorithm", "Threads", "variant", "test_name"],
            ["kernel_runtime", "roi_runtime", "run_power_w", "energy_j", "edp_j_s"],
            aggregator="mean",
            include_std=True,
        )

        dev_data[device] = add_relative_to_group_x(
            dev_data[device],
            "kernel_runtime",
            ["test_name"],
            "operating_frequency_mhz",
            "max",
            higher_is_better=False,
            out_col="rel_kernel_runtime"
        )

        dev_data[device] = add_relative_to_group_x(
            dev_data[device],
            "energy_j",
            ["test_name"],
            "operating_frequency_mhz",
            "max",
            higher_is_better=True,
            out_col="rel_energy_j"
        )


    agg = plot_data.aggregate(
            ["Device", "frequency_component", "operating_frequency_mhz", "Algorithm", "Threads", "variant"],
            ["kernel_runtime", "roi_runtime", "run_power_w", "energy_j", "edp_j_s"],
            aggregator="mean",
            include_std=True,
    )

    make_algorithm_stats_table(agg, "CPU").to_csv(out_dir / "cpu_algorithm_stats.csv", index=False)
    make_algorithm_stats_table(agg, "GPU").to_csv(out_dir / "gpu_algorithm_stats.csv", index=False)



    dimensions = {
        "gpu_freq_mhz": Dimension("gpu_freq_mhz", "GPU Frequency", "MHz"),
        "cpu_freq_mhz": Dimension("cpu_freq_mhz", "CPU Frequency", "MHz"),
        "operating_frequency_mhz": Dimension("operating_frequency_mhz", "Operating Frequency", "MHz"),
        "frequency_component": Dimension("frequency_component", "Frequency Component"),
        "Device": Dimension("Device", "Device"),
        "kernel_runtime": Dimension("kernel_runtime", "Kernel Runtime"),
        "roi_runtime": Dimension("roi_runtime", "Region of Interest"),
        "run_power_w": Dimension("runtime_power_w", "Runtime Power"),
        "energy_j": Dimension("energy_j", "Energy"),
        "test_name": Dimension("test_name", "Test Name"),
        "algorithm": Dimension("algorithm", "Algorithm"),
        "variant": Dimension("variant", "Variant"),
    }
    metrics = {m.name: m for m in DERIVED}
    plotter = Plotter(dimensions, metrics)

    plot_energy_by_power(agg, plotter, out_dir)


    plotter.plot(agg, PlotSpec(
        kind="scatter",
        x="run_power_w",
        xlabel="Average Power (W)",
        y="kernel_runtime",
        ylabel="Kernel Runtime (ms)",
        yscale="log",

        figsize=(8,4),

        title="Kernel Runtime Across Implementations vs Average Power Draw",
        output= out_dir / "runtime_by_power.png",

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
        legend_fontsize=8,
    ))


    plotter.plot_panels(
        [
            (agg.loc[agg["Algorithm"].eq("BFS")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel=None,
                y="kernel_runtime",
                ylabel=None,
                yscale="log",

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                    
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("BFS")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel=None,
                y="energy_j",
                yscale="log",
                ylabel=None,

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("FFT")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel=None,
                y="kernel_runtime",
                ylabel=None,
                yscale="log",

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                    
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("FFT")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel=None,
                y="energy_j",
                yscale="log",
                ylabel=None,

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("KMeans")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel=None,
                y="kernel_runtime",
                ylabel=None,
                yscale="log",

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                    
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("KMeans")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel=None,
                y="energy_j",
                yscale="log",
                ylabel=None,

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("SRAD")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel=None,
                y="kernel_runtime",
                ylabel=None,
                yscale="log",

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                    
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("SRAD")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel=None,
                y="energy_j",
                yscale="log",
                ylabel=None,

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("SPMV")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel="Average Power (W)",
                y="kernel_runtime",
                ylabel="Kernel Runtime (ms)",
                yscale="log",

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                    
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
            (agg.loc[agg["Algorithm"].eq("SPMV")].copy(), PlotSpec(
                kind="scatter",
                x="run_power_w",
                xlabel="Average Power (W)",
                y="energy_j",
                yscale="log",
                ylabel="Kernel Energy Consumption (J)",

                figsize=(6,4),

                title="",
                output="NaN",

                series_by=["Algorithm", "variant", "frequency_component"],
                
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

                legend="top_right",
                legend_fontsize=8,
                legend_style="values",
            )),
        ],
        output= out_dir / "runtime_energy_by_power.png",
        figsize=(9, 10),
        sharex=False,
        sharey=False,
        legend="none",
#        legend_label_style="values",
        ncols=2,
        legend_ncol=1,
        dpi=400,

        column_gap = 0.06,

        column_xlabels = [
            "Average Power (W)",
            "Average Power (W)",
        ],
        column_xlabel_fontsize=16,

        column_ylabels = [
            "Kernel Runtime (ms)",
            "Kernel Energy Consumption (J)",
        ],
        column_ylabel_fontsize=16,

        column_ylabel_pad=0.065,
        column_xlabel_pad=0.035,

    )


    plot_bar_charts(agg, plotter, out_dir)

    plotter.plot_panels(
        [
            (dev_data["GPU"], PlotSpec(
                kind="line",
                x="operating_frequency_mhz",
                xlabel="GPU Frequency (MHz)",
                y="run_power_w",
                ylabel="Average Power Draw (W)",

                figsize=(5,4),

                title="Average Power Consumtion vs GPU Frequency",
                title_fontsize=10,
                output="graphs/HPEC/opencl_power_by_gpu_freq.png",

                series_by=["Algorithm", "variant"],

                highlight_lowest_by=["Algorithm"],

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

                legend="none",
                legend_fontsize=8,
            )),
            (dev_data["CPU"], PlotSpec(
                kind="line",
                x="operating_frequency_mhz",
                xlabel="CPU Frequency (MHz)",
                y="run_power_w",
                ylabel="Average Power Draw (W)",

                figsize=(5,4),

                title="Average Power Consumtion vs CPU Frequency",
                title_fontsize=10,
                output="graphs/HPEC/openmp_power_by_cpu_freq.png",

                series_by=["Algorithm", "variant"],

                highlight_lowest_by=["Algorithm", "variant"],
                highlight_lowest_label="Lowest Power",

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

                legend="none",
                legend_fontsize=8,
            ))
        ],
        output= out_dir / "power_by_frequency_cpu_gpu.png",
        figsize=(10, 4),
        sharey=True,
        legend="right",
        legend_ncol=4,  
        ncols=2,
        dpi=400,

        legend_style="values",

    )

    plotter.plot_panels(
        [
            (dev_data["CPU"].loc[dev_data["CPU"]["variant"].eq("Serial")].copy(), PlotSpec(
                kind="line",
                x="operating_frequency_mhz",
                xlabel="CPU Frequency (MHz)",
                y="run_power_w",
                ylabel=None,

                figsize=(5,4),

                title="Serial\N{RIGHTWARDS ARROW}CPU",
                title_fontsize=14,
                output="graphs/HPEC/openmp_power_by_cpu_freq.png",

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

                legend="top_left",
                legend_style="values",
                legend_fontsize=8,
            )),
            (dev_data["CPU"].loc[dev_data["CPU"]["variant"].eq("OpenMP")].copy(), PlotSpec(
                kind="line",
                x="operating_frequency_mhz",
                xlabel="CPU Frequency (MHz)",
                y="run_power_w",
                ylabel=None,

                figsize=(5,4),

                title="OpenMP\N{RIGHTWARDS ARROW}CPU",
                title_fontsize=14,
                output="graphs/HPEC/openmp_power_by_cpu_freq.png",

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

                legend="top_left",
                legend_style="values",
                legend_fontsize=8,
            )),
            (dev_data["CPU"].loc[dev_data["CPU"]["variant"].eq("PoCL")].copy(), PlotSpec(
                kind="line",
                x="operating_frequency_mhz",
                xlabel="CPU Frequency (MHz)",
                y="run_power_w",
                ylabel=None,

                figsize=(5,4),

                title="PoCL\N{RIGHTWARDS ARROW}CPU",
                title_fontsize=14,
                output="graphs/HPEC/openmp_power_by_cpu_freq.png",

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

                legend="top_left",
                legend_style="values",
                legend_fontsize=8,
            )),
            (dev_data["GPU"], PlotSpec(
                kind="line",
                x="operating_frequency_mhz",
                xlabel="GPU Frequency (MHz)",
                y="run_power_w",
                ylabel=None,

                figsize=(5,4),

                title="clvk\N{RIGHTWARDS ARROW}GPU",
                title_fontsize=14,
                output="graphs/HPEC/opencl_power_by_gpu_freq.png",

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

                legend="top_left",
                legend_style="values",
                legend_fontsize=8,
            )),
        ],
        output= out_dir / "power_by_frequency_cpu_gpu2.png",
        figsize=(8, 6),
        sharey=True,
        legend="none",
        legend_ncol=4,  
        ncols=2,
        dpi=400,

        legend_style="values",

        shared_ylabel="Average Power (W)",
        shared_ylabel_fontsize=14,

    )



    plotter.plot(dev_data["GPU"], PlotSpec(
        kind="line",
        x="operating_frequency_mhz",
        xlabel="GPU Frequency (MHz)",
        y="rel_kernel_runtime",
        ylabel="Relative Performance Compared to 960 MHz",

        figsize=(8,4),

        title="Relative clvk Kernel Performance vs GPU Frequency",
        output= out_dir / "opencl_runtime_by_gpu_freq.png",

        series_by=["Algorithm", "variant", "Device"],

        highlight_highest_by=["Algorithm"],
        highlight_highest_label="Best Performance",

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
        legend_fontsize=8,
    ))

    plotter.plot(dev_data["GPU"], PlotSpec(
        kind="line",
        x="operating_frequency_mhz",
        xlabel="GPU Frequency (MHz)",
        y="kernel_runtime",
        ylabel="Kernel-Region Runtime (ms)",
        yscale="log",

        figsize=(8,8),

        title="Kernel-Region Runtime vs GPU Frequency for All GPU Tests",
        output= out_dir / "all_gpu_freq_v_runtime.png",

        series_by=["Algorithm", "variant"],

        #highlight_highest_by=["Algorithm"],
        #highlight_highest_label="Best Performance",

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

        marker_values = {
            "Serial\u2192RPI": "s",
            "OpenMP\u2192RPI": "^",
            "PoCL\u2192RPI": "D",

            "clvk\u2192RPI": "o",
            "clvk\u2192OPI": "P",
            "clvk\u2192ONS": "X",

            "OpenCL\u2192OPI": "v",
            "CUDA\u2192ONS": "*",
        },

        legend="outside_right",
        legend_fontsize=8,
    ))

    plotter.plot(dev_data["GPU"], PlotSpec(
        kind="line",
        x="operating_frequency_mhz",
        xlabel="GPU Frequency (MHz)",
        y="run_power_w",
        ylabel="Average Power Draw (W)",

        figsize=(5,4),

        title="Average Power Consumtion vs GPU Frequency",
        output= out_dir / "opencl_power_by_gpu_freq.png",

        series_by=["Algorithm", "variant", "Device"],

        highlight_lowest_by=["Algorithm"],
        highlight_lowest_label="Lowest Power",

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

        legend="none",
        legend_fontsize=8,
    ))

    plotter.plot(dev_data["GPU"], PlotSpec(
        kind="line",
        x="operating_frequency_mhz",
        xlabel="GPU Frequency (MHz)",
        y="energy_j",
        ylabel="Relative (%) of Max Evergy Consumption",
        yscale="log",

        figsize=(8,4),
        
        title="Relative clvk Energy-to-Solution vs GPU Frequency",
        output= out_dir / "opencl_energy_j_by_gpu_freq.png",

        series_by=["Algorithm", "variant", "Device"],

        highlight_lowest_by=["Algorithm"],
        highlight_lowest_label="Lowest Energy Consumption",

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
        legend_fontsize=8,
    ))
    

    plotter.plot(dev_data["CPU"], PlotSpec(
        kind="line",
        x="operating_frequency_mhz",
        xlabel="CPU Frequency (MHz)",
        y="rel_kernel_runtime",
        ylabel="Relative Performance Compared to 2400 MHz",

        figsize=(7,5),

        title="Relative CPU Kernel Performance vs CPU Frequency",
        output= out_dir / "openmp_runtime_by_cpu_freq.png",

        series_by=["Algorithm", "variant", "Device"],

        highlight_highest_by=["Algorithm", "variant"],
        highlight_highest_label="Best Performance/Efficiency",

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

        legend="none",
        legend_fontsize=8,
    ))


    plotter.plot(dev_data["CPU"], PlotSpec(
        kind="line",
        x="operating_frequency_mhz",
        xlabel="CPU Frequency (MHz)",
        y="run_power_w",
        ylabel="Average Power Draw (W)",

        figsize=(5,4),

        title="Average Power Consumtion vs CPU Frequency",
        output= out_dir / "openmp_power_by_cpu_freq.png",

        series_by=["Algorithm", "variant", "Device"],

        highlight_lowest_by=["Algorithm", "variant"],
        highlight_lowest_label="Lowest Power",

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

        legend="none",
        legend_fontsize=8,
    ))


    plotter.plot(dev_data["CPU"], PlotSpec(
        kind="line",
        x="operating_frequency_mhz",
        xlabel="CPU Frequency (MHz)",
        y="rel_energy_j",
        ylabel="Relative Energy (J) Compared to 2400 MHz",

        figsize=(7,5),
        
        title="Relative CPU Energy-to-Solution vs CPU Frequency",
        output= out_dir / "openmp_energy_j_by_cpu_freq.png",

        series_by=["Algorithm", "variant", "Device"],

        highlight_lowest_by=["Algorithm", "variant"],
        highlight_lowest_label="Lowest Energy Consumption",

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

        legend="none",
        legend_fontsize=8,
    ))

    agg.to_csv(out_dir / "thesis_cumulative.csv")




if __name__ == "__main__":
    main()
