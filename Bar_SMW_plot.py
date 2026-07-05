from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import pandas as pd

from graph_tool_v2_series_bar import (
    BenchmarkData,
    Dimension,
    Metric,
    PlotSpec,
    Plotter,
    TrimSpec,
)

# Operation counts for one complete benchmarked run.
# These are NOT FLOPS yet; FLOPS = operations / runtime_seconds.
# Note: I have used UPF-OS and UPOLS interchangeably in notes/scripts.
# The plot labels use UPF-OS, but the string recognizer accepts both.
OP_COUNTS = {
    # (Algorithm, Channels, Banks)
    ("UPF-OS", 16, 16) : 2.44907e10,
    ("OLA", 16, 16) : 2.83961e10,
    ("UPF-OS", 16, 8) : 1.29654e10,
    ("OLA", 16, 8) : 1.50287e10,
    ("UPF-OS", 16, 4): 7.20288e9,
    ("OLA", 16, 4) : 8.34503e9,
    ("UPF-OS", 16, 2) : 4.32158e9,
    ("OLA", 16, 2) : 5.00318e9,
    ("UPF-OS", 16, 1) : 2.88093e9,
    ("OLA", 16, 1) : 3.33226e9,
}

# Change these three lines to reuse the same script for alternate bar charts.
BAR_VALUE = "mflops_per_w"          # y-axis metric to plot
BAR_VALUE_STD = "mflops_per_w_std"  # std column after aggregation
SELECTION_METRIC = "mflops_per_w"   # metric used to choose one operating frequency
SELECTION_MODE: Literal["max", "min"] = "max"


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


def algorithm_family(df: pd.DataFrame) -> pd.Series:
    """Collapse backend names such as clfft-upols/fftw-upols/clfft-ola to UPF-OS/OLA."""
    if "Backend" in df.columns:
        src = df["Backend"].fillna("").astype(str).str.lower()
    elif "Algorithm" in df.columns:
        src = df["Algorithm"].fillna("").astype(str).str.lower()
    else:
        src = pd.Series("", index=df.index)

    is_upols = src.str.contains("upols", na=False) | src.str.contains("upf", na=False)
    is_ola = src.str.contains("ola", na=False)

    return pd.Series(
        np.select(
            [is_upols, is_ola],
            ["UPF-OS", "OLA"],
            default=src.str.upper(),
        ),
        index=df.index,
    )


def implementation_component(df: pd.DataFrame) -> pd.Series:
    """Return which component should define the operating frequency: GPU or CPU."""
    backend = df.get("Backend", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    platform = df.get("Platform", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()

    cpu_impl = (
        backend.str.contains("fftw|openmp|cpu", regex=True, na=False)
        | platform.str.contains("openmp|fftw|cpu", regex=True, na=False)
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


def device_backend(df: pd.DataFrame) -> pd.Series:
    """Collapse device names and backends into labels such as RPi_clfft or nano_cufft."""
    dev = df.get("Device", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()
    be = df.get("Backend", pd.Series("", index=df.index)).fillna("").astype(str).str.lower()

    be_name = np.select(
        [
            be.str.contains("fftw", na=False),
            be.str.contains("clfft", na=False),
            be.str.contains("cufft", na=False),
        ],
        ["fftw", "clfft", "cufft"],
        default="unknown",
    )

    dev_name = np.select(
        [
            dev.str.contains("jetsonorinnanosuper", na=False),
            dev.str.contains("raspberrypicomputemodule5", na=False),
        ],
        ["nano", "RPi"],
        default="unknown",
    )

    return pd.Series(dev_name + "_" + be_name, index=df.index)


def operation_count(df: pd.DataFrame) -> pd.Series:
    """Look up total operation count from algorithm family, channel count, and bank count."""

    alg = algorithm_family(df)

    channels = pd.to_numeric(df["Channels"], errors="coerce").astype("Int64")
    banks = pd.to_numeric(df["Banks"], errors="coerce").astype("Int64")

    keys = pd.Series(
        list(zip(alg, channels, banks)),
        index=df.index,
        dtype="object",
    )

    return keys.map(OP_COUNTS).astype(float)


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
    Metric("frequency_component", "Frequency Component", "", implementation_component),
    Metric("operating_frequency_mhz", "Operating Frequency", "MHz", pull_important_frequency),
    Metric("algorithm_family", "Algorithm", "", algorithm_family),
    Metric("dev_backend", "Device + Backend", "", device_backend),
    Metric("operation_count", "Operation Count", "ops", operation_count),
    Metric(
        "flops",
        "FLOPS",
        "FLOP/s",
        lambda df: df["operation_count"] / (pd.to_numeric(df["total_exec_ms"], errors="coerce") / 1000.0),
        higher_is_better=True,
    ),
    Metric(
        "mflops_per_w",
        "Efficiency",
        "MFLOPS/W",
        lambda df: (df["flops"] / 1.0e6) / pd.to_numeric(df["run_power_w"], errors="coerce"),
        higher_is_better=True,
    ),
    Metric(
        "throughput",
        "Throughput",
        "Input Samples/s",
        lambda df: 1_000_000 / (pd.to_numeric(df["total_exec_ms"], errors="coerce") / 1000.0),
        higher_is_better=True,
    ),
]


def main() -> None:
    csv_path = Path("./aggregated_csv/streaming_conv_results.csv")
    out_dir = Path("graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = BenchmarkData.from_csv(csv_path).with_metrics(DERIVED)

    # Use raw rows here. Once every bucket has enough repeats, you can replace
    # `plot_data` with a trimmed version that drops fastest/slowest runs per bucket.
    plot_data = data.where(Governor="performance")

    # First aggregate repeated benchmark runs at each operating frequency.
    freq_agg = plot_data.aggregate(
        ["Banks", "algorithm_family", "dev_backend", "frequency_component", "operating_frequency_mhz"],
        ["mflops_per_w", "throughput", "run_power_w"],
        aggregator="mean",
        include_std=True,
    )

    # Then reduce the frequency sweep to one row per visible bar. This is the
    # important step that prevents the bar chart from producing one bar per freq.
    bar_df = select_best_per_group(
        freq_agg,
        group_by=["Banks", "algorithm_family", "dev_backend"],
        value_col=SELECTION_METRIC,
        mode=SELECTION_MODE,
    )

    bar_pf = select_best_per_group(
        freq_agg,
        group_by=["Banks", "algorithm_family", "dev_backend"],
        value_col="throughput",
        mode=SELECTION_MODE,
    )

    dimensions = {
        "gpu_freq_mhz": Dimension("gpu_freq_mhz", "GPU Frequency", "MHz"),
        "cpu_freq_mhz": Dimension("cpu_freq_mhz", "CPU Frequency", "MHz"),
        "operating_frequency_mhz": Dimension("operating_frequency_mhz", "Operating Frequency", "MHz"),
        "frequency_component": Dimension("frequency_component", "Frequency Component"),
        "Device": Dimension("Device", "Device"),
        "algorithm_family": Dimension("algorithm_family", "Algorithm"),
        "dev_backend": Dimension("dev_backend", "Device + Backend"),
        "Banks": Dimension("Banks", "Problem Size"),
    }
    metrics = {m.name: m for m in DERIVED}
    plotter = Plotter(dimensions, metrics)

    plotter.plot(
        bar_df,
        PlotSpec(
            kind="bar",
            x="Banks",
            y=BAR_VALUE,
            title="FFT Convolution Efficiency by Backend, Algorithm, and Bank Count",
            output=str(out_dir / "efficiency_bar_best_frequency.png"),
            bar_group_by="dev_backend",
            bar_subgroup_by="algorithm_family",
            bar_label_mode="hierarchical",
            # bar_brackets="group",
            # bar_bracket_color="black",
            # bar_bracket_alpha=0.45,
            bar_multilevel_bottom=0.30,
            category_orders={
                "dev_backend": ["RPi_fftw", "RPi_clfft", "nano_clfft", "nano_cufft"],
                "algorithm_family": ["OLA", "UPF-OS"],
                "Banks": [16, 8, 4, 1],
            },
            value_aliases={
                "Banks": {
                    16: "16",
                    8: "8",
                    4: "4",
                    1: "1",    
                },
                "dev_backend": {
                    "RPi_fftw": "FFTW \N{RIGHTWARDS ARROW} CM5 CPU",
                    "RPi_clfft": "clFFT \N{RIGHTWARDS ARROW} CM5 GPU",
                    "nano_clfft": "clFFT \N{RIGHTWARDS ARROW} Orin Nano GPU",
                    "nano_cufft": "cuFFT \N{RIGHTWARDS ARROW} Orin Nano GPU",
                },
            },
            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano_cufft": "#54a24b",
                "nano_clfft": "#7634ac",
                "RPi_clfft": "#db1916",
                "RPi_fftw": "#162adb",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },
            legend="none",
            legend_fontsize=8,
            figsize=(10, 3),
            bar_value_labels=False,
        ),
    )

    plotter.plot(
        bar_pf,
        PlotSpec(
            kind="bar",
            x="Banks",
            y="throughput",
            title="FFT Convolution Throughput by Backend, Algorithm, and Bank Count",
            output=str(out_dir / "throughput_bar_best_frequency.png"),
            bar_group_by="dev_backend",
            bar_subgroup_by="algorithm_family",
            bar_label_mode="hierarchical",
            # bar_brackets="group",
            # bar_bracket_color="black",
            # bar_bracket_alpha=0.45,
            bar_multilevel_bottom=0.30,
            category_orders={
                "dev_backend": ["RPi_fftw", "RPi_clfft", "nano_clfft", "nano_cufft"],
                "algorithm_family": ["OLA", "UPF-OS"],
                "Banks": [16, 8, 4, 1],
            },
            value_aliases={
                "Banks": {
                    16: "16",
                    8: "8",
                    4: "4",
                    1: "1",    
                },
                "dev_backend": {
                    "RPi_fftw": "FFTW \N{RIGHTWARDS ARROW} CM5 CPU",
                    "RPi_clfft": "clFFT \N{RIGHTWARDS ARROW} CM5 GPU",
                    "nano_clfft": "clFFT \N{RIGHTWARDS ARROW} Orin Nano GPU",
                    "nano_cufft": "cuFFT \N{RIGHTWARDS ARROW} Orin Nano GPU",
                },
            },
            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano_cufft": "#54a24b",
                "nano_clfft": "#7634ac",
                "RPi_clfft": "#db1916",
                "RPi_fftw": "#162adb",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },
            legend="none",
            legend_fontsize=8,
            figsize=(10, 5),
            bar_value_labels=False,
        ),
    )

    freq_agg.to_csv(out_dir / "banks_vs_metrics_all_operating_frequencies.csv", index=False)
    bar_df.to_csv(out_dir / "banks_vs_metrics_best_operating_frequency.csv", index=False)


if __name__ == "__main__":
    main()
