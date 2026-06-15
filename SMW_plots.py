from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from graph_tool_v2_series import (
    BenchmarkData,
    Dimension,
    Metric,
    PlotSpec,
    Plotter,
    TrimSpec,
)

# Operation counts for one complete benchmarked run.
# These are NOT FLOPS yet; FLOPS = operations / runtime_seconds.
OP_COUNTS = {
    "UPF-OS": 2.44907e10,
    "OLA": 2.83961e10,
}


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


def first_positive_frequency_mhz(df: pd.DataFrame, columns: list[str]) -> pd.Series:
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

    return pd.Series(
        np.select(
            [src.str.contains("upols"), src.str.contains("ola")],
            ["UPF-OS", "OLA"],
            default=src.str.upper(),
        ),
        index=df.index,
    )


def operation_count(df: pd.DataFrame) -> pd.Series:
    fam = algorithm_family(df)
    return fam.map(OP_COUNTS).astype(float)


# Derived columns/metrics to add to BenchmarkData.
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
    Metric("algorithm_family", "Algorithm", "", algorithm_family),
    Metric("operation_count", "Operation Count", "ops", operation_count),
    Metric(
        "flops",
        "Throughput",
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
]


def main() -> None:
    csv_path = Path("./aggregated_csv/aggregated_results.csv")
    out_dir = Path("graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = BenchmarkData.from_csv(csv_path).with_metrics(DERIVED)

    # Use raw rows here. Once every bucket has enough repeats, you can replace
    # `data` with a trimmed version that drops fastest/slowest runs per bucket.
    plot_data = data

    gpu = plot_data.subset(
        lambda df: (
            df["gpu_freq_mhz"].notna()
            & (df["gpu_freq_mhz"] > 0)
            & df["Backend"].astype(str).str.startswith("clfft")
        )
    )
    cpu = plot_data.subset(
        lambda df: (
            df["cpu_freq_mhz"].notna()
            & (df["cpu_freq_mhz"] > 0)
            & df["Backend"].astype(str).str.startswith(("fftw", "omp", "openmp"))
        )
    )

    gpu_agg = gpu.aggregate(
        ["Device", "algorithm_family", "gpu_freq_mhz"],
        ["mflops_per_w"],
        aggregator="mean",
        include_std=True,
    )
    cpu_agg = cpu.aggregate(
        ["Device", "algorithm_family", "cpu_freq_mhz"],
        ["mflops_per_w"],
        aggregator="mean",
        include_std=True,
    )

    gpu_agg.to_csv(out_dir / "gpu_freq_vs_mflops_per_w.csv", index=False)
    cpu_agg.to_csv(out_dir / "cpu_freq_vs_mflops_per_w.csv", index=False)

    dimensions = {
        "gpu_freq_mhz": Dimension("gpu_freq_mhz", "GPU Frequency", "MHz"),
        "cpu_freq_mhz": Dimension("cpu_freq_mhz", "CPU Frequency", "MHz"),
        "Device": Dimension("Device", "Device"),
        "algorithm_family": Dimension("algorithm_family", "Algorithm"),
    }
    metrics = {m.name: m for m in DERIVED}
    plotter = Plotter(dimensions, metrics)

    plotter.plot(
        gpu_agg,
        PlotSpec(
            kind="line",

            x="gpu_freq_mhz",
            xlabel_fontsize=20,
            y="mflops_per_w",
            ylabel_fontsize=20,
            yerr="mflops_per_w_std",
            series_by=["Device", "algorithm_family"],

            ylim=(110, 500),


            hue_by="Device",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "JetsonOrinNanoSuper": "#54a24b",
                "RaspberryPiComputeModule5": "#db1916",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title="GPU Frequency vs Convolution Efficiency",
            title_fontsize=24,
            output=str(out_dir / "gpu_freq_vs_mflops_per_w.png"),

            legend="none",


            highlight_highest_by=["Device", "algorithm_family"],
            highlight_highest_label="circled: best point within each device/algorithm",
            highlight_linewidth=2,
            highlight_size=115,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        cpu_agg,
        PlotSpec(
            kind="line",


            x="cpu_freq_mhz",
            xlabel_fontsize=20,
            y="mflops_per_w",
            ylabel_fontsize=20,
            yerr="mflops_per_w_std",
            series_by=["Device", "algorithm_family"],

            ylim=(110, 500),


            hue_by="Device",
            marker_by="algorithm_family",

            base_colors={
                "JetsonOrinNanoSuper": "#7634ac",
                "RaspberryPiComputeModule5": "#162adb",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title="CPU Frequency vs Convolution Efficiency",
            title_fontsize=24,
            output=str(out_dir / "cpu_freq_vs_mflops_per_w.png"),

            legend="none",


            highlight_highest_by=["Device", "algorithm_family"],
            highlight_highest_label="circled: best point within each device/algorithm",
            highlight_linewidth=3,
            highlight_size=115,
            highlight_alpha=1,
        ),
    )

    print(out_dir)
    print(gpu_agg)
    print(cpu_agg)


if __name__ == "__main__":
    main()
