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


def device_backend(df: pd.DataFrame) -> pd.Series:
    """Collapse device names and backends into labels such as:
       RPi-clfft, nano-cufft, etc.
    """

    if "Device" in df.columns:
        dev = df["Device"].fillna("").astype(str).str.lower()
    else:
        dev = pd.Series("", index=df.index)

    if "Backend" in df.columns:
        be = df["Backend"].fillna("").astype(str).str.lower()
    else:
        be = pd.Series("", index=df.index)

    # Backend names
    be_name = np.select(
        [
            be.str.contains("fftw", na=False),
            be.str.contains("clfft", na=False),
            be.str.contains("cufft", na=False),
        ],
        [
            "fftw",
            "clfft",
            "cufft",
        ],
        default="unknown",
    )

    # Device names
    dev_name = np.select(
        [
            dev.str.contains("jetsonorinnanosuper", na=False),
            dev.str.contains("raspberrypicomputemodule5", na=False),
        ],
        [
            "nano",
            "RPi",
        ],
        default="unknown",
    )

    return pd.Series(
        dev_name + "-" + be_name,
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
    Metric("dev_backend", "Variant", "", device_backend),
    Metric("operation_count", "Operation Count", "ops", operation_count),
    Metric(
        "flops",
        "flops",
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
        lambda df: 1000000 / (pd.to_numeric(df["total_exec_ms"], errors="coerce") / 1000.0),
        higher_is_better=True,
    )
]


def smw_plot(bank_count: int = 0) -> None:
    requested_bank_count = bank_count != 0
    bank_count = bank_count or 16

    def out_file(name: str) -> Path:
        path = out_dir / name
        if requested_bank_count:
            return path.with_stem(f"{path.stem}_{bank_count}")
        return path

    def plot_title(title: str) -> str:
        if not requested_bank_count:
            return title

        bank_word = "bank" if bank_count == 1 else "banks"
        return f"{title} - {bank_count} {bank_word}"

    csv_path = Path("./aggregated_csv/streaming_conv_results.csv")
    out_dir = Path("graphs")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = BenchmarkData.from_csv(csv_path).with_metrics(DERIVED)

    # Use raw rows here. Once every bucket has enough repeats, you can replace
    # `data` with a trimmed version that drops fastest/slowest runs per bucket.
    plot_data = data.where(Banks=bank_count).where(Governor="performance")


    gpu = plot_data.subset(
        lambda df: (
            df["gpu_freq_mhz"].notna()
            & (df["gpu_freq_mhz"] > 0)
            & df["Backend"].astype(str).str.startswith(("clfft", "cufft"))
        )
    )
    cpu = plot_data.subset(
        lambda df: (
            df["cpu_freq_mhz"].notna()
            & (df["cpu_freq_mhz"] > 0)
            & df["Backend"].astype(str).str.startswith(("fftw", "omp", "openmp"))
        )
    )

    rpi_gpu = gpu.where(Device="RaspberryPiComputeModule5")

    all = data.aggregate(
        ["Device", "algorithm_family", "dev_backend", "gpu_freq_mhz"],
        ["mflops_per_w", "throughput", "run_power_w", "conv_avg_ms"],
        aggregator="mean",
        include_std=True,
    )

    gpu_agg = gpu.aggregate(
        ["Device", "algorithm_family", "dev_backend", "gpu_freq_mhz"],
        ["mflops_per_w", "throughput", "run_power_w", "conv_avg_ms"],
        aggregator="mean",
        include_std=True,
    )
    cpu_agg = cpu.aggregate(
        ["Device", "algorithm_family", "dev_backend", "cpu_freq_mhz"],
        ["mflops_per_w", "throughput", "run_power_w", "conv_avg_ms"],
        aggregator="mean",
        include_std=True,
    )

    rpi_gpu = rpi_gpu.aggregate(
        ["Device", "algorithm_family", "dev_backend", "gpu_freq_mhz"],
        ["mflops_per_w", "throughput", "run_power_w", "conv_avg_ms"],
        aggregator="mean",
        include_std=True,
    )

    gpu_agg.to_csv(out_file("gpu_freq_vs_mflops_per_w.csv"), index=False)
    cpu_agg.to_csv(out_file("cpu_freq_vs_mflops_per_w.csv"), index=False)

    dimensions = {
        "gpu_freq_mhz": Dimension("gpu_freq_mhz", "GPU Frequency", "MHz"),
        "cpu_freq_mhz": Dimension("cpu_freq_mhz", "CPU Frequency", "MHz"),
        "Device": Dimension("Device", "Device"),
        "algorithm_family": Dimension("algorithm_family", "Algorithm"),
    }
    metrics = {m.name: m for m in DERIVED}
    plotter = Plotter(dimensions, metrics)


    plotter.plot(
        all,
        PlotSpec(
            kind="line",

            x="gpu_freq_mhz",
            xlabel_fontsize=14,
            y="mflops_per_w",
            ylabel_fontsize=14,
            yerr="mflops_per_w_std",
            series_by=["algorithm_family", "dev_backend"],

            #ylim=(120, 1900),


            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano-cufft": "#54a24b",
                "nano-clfft": "#7634ac",
                "RPi-clfft": "#db1916",
                "RPi-fftw": "#162adb",
                "RaspberryPiComputeModule5": "#162adb",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("GPU Frequency vs Convolution Efficiency"),
            title_fontsize=10,
            output=out_file("aggregate.png"),

            legend="outside_right",


            highlight_highest_by=["dev_backend", "algorithm_family"],
            highlight_highest_label="Lowest Power",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        gpu_agg,
        PlotSpec(
            kind="line",

            x="gpu_freq_mhz",
            xlabel_fontsize=14,
            y="mflops_per_w",
            ylabel_fontsize=14,
            yerr="mflops_per_w_std",
            series_by=["algorithm_family", "dev_backend"],

            #ylim=(120, 1900),


            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano-cufft": "#54a24b",
                "nano-clfft": "#7634ac",
                "RPi-clfft": "#db1916",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("GPU Frequency vs Convolution Efficiency"),
            title_fontsize=10,
            output=out_file("gpu_freq_vs_mflops_per_w.png"),

            legend="none",


            highlight_highest_by=["dev_backend", "algorithm_family"],
            highlight_highest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        gpu_agg,
        PlotSpec(
            kind="line",

            x="gpu_freq_mhz",
            xlabel_fontsize=14,
            y="throughput",
            ylabel="Samples/Second",
            ylabel_fontsize=14,
            yerr="throughput_std",
            series_by=["algorithm_family", "dev_backend"],

            #ylim=(120, 1900),


            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano-cufft": "#54a24b",
                "nano-clfft": "#7634ac",
                "RPi-clfft": "#db1916",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("GPU Frequency vs Input Samples per Second Throughput"),
            title_fontsize=10,
            output=out_file("gpu_freq_vs_throughput.png"),

            legend="none",


            highlight_highest_by=["dev_backend", "algorithm_family"],
            highlight_highest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        gpu_agg,
        PlotSpec(
            kind="line",

            x="gpu_freq_mhz",
            xlabel_fontsize=12,
            y="run_power_w",
            ylabel="Average Power (W)",
            ylabel_fontsize=12,
            yerr="run_power_w_std",
            series_by=["algorithm_family", "dev_backend"],
            figsize=(5,3),

            #ylim=(120, 1900),


            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano-cufft": "#54a24b",
                "nano-clfft": "#7634ac",
                "RPi-clfft": "#db1916",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("GPU Frequency vs Average Power Draw"),
            title_fontsize=10,
            output=out_file("gpu_freq_vs_power.png"),

            legend="none",


            highlight_lowest_by=["dev_backend", "algorithm_family"],
            highlight_lowest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )


    plotter.plot(
        rpi_gpu,
        PlotSpec(
            kind="line",

            x="gpu_freq_mhz",
            xlabel_fontsize=14,
            y="mflops_per_w",
            ylabel_fontsize=14,
            yerr="mflops_per_w_std",
            series_by=["algorithm_family", "dev_backend"],

            #ylim=(120, 1900),


            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano-cufft": "#54a24b",
                "nano-clfft": "#7634ac",
                "RPi-clfft": "#db1916",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("GPU Frequency vs Convolution Efficiency"),
            title_fontsize=10,
            output=out_file("gpu_freq_vs_mflops_per_w_pi_only.png"),

            legend="none",


            highlight_highest_by=["dev_backend", "algorithm_family"],
            highlight_highest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        rpi_gpu,
        PlotSpec(
            kind="line",

            x="gpu_freq_mhz",
            xlabel_fontsize=14,
            y="throughput",
            ylabel="Samples/Second",
            ylabel_fontsize=14,
            yerr="throughput_std",
            series_by=["algorithm_family", "dev_backend"],

            #ylim=(120, 1900),


            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano-cufft": "#54a24b",
                "nano-clfft": "#7634ac",
                "RPi-clfft": "#db1916",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("GPU Frequency vs Input Samples per Second Throughput"),
            title_fontsize=10,
            output=out_file("gpu_freq_vs_throughput_pi_only.png"),

            legend="none",


            highlight_highest_by=["dev_backend", "algorithm_family"],
            highlight_highest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        rpi_gpu,
        PlotSpec(
            kind="line",

            x="gpu_freq_mhz",
            xlabel_fontsize=12,
            y="run_power_w",
            ylabel="Average Power (W)",
            ylabel_fontsize=12,
            yerr="run_power_w_std",
            series_by=["algorithm_family", "dev_backend"],
            figsize=(5,2),

            #ylim=(120, 1900),


            hue_by="dev_backend",
            shade_by="algorithm_family",
            marker_by="algorithm_family",

            base_colors={
                "nano-cufft": "#54a24b",
                "nano-clfft": "#7634ac",
                "RPi-clfft": "#db1916",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("GPU Frequency vs Average Power Draw"),
            title_fontsize=10,
            output=out_file("gpu_freq_vs_power_pi_only.png"),

            legend="none",


            highlight_lowest_by=["dev_backend", "algorithm_family"],
            highlight_lowest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        cpu_agg,
        PlotSpec(
            kind="line",


            x="cpu_freq_mhz",
            xlabel_fontsize=14,
            y="mflops_per_w",
            ylabel_fontsize=14,
            yerr="mflops_per_w_std",
            series_by=["Device", "algorithm_family"],

            #ylim=(75, 1500),


            hue_by="Device",
            marker_by="algorithm_family",
            shade_by="algorithm_family",

            base_colors={
                "JetsonOrinNanoSuper": "#7634ac",
                "RaspberryPiComputeModule5": "#162adb",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("CPU Frequency vs Convolution Efficiency"),
            title_fontsize=10,
            output=out_file("cpu_freq_vs_mflops_per_w.png"),

            legend="none",


            highlight_highest_by=["Device", "algorithm_family"],
            highlight_highest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        cpu_agg,
        PlotSpec(
            kind="line",


            x="cpu_freq_mhz",
            xlabel_fontsize=14,
            y="throughput",
            ylabel="Samples/Second",
            ylabel_fontsize=14,
            yerr="throughput_std",
            series_by=["Device", "algorithm_family"],

            #ylim=(75, 1500),


            hue_by="Device",
            marker_by="algorithm_family",
            shade_by="algorithm_family",

            base_colors={
                "JetsonOrinNanoSuper": "#7634ac",
                "RaspberryPiComputeModule5": "#162adb",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("CPU Frequency vs Input Samples per Second Throughput"),
            title_fontsize=10,
            output=out_file("cpu_freq_vs_throughput.png"),

            legend="none",


            highlight_highest_by=["Device", "algorithm_family"],
            highlight_highest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    plotter.plot(
        cpu_agg,
        PlotSpec(
            kind="line",


            x="cpu_freq_mhz",
            xlabel_fontsize=12,
            y="run_power_w",
            ylabel="Average Power (W)",
            ylabel_fontsize=12,
            yerr="run_power_w_std",
            series_by=["Device", "algorithm_family"],
            figsize=(5,2),

            #ylim=(75, 1500),


            hue_by="Device",
            marker_by="algorithm_family",
            shade_by="algorithm_family",

            base_colors={
                "JetsonOrinNanoSuper": "#7634ac",
                "RaspberryPiComputeModule5": "#162adb",
            },

            shade_values={
                "UPF-OS":-0.35,
                "OLA": 0.35,
            },


            title=plot_title("CPU Frequency vs Average Power Draw"),
            title_fontsize=10,
            output=out_file("cpu_freq_vs_power.png"),

            legend="none",


            highlight_lowest_by=["Device", "algorithm_family"],
            highlight_lowest_label="circled: best point within each device/algorithm",
            highlight_linewidth=1.5,
            highlight_size=95,
            highlight_alpha=1,
        ),
    )

    print(out_dir)
    print(gpu_agg)
    print(cpu_agg)


def main() -> None:
    smw_plot(16)

if __name__ == "__main__":
    main()
