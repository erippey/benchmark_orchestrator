import math

from graph_tool_v2_series import *


if __name__ == "__main__":
   
   # define dimensions I guess?
    dims = {
        "gpu_freq": Dimension("v3d_freq_mhz", "GPU frequency", "MHz"),
        "cpu_freq": Dimension("arm_freq", "CPU frequency", "MHz"),
        "algorithm": Dimension("algorithm", "Algorithm")
    }

    # pull important rows of data from CSV file
    data = BenchmarkData.from_csv("aggregated_csv/aggregated_results.csv").with_metrics([
        runtime_ms("Kernel Runtime", "kernel_runtime", "Kernel Runtime"),
        runtime_ms("Region of Interest", "roi_runtime", "Region of Interest"),
        runtime_power(),
        runtime_power("idle_power_w", "idle_power_w", "Idle Power"),
        energy_j("Kernel Runtime", "run_power_w"),
        edp("Kernel Runtime", "energy_j"),
    ])

    # isolate OpenMP and OpenCL produced datasets
    opencl_data = data.where(Platform="OpenCL")
    openmp_data = data.where(Platform="OpenMP")

    # Make aggregated grouped datasets
    agg = {
        "ALL": data.aggregate(
            group_by=[
                "Algorithm",
                "v3d_freq",
                "arm_freq",
                "Platform",
                "Threads"
            ],
            value_cols=[
                "kernel_runtime",
                "energy_j",
                "edp_j_s",
                "run_power_w"
            ]
        ),
        "CPU": openmp_data.aggregate(
            group_by=[
                "Algorithm",
                "Threads",
                "arm_freq",
                "Platform"
            ],
            value_cols=[
                "kernel_runtime",
                "roi_runtime",
                "run_power_w",
                "energy_j",
                "edp_j_s",
            ],
        ),
        "GPU": opencl_data.aggregate(
            group_by=[
                "Algorithm",
                "v3d_freq",
                "Threads",
                "Platform",
            ],
            value_cols=[
                "kernel_runtime",
                "roi_runtime",
                "run_power_w",
                "energy_j",
                "edp_j_s",
            ],
        )
    }


    

    

    # ADD additional imporant rows generated from aggregated data seperated by datatype
    for device, relative_group in zip(["ALL", "CPU", "GPU"], [["Algorithm"], ["Algorithm"], ["Algorithm"]]):
        agg[device] = add_relative_to_group_best(
            agg[device],
            "kernel_runtime",
            relative_group,
            higher_is_better=True,
            out_col="rel_kernel_runtime",
        )

        agg[device] = add_relative_to_group_best(
            agg[device],
            "energy_j",
            relative_group,
            higher_is_better=True,
            out_col="rel_energy_j",
        )

        agg[device] = add_relative_to_group_best(
            agg[device],
            "edp_j_s",
            relative_group,
            higher_is_better=False,
            out_col="rel_edp",
        )

        agg[device]["Variant"] = np.where(
            agg[device]["Platform"] == "OpenCL",
            "OpenCL",
            "OpenMP-t" + agg[device]["Threads"].astype("Int64").astype(str)
        )

 

    plotter = Plotter(dims, data.metrics)

    plotter.plot(agg["ALL"], PlotSpec(
        kind="scatter",
        x="run_power_w",
        xlabel="Average Power (W)",
        y="rel_kernel_runtime",
        ylabel="Relative (%) of Max Runtime",
        ylim=(0,0.002),

        figsize=(7.6,6.6),

        title="Relative Kernel Runtime Accross Implementations vs Average Power",
        output="graphs/runtime_by_power.png",

        series_by=["Algorithm", "Variant"],
        
        hue_by="Algorithm",
        shade_by="Variant",
        marker_by="Variant",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
        },

        shade_values={
            "OpenCL":   -0.25,   # dark
            "OpenMP-t1":  0.00,   # medium
            "OpenMP-t4":  0.35,   # light
        },

        legend="none",
        legend_fontsize=8,

    ))

    plotter.plot(agg["ALL"], PlotSpec(
        kind="scatter",
        x="run_power_w",
        xlabel="Average Power (W)",
        xlim=(3.5, 4.5),
        y="rel_energy_j",
        ylabel="Relative (%) of Max Energy Consumption",

        figsize=(7.6,6.6),

        title="Relative Kernel Energy Consumption Across Implementations vs Average Power Draw",
        output="graphs/energy_by_power.png",

        series_by=["Algorithm", "Variant"],
        
        hue_by="Algorithm",
        shade_by="Variant",
        marker_by="Variant",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
        },

        shade_values={
            "OpenCL":   -0.25,   # dark
            "OpenMP-t1":  0.00,   # medium
            "OpenMP-t4":  0.35,   # light
        },

        legend="none",
        legend_fontsize=8,
    ))

    
    plotter.plot(agg["GPU"], PlotSpec(
        kind="line",
        x="v3d_freq",
        xlabel="GPU Frequency (MHz)",
        y="rel_kernel_runtime",
        ylabel="Relative (%) of Max Runtime",

        figsize=(7.6,6.6),

        title="Relative OpenCL Kernel Runtime vs GPU Frequency",
        output="graphs/opencl_runtime_by_gpu_freq.png",

        series_by=["Algorithm"],

        hue_by="Algorithm",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
        },

        shade_values={
            "OpenCL":   -0.25,   # dark
        },

        legend="none",
        legend_fontsize=8,
    ))

    plotter.plot(agg["GPU"], PlotSpec(
        kind="line",
        x="v3d_freq",
        xlabel="GPU Frequency (MHz)",
        y="run_power_w",
        ylabel="Average Power Draw (W)",

        figsize=(7.6,6.6),

        title="OpenCL Average Power Consumtion vs GPU Frequency",
        output="graphs/opencl_power_by_gpu_freq.png",

        series_by=["Algorithm"],

        hue_by="Algorithm",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
        },

        shade_values={
            "OpenCL":   -0.25,   # dark
        },

        legend="none",
        legend_fontsize=8,
    ))

    plotter.plot(agg["GPU"], PlotSpec(
        kind="line",
        x="v3d_freq",
        xlabel="GPU Frequency (MHz)",
        y="rel_energy_j",
        ylabel="Relative (%) of Max Evergy Consumption",
        
        title="Relative OpenCL Energy-to-Solution vs GPU Frequency",
        output="graphs/opencl_energy_j_by_gpu_freq.png",

        series_by=["Algorithm"],

        hue_by="Algorithm",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
        },

        shade_values={
            "OpenCL":   -0.25,   # dark
        },

        legend="outside_right",
        legend_fontsize=8,
    ))
    

    plotter.plot(agg["CPU"], PlotSpec(
        kind="line",
        x="arm_freq",
        xlabel="CPU Frequency (MHz)",
        y="rel_kernel_runtime",
        ylabel="Relative (%) of Max Runtime",

        figsize=(7.6,6.6),

        title="Relative OpenMP Kernel Runtime vs CPU Frequency",
        output="graphs/openmp_runtime_by_cpu_freq.png",

        series_by=["Algorithm", "Variant"],

        hue_by="Algorithm",
        shade_by="Variant",
        marker_by="Variant",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
        },

        shade_values={
            "OpenMP-t1":  0.00,   # medium
            "OpenMP-t4":  0.35,   # light
        },

        legend="none",
        legend_fontsize=8,
    ))


    plotter.plot(agg["CPU"], PlotSpec(
        kind="line",
        x="arm_freq",
        xlabel="CPU Frequency (MHz)",
        y="run_power_w",
        ylabel="Average Power Draw (W)",

        figsize=(7.6,6.6),

        title="OpenMP Average Power Consumtion vs CPU Frequency",
        output="graphs/openmp_power_by_cpu_freq.png",

        series_by=["Algorithm", "Variant"],

        hue_by="Algorithm",
        shade_by="Variant",
        marker_by="Variant",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
        },

        shade_values={
            "OpenMP-t1":  0.00,   # medium
            "OpenMP-t4":  0.35,   # light
        },

        legend="none",
        legend_fontsize=8,
    ))


    plotter.plot(agg["CPU"], PlotSpec(
        kind="line",
        x="arm_freq",
        xlabel="CPU Frequency (MHz)",
        y="rel_energy_j",
        ylabel="Relative (%) of Max Evergy Consumption",
        
        title="Relative OpenMP Energy-to-Solution vs CPU Frequency",
        output="graphs/openmp_energy_j_by_cpu_freq.png",

        series_by=["Algorithm", "Variant"],

        hue_by="Algorithm",
        shade_by="Variant",
        marker_by="Variant",

        base_colors={
            "BFS":    "#4c78a8",
            "FFT":    "#54a24b",
            "KMeans": "#9c6ade",
            "SRAD":   "#7f7f7f",
        },

        shade_values={
            "OpenMP-t1":  0.00,   # medium
            "OpenMP-t4":  0.35,   # light
        },

        legend="outside_right",
        legend_fontsize=8,
    ))

    
