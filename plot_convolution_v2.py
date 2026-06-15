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
        runtime_ms("conv_avg_ms", "convolution_rt", "Convolution Latency"),
        runtime_ms("total_exec_ms", "total_exec", "Total Execution"),
        runtime_power(),
        runtime_power("idle_power_w", "idle_power_w", "Idle Power"),
        energy_j("total_exec_ms", "run_power_w"),
        edp("total_exec_ms", "energy_j"),
    ])

    data = data.where(Governor="powersave").where(Device="RaspberryPiComputeModule5")
    gpu_data = data.where (Backend="fftw-upols")
    #cpu_data = data.where_or(Backend="fftwola", Backend="fftw-upols")


    # Make aggregated grouped datasets
    # agg =  data.aggregate(
    #         group_by=[
    #             "Backend",
    #             "v3d_freq_min",
    #             "arm_freq_min",
    #             "Platform",
    #         ],
    #         value_cols=[
    #             "convolution_rt",
    #             "total_exec",
    #             "energy_j",
    #             "edp_j_s",
    #             "run_power_w"
    #         ]
    #     )
    
    gpu_agg = gpu_data.aggregate(
        group_by=[
                "v3d_freq_min",
                "arm_freq_min",
                "Platform",
                "Banks",
            ],
            value_cols=[
                "convolution_rt",
                "total_exec",
                "energy_j",
                "edp_j_s",
                "run_power_w"
            ]
    )

    # gpu_agg = cpu_data.aggregate(
    #     group_by=[
    #             "Backend",
    #             "v3d_freq_min",
    #             "arm_freq_min",
    #             "Platform",
    #         ],
    #         value_cols=[
    #             "convolution_rt",
    #             "total_exec",
    #             "energy_j",
    #             "edp_j_s",
    #             "run_power_w"
    #         ]
    # )

 

    plotter = Plotter(dims, data.metrics)

    plotter.plot(gpu_agg, PlotSpec(
        kind="line",
        x="arm_freq_min",
        xlabel="CPU Frequency MHz",
        y="convolution_rt",
        ylabel="Convolution Runtime",

        figsize=(7.6,6.6),

        title="Runtime of FFTW Uniformly Partitioned Overlap Save Convolution Call vs CPU Frequency",
        output="graphs/convolution_time_by_cpu_freq.png",

        series_by=["Banks"],
        
        hue_by="Banks",
        marker_by="Banks",


        legend="top left",
        legend_fontsize=8,

    ))

    plotter.plot(gpu_agg, PlotSpec(
        kind="line",
        x="arm_freq_min",
        xlabel="CPU Frequency MHz",
        y="run_power_w",
        ylabel="Average Power Consumption (W)",

        figsize=(7.6,6.6),

        title="Average Power Consumption of FFTW Uniformly Partitioned Overlap Save Convolution Call vs CPU Frequency",
        output="graphs/upols_power_by_cpu_freq.png",

        series_by=["Banks"],
        
        hue_by="Banks",
        marker_by="Banks",


        legend="top left",
        legend_fontsize=8,

    ))

