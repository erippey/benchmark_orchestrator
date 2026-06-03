import math

from graph_tool_v2_skeleton import *


def plot_bank_count(gpu_test_name, cpu_test_name, omp_test_name):
    independent_variable = {
        "var_name": "v3d_freq_min",
        "proper_name": "GPU Frequency (MHz)"
    }

 

    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", forward_complexity),
        ("Multiply Accumulate Execution Time", "complex_multiply", mac_complexity),
        ("Inverse FFT Execution Time", "inverse", inverse_complexity)
    }
    clvk_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "UPOLS using clvk\N{RIGHTWARDS ARROW}GPU", 
                   test_name=TEST_NAME_MAP[("clfft",None,bank_count)]["test_name"], graph_name_for_file=TEST_NAME_MAP[("clfft",None,bank_count)]["graph_name"], kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops, peak_mflops=5270)
    
    independent_variable = {
        "var_name": "arm_freq_min",
        "proper_name": "CPU Frequency (MHz)"
    }

    forward_complexity = n_fft * math.log2(n_fft) * 5
    mac_complexity = parts * n_fft * 8 
    inverse_complexity = n_fft * math.log2(n_fft) * 5

    total_ops_2 = (forward_complexity + (mac_complexity + inverse_complexity) * banks) * channels * total_blocks
    if total_ops != total_ops:
        raise Exception(f"Operations do not equal: {total_ops} vs {total_ops_2}")

    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", forward_complexity),
        ("Multiply Accumulate Execution Time", "complex_multiply", mac_complexity),
        ("Inverse FFT Execution Time", "inverse", inverse_complexity),
    }

    fftw_graph_ps = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}CPU", 
                   test_name=TEST_NAME_MAP[("fftw","powersave",bank_count)]["test_name"], graph_name_for_file=TEST_NAME_MAP[("fftw","powersave",bank_count)]["graph_name"], kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops_2, peak_mflops=38374)

    independent_variable = {
        "var_name": "arm_freq",
        "proper_name": "CPU Frequency (MHz)"
    }
    
    fftw_graph_pf = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}CPU", 
                   test_name=TEST_NAME_MAP[("fftw","performance",bank_count)]["test_name"], graph_name_for_file=TEST_NAME_MAP[("fftw","performance",bank_count)]["graph_name"], kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops_2, peak_mflops=38374)

    
    #normalize_graphs([clvk_graph, fftw_graph_pf, fftw_graph_ps])

    clvk_graph.plot()
    fftw_graph_pf.plot()
    fftw_graph_ps.plot()

if __name__ == "__main__":
   
    dims = {
        "gpu_freq": Dimension("v3d_freq_mhz", "GPU frequency", "MHz"),
        "cpu_freq": Dimension("arm_freq", "CPU frequency", "MHz"),
        "algorithm": Dimension("algorithm", "Algorithm")
    }


    data = BenchmarkData.from_csv("aggregated_csv/aggregated_results.csv").with_metrics([
        runtime_ms("Kernel Runtime", "kernel_runtime", "Kernel Runtime"),
        runtime_ms("Region of Interest", "roi_runtime", "Region of Interest"),
        runtime_power()
        rutnime_power("idle_power_w", "idle_power_w", "Idle Power"),
        energy_j("Kernel Runtime", "run_power_w"),
        edp("Kernel Runtime", "energy_j")
    ])

    agg = data.aggregate(
        group_by=["Algorithm", "v3d_freq", "arm_freq"],
        value_cols=["kernel_runtime", "roi_runtime", "run_power_w", "energy_j", "edp_j_s"]
    )

    plotter = Plotter(dims, data.metrics)
    plotter.plot(agg, PlotSpec(
        kind="line",
        x="gpu_freq",
        y="runtime_ms", 
        yerr="runtime_ms_std",
        color_by="algorithm",
        title="Runtime vs GPU frequency",
        output="graphs/runtime_by_gpu_freq.png"
    ))

    
