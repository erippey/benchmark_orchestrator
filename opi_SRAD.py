import math

from automated_csv_to_plot import *

orange_pi5_cpu_little_uv = {
    408000.0: 675_000,
    600000.0: 675_000,
    816000.0: 675_000,
    1008000.0: 675_000,
    1200000.0: 712_500,
    1416000.0: 762_500,
    1608000.0: 850_000,
    1800000.0: 950_000,
}

orange_pi5_cpu_big_uv = {
    408000.0: 675_000,
    600000.0: 675_000,
    816000.0: 675_000,
    1008000.0: 675_000,
    1200000.0: 675_000,
    1416000.0: 725_000,
    1608000.0: 762_500,
    1800000.0: 850_000,
    2016000.0: 925_000,
    2208000.0: 987_500,
    2256000.0: 1_000_000,
    2304000.0: 1_000_000,
    2352000.0: 1_000_000,
    2400000.0: 1_000_000,
}

orange_pi5_gpu_uv = {
    300_000_000: 675_000,
    400_000_000: 675_000,
    500_000_000: 675_000,
    600_000_000: 675_000,
    700_000_000: 700_000,
    800_000_000: 750_000,
    900_000_000: 800_000,
    1_000_000_000: 850_000,
}

def plot_test(cpu_freq):
    independent_variable = {
        "var_name": "gpu_freq",
        "proper_name": "GPU Frequency (MHz)"
    }
    
    gpu_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "UPOLS using OpenCL\N{RIGHTWARDS ARROW}GPU on Orange PI", 
                   test_name=f"SRAD_opencl_vs_GPU_freq_A76_{cpu_freq}", graph_name_for_file=f"graphs/Orange_Pi_5/SRAD/opencl_srad_vs_gpu_freq_a76_{cpu_freq}",
                   drop_highest_n=1, opp=orange_pi5_gpu_uv, performance=False, kernel_percent_peak=False)
    
    independent_variable = {
        "var_name": "policy0_freq",
        "proper_name": "CPU Frequency (MHz)"
    }

    # LITTL graph does not exist yet

    independent_variable = {
        "var_name": "policy4_freq",
        "proper_name": "CPU Frequency (MHz)"
    }

    # big graph does not exist yet 

    gpu_graph.plot(show_estimate=True)


    

if __name__ == "__main__":
    independent_variable = {
        "var_name": "gpu_freq",
        "proper_name": "GPU Frequency (MHz)"
    }


    # for i in [408, 600, 816, 1008, 1200, 1416, 1608, 1800, 2016, 2208, 2256]:

    #     plot_test(i)

    for i in [408, 600, 816, 1008]:

        plot_test(i)