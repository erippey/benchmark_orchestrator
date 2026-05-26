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
    

if __name__ == "__main__":
    independent_variable = {
        "var_name": "gpu_freq",
        "proper_name": "GPU Frequency (MHz)"
    }
    block_size = 256
    n_fft = 2 * block_size
    banks = 16
    n_fir = 3047
    channels = 16
    parts = math.ceil(n_fir / block_size)
    signal_len = 1000000
    total_blocks = math.ceil(signal_len / block_size)

    forward_complexity = n_fft * math.log2(n_fft) * 5 * channels
    mac_complexity = parts * n_fft * banks * channels * 8 
    inverse_complexity = n_fft * math.log2(n_fft) * 5 * banks * channels
    total_ops = (forward_complexity + mac_complexity + inverse_complexity) * total_blocks


    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", forward_complexity),
        ("Multiply Accumulate Execution Time", "complex_multiply", mac_complexity),
        ("Inverse FFT Execution Time", "inverse", inverse_complexity)
    }
    clvk_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "UPOLS using OpenCL\N{RIGHTWARDS ARROW}GPU on Orange PI", 
                   test_name="clfft_upols_vs_v3d_freq_16_banks", graph_name_for_file="graphs/clfft_upols_16_bank_opi", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops, peak_mflops=5270, drop_highest_n=1, opp=orange_pi5_gpu_uv)
    
    clvk_graph_1kb = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "UPOLS using OpenCL\N{RIGHTWARDS ARROW}GPU on Orange PI", 
                   test_name="clfft_upols_vs_gpu_freq_ondemand_1_knob_16_banks", graph_name_for_file="graphs/clfft_upols_16_bank_opi_1_knob", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops, peak_mflops=5270, drop_highest_n=1, opp=orange_pi5_gpu_uv)
    
    independent_variable = {
        "var_name": "policy0_freq",
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

    fftw_graph_ps_a55 = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}OPI A55 CPU", 
                   test_name="fftw_upols_vs_A55_freq_powersave_16_banks", graph_name_for_file="graphs/fftw_upols_ps_16_bank_opi_A55", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops_2, peak_mflops=38374, drop_highest_n=1, opp=orange_pi5_cpu_little_uv)

    
    fftw_graph_pf_a55 = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}OPI A55 CPU", 
                   test_name="ffrw_upols_vs_A55_freq_performance_16_banks", graph_name_for_file="graphs/fftw_upols_pf_16_bank_opi_A55", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops_2, peak_mflops=38374, opp=orange_pi5_cpu_little_uv)
    
    fftw_graph_od_a55 = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}OPI A55 CPU", 
                   test_name="fftw_upols_vs_A55_freq_ondemand_1_knob_16_banks", graph_name_for_file="graphs/fftw_upols_od_16_bank_opi_A55_1_knob", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops_2, peak_mflops=38374, opp=orange_pi5_cpu_little_uv)

    independent_variable = {
        "var_name": "policy4_freq",
        "proper_name": "CPU Frequency (MHz)"
    }

    fftw_graph_ps_a76 = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}OPI A76 CPU", 
                   test_name="fftw_upols_vs_A76_freq_powersave_16_banks", graph_name_for_file="graphs/fftw_upols_ps_16_bank_opi_A76", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops_2, peak_mflops=38374, drop_highest_n=1, opp=orange_pi5_cpu_big_uv)

    
    fftw_graph_pf_a76 = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}OPI A76 CPU", 
                   test_name="ffrw_upols_vs_A76_freq_performance_16_banks", graph_name_for_file="graphs/fftw_upols_pf_16_bank_opi_A76", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops_2, peak_mflops=38374, drop_highest_n=1, opp=orange_pi5_cpu_big_uv)
    
    fftw_graph_od_a76 = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}OPI A55 CPU", 
                   test_name="fftw_upols_vs_A76_freq_ondemand_1_knob_16_banks", graph_name_for_file="graphs/fftw_upols_od_16_bank_opi_A76_1_knob", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops_2, peak_mflops=38374, opp=orange_pi5_cpu_big_uv)


    
    #normalize_graphs([clvk_graph, fftw_graph_pf_a55, fftw_graph_ps_a55, fftw_graph_pf_a76, fftw_graph_ps_a76])

    clvk_graph.plot(show_estimate=True)
    fftw_graph_pf_a55.plot(show_estimate=True)
    fftw_graph_ps_a55.plot(show_estimate=True)
    fftw_graph_pf_a76.plot(show_estimate=True)
    fftw_graph_ps_a76.plot(show_estimate=True)

    fftw_graph_od_a55.plot(show_estimate=True)
    fftw_graph_od_a76.plot(show_estimate=True)
    clvk_graph_1kb.plot(show_estimate=True)
    
