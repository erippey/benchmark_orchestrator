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
    n_fft = 16384
    banks = 1
    n_fir = 3047
    l_block = n_fft - n_fir + 1
    channels = 1
    signal_len = 21368782
    total_blocks = math.ceil(signal_len / l_block)

    forward_complexity = n_fft * math.log2(n_fft) * 5 * channels
    mac_complexity = n_fft * banks * channels * 6
    inverse_complexity = n_fft * math.log2(n_fft) * 5 * banks * channels
    total_ops = (forward_complexity + mac_complexity + inverse_complexity) * total_blocks


    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", forward_complexity),
        ("Complex Multiply Execution Time", "complex_multiply", mac_complexity),
        ("Inverse FFT Execution Time", "inverse", inverse_complexity)
    }

    clvk_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "UPOLS using OpenCL\N{RIGHTWARDS ARROW}GPU on Orange PI", 
                   test_name="clfft_ola_vs_gpu_freq_powersave_16_banks", graph_name_for_file="graphs/clfft_ola_16_bank_opi", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops, peak_mflops=5270, opp=orange_pi5_gpu_uv)
    
    independent_variable = {
        "var_name": "policy0_freq",
        "proper_name": "CPU Frequency (MHz)"
    }



    fftw_graph_ps_a55 = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}OPI A55 CPU", 
                   test_name="fftw_ola_vs_A55_freq_powersave_16_banks", graph_name_for_file="graphs/fftw_ola_ps_16_bank_opi_A55", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops, peak_mflops=38374, opp=orange_pi5_cpu_little_uv)

    


    independent_variable = {
        "var_name": "policy4_freq",
        "proper_name": "CPU Frequency (MHz)"
    }


    fftw_graph_ps_a76 = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}OPI A76 CPU", 
                   test_name="fftw_ola_vs_A76_freq_powersave_16_banks", graph_name_for_file="graphs/fftw_ola_ps_16_bank_opi_A76", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=total_ops, peak_mflops=38374, opp=orange_pi5_cpu_big_uv)

    
    #normalize_graphs([clvk_graph, fftw_graph_pf_a55, fftw_graph_ps_a55])

    clvk_graph.plot(show_estimate=True)
    fftw_graph_ps_a76.plot(show_estimate=True)
    fftw_graph_ps_a55.plot(show_estimate=True)
    
