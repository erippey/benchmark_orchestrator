import math

from automated_csv_to_plot import *

TEST_NAME_MAP = {
    # ===== clFFT / clvk =====
    ("clfft", None, 1): {
        "test_name": "clfft_upols_vs_v3d_freq_1_bank",
        "graph_name": "graphs/clvk_upols_1_bank",
    },
    ("clfft", None, 2): {
        "test_name": "clfft_upols_vs_v3d_freq_2_bank",
        "graph_name": "graphs/clvk_upols_2_bank",
    },
    ("clfft", None, 4): {
        "test_name": "clfft_upols_vs_v3d_freq_4_banks",
        "graph_name": "graphs/clvk_upols_4_bank",
    },
    ("clfft", None, 8): {
        "test_name": "clfft_upols_vs_v3d_freq_8_banks",
        "graph_name": "graphs/clvk_upols_8_bank",
    },
    ("clfft", None, 16): {
        "test_name": "clfft_upols_vs_v3d_freq_16_banks",
        "graph_name": "graphs/clvk_upols_16_bank",
    },

    # ===== FFTW performance governor =====
    ("fftw", "performance", 1): {
        "test_name": "ffrw_upols_vs_arm_freq_performance_1_bank",
        "graph_name": "graphs/fftw_upols_pf_1_bank",
    },
    ("fftw", "performance", 2): {
        "test_name": "ffrw_upols_vs_arm_freq_performance_2_banks",
        "graph_name": "graphs/fftw_upols_pf_2_bank",
    },
    ("fftw", "performance", 4): {
        "test_name": "ffrw_upols_vs_arm_freq_performance_4_banks",
        "graph_name": "graphs/fftw_upols_pf_4_bank",
    },
    ("fftw", "performance", 8): {
        "test_name": "ffrw_upols_vs_arm_freq_performance_8_banks",
        "graph_name": "graphs/fftw_upols_pf_8_bank",
    },
    ("fftw", "performance", 16): {
        "test_name": "ffrw_upols_vs_arm_freq_performance_16_banks",
        "graph_name": "graphs/fftw_upols_pf_16_bank",
    },

    # ===== FFTW powersave governor =====
    ("fftw", "powersave", 1): {
        "test_name": "fftw_upols_vs_arm_freq_powersave_1_bank",
        "graph_name": "graphs/fftw_upols_ps_1_bank",
    },
    ("fftw", "powersave", 2): {
        "test_name": "fftw_upols_vs_arm_freq_powersave_2_banks",
        "graph_name": "graphs/fftw_upols_ps_2_bank",
    },
    ("fftw", "powersave", 4): {
        "test_name": "fftw_upols_vs_arm_freq_powersave_4_banks",
        "graph_name": "graphs/fftw_upols_ps_4_bank",
    },
    ("fftw", "powersave", 8): {
        "test_name": "fftw_upols_vs_arm_freq_powersave_8_banks",
        "graph_name": "graphs/fftw_upols_ps_8_bank",
    },
    ("fftw", "powersave", 16): {
        "test_name": "fftw_upols_vs_arm_freq_powersave_16_banks",
        "graph_name": "graphs/fftw_upols_ps_16_bank",
    },
}

def plot_bank_count(bank_count):
    independent_variable = {
        "var_name": "v3d_freq_min",
        "proper_name": "GPU Frequency (MHz)"
    }
    block_size = 256
    n_fft = 2 * block_size
    banks = bank_count
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
   for i in [1, 2, 4, 8, 16]:
       plot_bank_count(i)

    
