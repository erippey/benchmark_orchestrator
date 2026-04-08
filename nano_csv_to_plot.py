from automated_csv_to_plot import *

if __name__ == "__main__":
    # independent_variable = {
    #     "var_name": "gpu_freq_max",
    #     "proper_name": "GPU Frequency (Hz)"
    # }
    # kernel_names_and_ops = {
    #     ("Forward FFT Execution Time", "forward", 36700160),
    #     ("Complex Multiply Execution Time", "complex_multiply", 3145728),
    #     ("Inverse FFT Execution Time", "inverse", 36700160)
    # }
    # clvk_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using clvk\N{RIGHTWARDS ARROW}GPU", 
    #                test_name="", graph_name_for_file="graphs/clvk_powersave", kernel_names_and_ops=kernel_names_and_ops, 
    #                total_ops=76646414974, peak_mflops=5270)
    
    independent_variable = {
        "var_name": "cpu_freq_max",
        "proper_name": "CPU Frequency (GHz)"
    }

    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", 1146880),
        ("Complex Multiply Execution Time", "complex_multiply", 98304),
        ("Inverse FFT Execution Time", "inverse", 1146880),
        ("Overlap Add", "overlap_add", 16384)
    }

    fftw_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}CPU", 
                   test_name="fftw_cpu_freq", graph_name_for_file="graphs/fftw_cpu_nano", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)

    
    independent_variable = {
        "var_name": "cpu_freq_max",
        "proper_name": "CPU Frequency (GHz)"
    }

    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", 4587520),
        ("Complex Multiply Execution Time", "complex_multiply", 393216),
        ("Inverse FFT Execution Time", "inverse", 4587520),
        ("Overlap Add", "overlap_add", 65536)
    }

    pocl_graph_cpu = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using PoCL\N{RIGHTWARDS ARROW}CPU", 
                   test_name="pocl_cpu_freq", graph_name_for_file="graphs/pocl_cpu_nano", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)
    
    

    independent_variable = {
        "var_name": "arm_freq_min",
        "proper_name": "CPU Frequency (MHz)"
    }

    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", 1146880),
        ("Complex Multiply Execution Time", "complex_multiply", 98304),
        ("Inverse FFT Execution Time", "inverse", 1146880),
        ("Overlap Add", "overlap_add", 16384)
    }

    fftw_graph_pi_powersave = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}CPU", 
                   test_name="FFTW_arm_freq_powersave", graph_name_for_file="graphs/fftw_powersave", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)
    
    pocl_graph_ps = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using PoCL\N{RIGHTWARDS ARROW}CPU", 
                   test_name="PoCL_arm_freq_powersave", graph_name_for_file="graphs/pocl_powersave", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)
    
    
    normalize_graphs([fftw_graph, pocl_graph_cpu, fftw_graph_pi_powersave, pocl_graph_ps])
    
    
    #normalize_graphs([clvk_graph, fftw_graph_pf, fftw_graph_ps, pocl_graph_ps, pocl_graph_pf])

    fftw_graph.plot()
    pocl_graph_cpu.plot()
    fftw_graph_pi_powersave.plot()
    pocl_graph_ps.plot()


    independent_variable = {
        "var_name": "v3d_freq_min",
        "proper_name": "GPU Frequency (MHz)"
    }
    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", 36700160),
        ("Complex Multiply Execution Time", "complex_multiply", 3145728),
        ("Inverse FFT Execution Time", "inverse", 36700160)
    }
    clvk_graph_pi = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using clvk\N{RIGHTWARDS ARROW}GPU", 
                   test_name="clvk_arm_freq_powersave", graph_name_for_file="graphs/clvk_powersave", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=5270)
    
    independent_variable = {
        "var_name": "gpu_freq_max",
        "proper_name": "GPU Frequency (GHz)"
    }

    clvk_graph_nano = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using clvk\N{RIGHTWARDS ARROW}GPU", 
                   test_name="clvk_gpu_freq", graph_name_for_file="graphs/clvk_nano_gpu", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=5270)

    normalize_graphs([clvk_graph_pi, clvk_graph_nano])

    clvk_graph_nano.plot()
    clvk_graph_pi.plot()