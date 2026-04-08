from automated_csv_to_plot import *

if __name__ == "__main__":
    independent_variable = {
        "var_name": "v3d_freq_min",
        "proper_name": "GPU Frequency (MHz)"
    }
    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", 36700160),
        ("Complex Multiply Execution Time", "complex_multiply", 3145728),
        ("Inverse FFT Execution Time", "inverse", 36700160)
    }
    clvk_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using clvk\N{RIGHTWARDS ARROW}GPU", 
                   test_name="clvk_arm_freq_powersave", graph_name_for_file="graphs/clvk_powersave", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=5270)
    
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

    fftw_graph_ps = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}CPU", 
                   test_name="FFTW_arm_freq_powersave", graph_name_for_file="graphs/fftw_powersave", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)

    independent_variable = {
        "var_name": "arm_freq",
        "proper_name": "CPU Frequency (MHz)"
    }
    
    fftw_graph_pf = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}CPU", 
                   test_name="FFTW_arm_freq_performance", graph_name_for_file="graphs/fftw_performance", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)
    
    independent_variable = {
        "var_name": "arm_freq_min",
        "proper_name": "CPU Frequency (MHz)"
    }

    kernel_names_and_ops = {
        ("Forward FFT Execution Time", "forward", 4587520),
        ("Complex Multiply Execution Time", "complex_multiply", 393216),
        ("Inverse FFT Execution Time", "inverse", 4587520),
        ("Overlap Add", "overlap_add", 65536)
    }

    pocl_graph_ps = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using PoCL\N{RIGHTWARDS ARROW}CPU", 
                   test_name="PoCL_arm_freq_powersave", graph_name_for_file="graphs/pocl_powersave", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)
    
    
    independent_variable = {
        "var_name": "arm_freq",
        "proper_name": "CPU Frequency (MHz)"
    }

    pocl_graph_pf = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using PoCL\N{RIGHTWARDS ARROW}CPU", 
                   test_name="PoCL_arm_freq_performance", graph_name_for_file="graphs/pocl_performance", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)

    
    normalize_graphs([clvk_graph, fftw_graph_pf, fftw_graph_ps, pocl_graph_ps, pocl_graph_pf])

    clvk_graph.plot()
    fftw_graph_pf.plot()
    fftw_graph_ps.plot()
    pocl_graph_ps.plot()

    fftw_graph_ps.add_graph_plot(fftw_graph_pf, "power", include_err=True, color="salmon", label="Performance", marker="o")
    fftw_graph_ps.add_graph_plot(fftw_graph_pf, "efficiency", include_err=True, color="salmon", label="Performance", marker="o")
    fftw_graph_ps.add_graph_plot(fftw_graph_pf, "performance", color="salmon", label="Performance", marker="o" )
    fftw_graph_ps.plot_power(label="powersave", legend_title="Governor")
    fftw_graph_ps.plot_efficiency(label="powersave", legend_title="Governor")
    fftw_graph_ps.plot_performance(label="powersave", legend_title="Governor")
    

    pocl_graph_ps.add_graph_plot(pocl_graph_pf, "power", include_err=True, color="salmon", label="Performance", marker="o")
    pocl_graph_ps.add_graph_plot(pocl_graph_pf, "efficiency", include_err=True, color="salmon", label="Performance", marker="o")
    pocl_graph_ps.add_graph_plot(pocl_graph_pf, "performance", color="salmon", label="Performance", marker="o" )
    pocl_graph_ps.plot_power(label="powersave", legend_title="Governor")
    pocl_graph_ps.plot_efficiency(label="powersave", legend_title="Governor")
    pocl_graph_ps.plot_performance(label="powersave", legend_title="Governor")
    
