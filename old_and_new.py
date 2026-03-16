import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from automated_csv_to_plot import Graph

efficiency_normalize = False
performance_normalize = False
power_normalize = False
kernel_normalize = False


# ============ Data Section ============
samples = 21368782 * 20

df = pd.read_csv("aggregated_csv/clvk.csv")

# -------- Fix merged columns --------
cols_to_fill = [
    'v3d_freq (MHz)',
    'GFLOPS/W for entire execution',
    'forward GFLOPS',
    'cm GFLOPS',
    'inverse GFLOPS'
]

df[cols_to_fill] = df[cols_to_fill].ffill()

# -------- Force numeric --------
numeric_cols = [
    'total runtime (20 iters) in ms',
    'Average Power over Execution (W)',
    'GFLOPS/W for entire execution',
    'forward GFLOPS',
    'cm GFLOPS',
    'inverse GFLOPS'
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# -------- Group --------
grouped = df.groupby('v3d_freq (MHz)')

means = grouped.mean(numeric_only=True)
stds  = grouped.std(numeric_only=True)

clvk_frequencies = means.index.to_numpy(dtype=float)

# -------- Extract arrays --------
clvk_ex_time = means['total runtime (20 iters) in ms'].to_numpy()
clvk_ex_time_std = stds['total runtime (20 iters) in ms'].to_numpy()

clvk_power = means['Average Power over Execution (W)'].to_numpy()
clvk_power_std = stds['Average Power over Execution (W)'].to_numpy()

# -------- Samples/sec --------
clvk_samples_s = samples / clvk_ex_time

# Proper error propagation
clvk_samples_s_std = (samples / (clvk_ex_time**2)) * clvk_ex_time_std

# -------- MFLOPS/W --------
clvk_mflops_w = means['GFLOPS/W for entire execution'].to_numpy() * 1000
clvk_mflops_w_std = stds['GFLOPS/W for entire execution'].to_numpy() * 1000

# -------- Kernel GFLOPS % of peak --------
clvk_peak_f = 5.27

clvk_forward_gflops = means['forward GFLOPS'].to_numpy()
clvk_cm_gflops = means['cm GFLOPS'].to_numpy()
clvk_inverse_gflops = means['inverse GFLOPS'].to_numpy()

clvk_forward_percent = (clvk_forward_gflops / clvk_peak_f) * 100
clvk_cm_percent      = (clvk_cm_gflops      / clvk_peak_f) * 100
clvk_inverse_percent = (clvk_inverse_gflops / clvk_peak_f) * 100


df = pd.read_csv("aggregated_csv/fftw.csv")

# -------- Fix merged columns --------
cols_to_fill = [
    'arm_freq (MHz)',
    'GFLOPS/W for entire execution',
    'forward GFLOPS',
    'cm GFLOPS',
    'inverse GFLOPS',
    'oa GFLOPS'
]

df[cols_to_fill] = df[cols_to_fill].ffill()

# -------- Force numeric --------
numeric_cols = [
    'total runtime (20 iters) in ms',
    'Average Power over Execution (W)',
    'GFLOPS/W for entire execution',
    'forward GFLOPS',
    'cm GFLOPS',
    'inverse GFLOPS'
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# -------- Group --------
grouped = df.groupby('arm_freq (MHz)')

means = grouped.mean(numeric_only=True)
stds  = grouped.std(numeric_only=True)

fftw_frequencies = means.index.to_numpy(dtype=float)

# -------- Extract arrays --------
fftw_ex_time = means['total runtime (20 iters) in ms'].to_numpy()
fftw_ex_time_std = stds['total runtime (20 iters) in ms'].to_numpy()

fftw_power = means['Average Power over Execution (W)'].to_numpy()
fftw_power_std = stds['Average Power over Execution (W)'].to_numpy()

# -------- Samples/sec --------
fftw_samples_s = samples / fftw_ex_time

# Proper error propagation
fftw_samples_s_std = (samples / (fftw_ex_time**2)) * fftw_ex_time_std

# -------- MFLOPS/W --------
fftw_mflops_w = means['GFLOPS/W for entire execution'].to_numpy() * 1000
fftw_mflops_w_std = stds['GFLOPS/W for entire execution'].to_numpy() * 1000

# -------- Kernel GFLOPS % of peak --------
fftw_peak_f = 38.374

fftw_forward_gflops = means['forward GFLOPS'].to_numpy()
fftw_cm_gflops = means['cm GFLOPS'].to_numpy()
fftw_inverse_gflops = means['inverse GFLOPS'].to_numpy()
fftw_ola_gflops = means['oa GFLOPS'].to_numpy()

fftw_forward_percent = (fftw_forward_gflops / fftw_peak_f) * 100
fftw_cm_percent      = (fftw_cm_gflops      / fftw_peak_f) * 100
fftw_inverse_percent = (fftw_inverse_gflops / fftw_peak_f) * 100
fftw_ola_percent = (fftw_ola_gflops / fftw_peak_f) * 100



df = pd.read_csv("aggregated_csv/pocl.csv")

# -------- Fix merged columns --------
cols_to_fill = [
    'arm_freq (MHz)',
    'GFLOPS/W for entire execution',
    'forward GFLOPS',
    'cm GFLOPS',
    'inverse GFLOPS',
    'oa GFLOPS'
]

df[cols_to_fill] = df[cols_to_fill].ffill()

# -------- Force numeric --------
numeric_cols = [
    'total runtime (20 iters) in ms',
    'Average Power over Execution (W)',
    'GFLOPS/W for entire execution',
    'forward GFLOPS',
    'cm GFLOPS',
    'inverse GFLOPS'
]

df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# -------- Group --------
grouped = df.groupby('arm_freq (MHz)')

means = grouped.mean(numeric_only=True)
stds  = grouped.std(numeric_only=True)

pocl_frequencies = means.index.to_numpy(dtype=float)

# -------- Extract arrays --------
pocl_ex_time = means['total runtime (20 iters) in ms'].to_numpy()
pocl_ex_time_std = stds['total runtime (20 iters) in ms'].to_numpy()

pocl_power = means['Average Power over Execution (W)'].to_numpy()
pocl_power_std = stds['Average Power over Execution (W)'].to_numpy()

# -------- Samples/sec --------
pocl_samples_s = samples / pocl_ex_time

# Proper error propagation
pocl_samples_s_std = (samples / (pocl_ex_time**2)) * pocl_ex_time_std

# -------- MFLOPS/W --------
pocl_mflops_w = means['GFLOPS/W for entire execution'].to_numpy() * 1000
pocl_mflops_w_std = stds['GFLOPS/W for entire execution'].to_numpy() * 1000

# -------- Kernel GFLOPS % of peak --------
pocl_peak_f = 38.374

pocl_forward_gflops = means['forward GFLOPS'].to_numpy()
pocl_cm_gflops = means['cm GFLOPS'].to_numpy()
pocl_inverse_gflops = means['inverse GFLOPS'].to_numpy()
pocl_ola_gflops = means['oa GFLOPS'].to_numpy()

pocl_forward_percent = (pocl_forward_gflops / pocl_peak_f) * 100
pocl_cm_percent      = (pocl_cm_gflops      / pocl_peak_f) * 100
pocl_inverse_percent = (pocl_inverse_gflops / pocl_peak_f) * 100
pocl_ola_percent = (pocl_ola_gflops / pocl_peak_f) * 100


efficiency_y_max = max([max(clvk_mflops_w), max(fftw_mflops_w), max(pocl_mflops_w)])
perfomance_y_max = max([max(clvk_samples_s), max(fftw_samples_s), max(pocl_samples_s)])
power_y_max = max([max(clvk_power), max(fftw_power), max(pocl_power)])
percent_y_max = max([max(clvk_forward_gflops), max(clvk_cm_gflops), max(clvk_inverse_gflops),
        max(fftw_forward_gflops), max(fftw_cm_gflops), max(fftw_inverse_gflops), max(fftw_ola_gflops),
        max(pocl_forward_gflops), max(pocl_cm_gflops), max(pocl_inverse_gflops), max(pocl_ola_gflops)])

efficiency_y_min = min([min(clvk_mflops_w), min(fftw_mflops_w), min(pocl_mflops_w)])
perfomance_y_min = min([min(clvk_samples_s), min(fftw_samples_s), min(pocl_samples_s)])
power_y_min = min([min(clvk_power), min(fftw_power), min(pocl_power)])
percent_y_min = min([min(clvk_forward_gflops), min(clvk_cm_gflops), min(clvk_inverse_gflops),
        min(fftw_forward_gflops), min(fftw_cm_gflops), min(fftw_inverse_gflops), min(fftw_ola_gflops),
        min(pocl_forward_gflops), min(pocl_cm_gflops), min(pocl_inverse_gflops), min(pocl_ola_gflops)])



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
    "var_name": "arm_freq",
    "proper_name": "CPU Frequency (MHz)"
}

kernel_names_and_ops = {
    ("Forward FFT Execution Time", "forward", 1146880),
    ("Complex Multiply Execution Time", "complex_multiply", 98304),
    ("Inverse FFT Execution Time", "inverse", 1146880),
    ("Overlap Add", "overlap_add", 16384)
}
    
fftw_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using FFTW\N{RIGHTWARDS ARROW}CPU", 
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

pocl_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using PoCL\N{RIGHTWARDS ARROW}CPU", 
                test_name="PoCL_arm_freq_powersave", graph_name_for_file="graphs/pocl_powersave", kernel_names_and_ops=kernel_names_and_ops, 
                total_ops=76646414974, peak_mflops=38374)

clvk_graph.add_plot("efficiency", clvk_frequencies, clvk_mflops_w, color="coral", label="03-02-2026", marker="o")
clvk_graph.add_plot("power", clvk_frequencies, clvk_power, err=clvk_power_std, color="coral", label="03-02-2026", marker="o")

clvk_graph.plot_power(label="03-16-2026", legend_title="Date Presented")
clvk_graph.plot_efficiency(label="03-16-2026", legend_title="Date Presented")

fftw_graph.add_plot("efficiency", fftw_frequencies, fftw_mflops_w, color="coral", label="03-02-2026", marker="o")
fftw_graph.add_plot("power", fftw_frequencies, fftw_power, err=fftw_power_std, color="coral", label="03-02-2026", marker="o")

fftw_graph.plot_power(label="03-16-2026", legend_title="Date Presented")
fftw_graph.plot_efficiency(label="03-16-2026", legend_title="Date Presented")