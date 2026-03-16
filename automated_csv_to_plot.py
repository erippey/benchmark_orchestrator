import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

efficiency_normalize = True
performance_normalize = False
power_normalize = True
kernel_normalize = False


class Graph:

    def __init__(self, csv_path, independent_variable, graph_name, graph_name_for_file = None, test_name=None,
                 kernel_names_and_ops=None, power=True, performance=True, total_ops = 0,
                 efficiency=True, kernel_percent_peak=True, peak_mflops=0.0):
        self.csv_path = csv_path
        self.csv = pd.read_csv(csv_path)
        if test_name is not None and "TestName" in self.csv.columns:
            self.csv = self.csv[self.csv["TestName"] == test_name].reset_index(drop=True)
        self.graph_name = graph_name
        if graph_name_for_file is None:
            self.graph_name_for_file = self.graph_name
        else:
            self.graph_name_for_file = graph_name_for_file

        self.kernel_names_and_ops = kernel_names_and_ops
        self.power=power
        self.performance=performance and total_ops > 0
        self.efficiency=efficiency and total_ops > 0
        self.kernel_percent_peak=kernel_percent_peak and kernel_names_and_ops != None and peak_mflops > 0.0
        self.peak_mflops = peak_mflops

        self.independent_variable = independent_variable["var_name"]
        self.independent_variable_proper = independent_variable["proper_name"]

        if self.performance or self.efficiency:
            runtime_s = self.csv["total_exec_ms"] / 1000
            self.csv["mflops"] = (total_ops / runtime_s) / 1e6
        
        if self.efficiency:
            self.csv["mflops_w"] = self.csv["mflops"] / self.csv["run_power_w"]

        if self.kernel_percent_peak:
            for kernel, column_name, ops in self.kernel_names_and_ops:
                runtime_s = self.csv[kernel] / 1000

                mflops = (ops / runtime_s) / 1e6

                col = f"{column_name}_percent_peak"

                self.csv[col] = (mflops / self.peak_mflops) * 100

        self.grouped = self.csv.groupby(self.independent_variable)
        
        self.means = self.grouped.mean(numeric_only=True)
        self.stds = self.grouped.std(numeric_only=True)
        self.x_axis_values = self.means.index.to_numpy()

        self.y_max = None
        self.y_min = None 

        self.additional_plots = {
            'power': [],
            'performance': [],
            'efficiency': [],
            'kernel_percent_peak': []
        }

    def plot(self, label=None):
        if self.power:
            self.plot_power()
        if self.performance:
            self.plot_performance()
        if self.efficiency:
            self.plot_efficiency()
        if self.kernel_percent_peak:
            self.plot_kernel_percent_peak()


            
    def plot_power(self):
        plt.clf()
        fig, ax1 = plt.subplots()
        title = plt.title(f'{self.graph_name}: {self.independent_variable_proper} V AVerage Power Consumption', fontsize=16, wrap=True)
        ax1.set_xlabel(f'{self.independent_variable_proper}', fontsize=14)
        ax1.set_ylabel('Average Power (W)', fontsize=14)
        if self.y_max is not None:
            plt.ylim(top=self.y_max['power'])
        if self.y_min is not None:
            plt.ylim(bottom=self.y_min['power'])
        power = self.means['run_power_w'].to_numpy()
        power_err = self.stds['run_power_w'].to_numpy()
        ax1.grid(True, alpha=0.4)
        plt.errorbar(self.x_axis_values, power, yerr=power_err, marker='o', capsize=4)
        for plot in self.additional_plots['power']:
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"])
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_power.png", dpi=400)

    def plot_performance(self):
        plt.clf()
        fig, ax1 = plt.subplots()
        title = plt.title(f'{self.graph_name}: {self.independent_variable_proper} V Performance', fontsize=16, wrap=True)
        ax1.set_xlabel(f'{self.independent_variable_proper}', fontsize=14)
        ax1.set_ylabel('MFLOPS', fontsize=14)
        if self.y_max is not None:
            plt.ylim(top=self.y_max['performance'])
        if self.y_min is not None:
            plt.ylim(bottom=self.y_min['performance'])
        performance = self.means['mflops'].to_numpy()
        ax1.grid(True, alpha=0.4)
        ax1.plot(self.x_axis_values, performance, marker='o')
        for plot in self.additional_plots['performance']:
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"])
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_performance.png", dpi=400)

    def plot_efficiency(self):
        plt.clf()
        fig, ax1 = plt.subplots()
        title = plt.title(f'{self.graph_name}: {self.independent_variable_proper} V Efficiency', fontsize=16, wrap=True)
        ax1.set_xlabel(f'{self.independent_variable_proper}', fontsize=14)
        ax1.set_ylabel('MFLOPS/W', fontsize=14)
        if self.y_max is not None:
            plt.ylim(top=self.y_max['efficiency'])
        if self.y_min is not None:
            plt.ylim(bottom=self.y_min['efficiency'])
        efficiency = self.means['mflops_w'].to_numpy()
        ax1.grid(True, alpha=0.4)
        ax1.plot(self.x_axis_values, efficiency, marker='o')
        for plot in self.additional_plots['efficiency']:
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"])
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_efficiency.png", dpi=400)

    def plot_kernel_percent_peak(self):
        plt.clf()
        line_types = ['-.', '--', (0, (5, 10)), ':', '-.', '--', (0, (5, 10)), ':']
        line_color= ['cornflowerblue', 'mediumseagreen', 'darkorange', 'red']
        fig, ax1 = plt.subplots()
        title = plt.title(f'{self.graph_name}: {self.independent_variable_proper} V Efficiency', fontsize=16, wrap=True)
        ax1.set_xlabel(f'{self.independent_variable_proper}', fontsize=14)
        ax1.set_ylabel('% of Peak Measured Flops', fontsize=14)
        for i, knao in enumerate(self.kernel_names_and_ops):
            _, column_name, _ = knao
            performance_over_peak = self.means[f"{column_name}_percent_peak"].to_numpy()
            plt.plot(self.x_axis_values, performance_over_peak, linestyle=line_types[i], label=column_name.replace("_", " "), color=line_color[i])
        for plot in self.additional_plots['kernel_percent_peak']:
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"])
        ax1.grid(True, alpha=0.4)
        plt.legend(title='Kernel')
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_kernel_percent_peak.png", dpi=400)




    def clear_normalization(self):
        self.y_max = None
        self.y_min = None



    def add_plot(self, graph_type, x_axis, y_axis, err=None, color=None, label=None):
        self.additional_plots[graph_type].append({
            "x_axis": x_axis,
            "y_axis": y_axis,
            "err": err,
            "color": color,
            "label": label
        })
    
    def add_plot(self, graph, graph_type, include_err=False, color=None, label=None):
        if graph_type == 'power':
            y_axis = graph.means['run_power_w']
            if include_err:
                err = graph.stds['run_power_w']
            else:
                err = None
        if graph_type == 'performance':
            y_axis = graph.means['mflops']
            if include_err:
                err = graph.stds['mflops']
            else:
                err = None
        if graph_type == 'efficiency':
            y_axis = graph.means['mflops_w']
            if include_err:
                err = graph.stds['mflops_w']
            else:
                err = None
        if graph_type == 'kernel_percent_peak':
            # I haven't figured this one out yet, and I also don't care to
            return

        self.additional_plots[graph_type].append({
            "x_axis": graph.x_axis_values,
            "y_axis": y_axis,
            "err": err,
            "color": color,
            "label": label
        })
    
def normalize_graphs(graphs):

    global_max = {}
    global_min = {}

    for g in graphs:

        if g.power:
            vals = g.means["run_power_w"]
            global_max["power"] = max(global_max.get("power", -float("inf")), vals.max())
            global_min["power"] = min(global_min.get("power", float("inf")), vals.min())

        if g.performance:
            vals = g.means["mflops"]
            global_max["performance"] = max(global_max.get("performance", -float("inf")), vals.max())
            global_min["performance"] = min(global_min.get("performance", float("inf")), vals.min())

        if g.efficiency:
            vals = g.means["mflops_w"]
            global_max["efficiency"] = max(global_max.get("efficiency", -float("inf")), vals.max())
            global_min["efficiency"] = min(global_min.get("efficiency", float("inf")), vals.min())

    for g in graphs:
        g.y_max = global_max
        g.y_min = global_min



    
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

    pocl_graph = Graph("aggregated_csv/aggregated_results.csv", independent_variable, "Overlap Add using PoCL\N{RIGHTWARDS ARROW}CPU", 
                   test_name="PoCL_arm_freq_powersave", graph_name_for_file="graphs/pocl_powersave", kernel_names_and_ops=kernel_names_and_ops, 
                   total_ops=76646414974, peak_mflops=38374)
    
    normalize_graphs([clvk_graph, fftw_graph_pf, fftw_graph_ps, pocl_graph])

    clvk_graph.plot()
    fftw_graph_pf.plot()
    fftw_graph_ps.plot()
    pocl_graph.plot()
    
