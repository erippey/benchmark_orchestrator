import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

efficiency_normalize = True
performance_normalize = False
power_normalize = True
kernel_normalize = False

class Graph:

    def __init__(self, csv_path, independent_variable, graph_name, graph_name_for_file=None,
                 test_name=None, kernel_names_and_ops=None, total_ops=0, peak_mflops=0.0,
                 efficiency=True, kernel_percent_peak=True, power=True, performance=True,
                 latency=True,
                 drop_lowest_n=0, drop_highest_n=0,
                 drop_by="total_exec_ms",
                 outlier_group_cols=None,
                 min_runs_after_drop=2):

        self.csv_path = csv_path
        self.csv = pd.read_csv(csv_path)

        if test_name is not None and "TestName" in self.csv.columns:
            self.csv = self.csv[self.csv["TestName"] == test_name].reset_index(drop=True)

        self.graph_name = graph_name
        self.graph_name_for_file = graph_name if graph_name_for_file is None else graph_name_for_file

        self.kernel_names_and_ops = kernel_names_and_ops
        self.power = power
        self.latency = latency
        self.performance = performance and total_ops > 0
        self.efficiency = efficiency and total_ops > 0
        self.kernel_percent_peak = (
            kernel_percent_peak
            and kernel_names_and_ops is not None
            and peak_mflops > 0.0
        )
        self.peak_mflops = peak_mflops

        self.independent_variable = independent_variable["var_name"]
        self.independent_variable_proper = independent_variable["proper_name"]

        # Keep a copy of the filtered-but-untrimmed data.
        self.csv_untrimmed = self.csv.copy()
        self.dropped_outlier_rows = pd.DataFrame()
        self.outlier_drop_summary = pd.DataFrame()

        if drop_lowest_n > 0 or drop_highest_n > 0:
            group_cols = outlier_group_cols
            if group_cols is None:
                group_cols = [self.independent_variable]
            elif isinstance(group_cols, str):
                group_cols = [group_cols]

            self.csv, self.dropped_outlier_rows, self.outlier_drop_summary = (
                self._drop_extreme_rows_by_group(
                    df=self.csv,
                    group_cols=group_cols,
                    value_col=drop_by,
                    drop_lowest_n=drop_lowest_n,
                    drop_highest_n=drop_highest_n,
                    min_runs_after_drop=min_runs_after_drop,
                )
            )

        if self.performance or self.efficiency:
            runtime_s = self.csv["total_exec_ms"] / 1000
            self.csv["mflops"] = (total_ops / runtime_s) / 1e6

        if self.latency:
            self.csv["latency"] = self.csv["conv_avg_ms"]

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
            'kernel_percent_peak': [],
            'latency': []
        }

    @staticmethod
    def _drop_extreme_rows_by_group(df, group_cols, value_col,
                                    drop_lowest_n=0, drop_highest_n=0,
                                    min_runs_after_drop=2):
        if drop_lowest_n < 0 or drop_highest_n < 0:
            raise ValueError("drop_lowest_n and drop_highest_n must be >= 0")

        missing = [col for col in group_cols + [value_col] if col not in df.columns]
        if missing:
            raise KeyError(f"Cannot drop outliers. Missing columns: {missing}")

        drop_reasons = {}
        summary_rows = []

        for group_key, group in df.groupby(group_cols, dropna=False, sort=False):
            valid = group[group[value_col].notna()].sort_values(value_col, kind="mergesort")
            n_valid = len(valid)

            requested_drops = drop_lowest_n + drop_highest_n

            if requested_drops == 0 or n_valid == 0:
                continue

            if n_valid - requested_drops < min_runs_after_drop:
                raise ValueError(
                    f"Outlier trimming would leave too few runs for group {group_key}. "
                    f"Valid runs: {n_valid}, requested drops: {requested_drops}, "
                    f"minimum after drop: {min_runs_after_drop}."
                )

            low_indices = valid.head(drop_lowest_n).index.tolist() if drop_lowest_n else []

            # Avoid double-dropping the same row when a group is small.
            remaining = valid.drop(index=low_indices)

            high_indices = (
                remaining.tail(drop_highest_n).index.tolist()
                if drop_highest_n else []
            )

            for idx in low_indices:
                drop_reasons[idx] = f"lowest_{value_col}"

            for idx in high_indices:
                drop_reasons[idx] = f"highest_{value_col}"

            if not isinstance(group_key, tuple):
                group_key = (group_key,)

            summary = dict(zip(group_cols, group_key))
            summary.update({
                "value_col": value_col,
                "n_before": len(group),
                "n_valid_for_drop": n_valid,
                "dropped_lowest": len(low_indices),
                "dropped_highest": len(high_indices),
                "n_after": len(group) - len(low_indices) - len(high_indices),
            })
            summary_rows.append(summary)

        drop_indices = list(drop_reasons.keys())

        dropped = df.loc[drop_indices].copy()
        if not dropped.empty:
            dropped["outlier_drop_reason"] = dropped.index.map(drop_reasons)

        trimmed = df.drop(index=drop_indices).reset_index(drop=True)
        summary_df = pd.DataFrame(summary_rows)

        return trimmed, dropped, summary_df
    def plot(self, label=None):
        if self.power:
            self.plot_power()
        if self.performance:
            self.plot_performance()
        if self.efficiency:
            self.plot_efficiency()
        if self.kernel_percent_peak:
            self.plot_kernel_percent_peak()
        if self.latency:
            self.plot_latency()


            
    def plot_power(self, label=None, legend_title=None):
        plt.clf()
        show_label = label is not None and legend_title is not None
        fig, ax1 = plt.subplots()
        title = plt.title(f'{self.graph_name}: {self.independent_variable_proper} V Average Power Consumption', fontsize=16, wrap=True)
        ax1.set_xlabel(f'{self.independent_variable_proper}', fontsize=14)
        ax1.set_ylabel('Average Power (W)', fontsize=14)
        if self.y_max is not None:
            plt.ylim(top=self.y_max['power'])
        if self.y_min is not None:
            plt.ylim(bottom=self.y_min['power'])
        power = self.means['run_power_w'].to_numpy()
        power_err = self.stds['run_power_w'].to_numpy()
        ax1.grid(True, alpha=0.4)
        plt.errorbar(self.x_axis_values, power, yerr=power_err, marker='o', capsize=4, label=label)
        for plot in self.additional_plots['power']:
            if plot["label"] is not None:
                show_label = True
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"], marker=plot["marker"], capsize=4)
        if show_label:
            plt.legend(title=legend_title)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_power.png", dpi=400)
        plt.close()

    def plot_performance(self, label=None, legend_title=None):
        plt.clf()
        show_label = label is not None and legend_title is not None
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
        ax1.plot(self.x_axis_values, performance, marker='o', label=label)
        for plot in self.additional_plots['performance']:
            if plot["label"] is not None:
                show_label = True
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"], marker=plot["marker"])
        if show_label:
            plt.legend(title=legend_title)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_performance.png", dpi=400)
        plt.close()

    def plot_latency(self, label=None, legend_title=None):
        plt.clf()
        show_label = label is not None and legend_title is not None
        fig, ax1 = plt.subplots()
        title = plt.title(f'{self.graph_name}: {self.independent_variable_proper} V Latency', fontsize=16, wrap=True)
        ax1.set_xlabel(f'{self.independent_variable_proper}', fontsize=14)
        ax1.set_ylabel('Latency (ms)', fontsize=14)
        if self.y_max is not None:
            plt.ylim(top=self.y_max['latency'])
        if self.y_min is not None:
            plt.ylim(bottom=self.y_min['latency'])
        latency = self.means['latency'].to_numpy()
        ax1.grid(True, alpha=0.4)
        ax1.plot(self.x_axis_values, latency, marker='o', label=label)
        for plot in self.additional_plots['latency']:
            if plot["label"] is not None:
                show_label = True
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"], marker=plot["marker"])
        if show_label:
            plt.legend(title=legend_title) 
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_latency.png", dpi=400)
        plt.close()

    def plot_efficiency(self, label=None, legend_title=None):
        plt.clf()
        show_label = label is not None and legend_title is not None
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
        ax1.plot(self.x_axis_values, efficiency, marker='o', label=label)
        for plot in self.additional_plots['efficiency']:
            if plot["label"] is not None:
                show_label = True
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"], marker=plot["marker"], capsize=4)
        if show_label:
            plt.legend(title=legend_title)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_efficiency.png", dpi=400)
        plt.close()

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
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"], marker=plot["marker"], capsize=4)
        ax1.grid(True, alpha=0.4)
        plt.legend(title='Kernel')
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_kernel_percent_peak.png", dpi=400)
        plt.close()





    def clear_normalization(self):
        self.y_max = None
        self.y_min = None



    def add_plot(self, graph_type, x_axis, y_axis, err=None, color=None, label=None, marker=None):
        self.additional_plots[graph_type].append({
            "x_axis": x_axis,
            "y_axis": y_axis,
            "err": err,
            "color": color,
            "label": label,
            "marker": marker
        })
    
    def add_graph_plot(self, graph, graph_type, include_err=False, color=None, label=None, marker=None):
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
        if graph_type == 'latency':
            y_axis = graph.means['latency']
            if include_err:
                err = graph.stds['latency']
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
            "label": label,
            "marker": marker
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
        if g.latency:
            vals = g.means["latency"]
            global_max["latency"] = max(global_max.get("latency", -float("inf")), vals.max())
            global_min["latency"] = min(global_min.get("latency", float("inf")), vals.min())

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
    
