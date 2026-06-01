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
                 min_runs_after_drop=2,
                 opp=None):

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

        self.opp = opp

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

        runtime_s = self.csv["total_exec_ms"] / 1000
        self.csv["mflops"] = (total_ops / runtime_s) / 1e6

        self.csv["latency"] = self.csv["conv_avg_ms"]


        self.csv["mflops_w"] = self.csv["mflops"] / self.csv["run_power_w"]

        if self.kernel_names_and_ops is not None:
            for kernel, column_name, ops in self.kernel_names_and_ops:
                runtime_s = self.csv[kernel] / 1000
                mflops = (ops / runtime_s) / 1e6
                col = f"{column_name}_percent_peak"
                self.csv[col] = (mflops / self.peak_mflops) * 100

        self.grouped = self.csv.groupby(self.independent_variable)

        self.means = self.grouped.mean(numeric_only=True)
        self.stds = self.grouped.std(numeric_only=True)
        self.x_axis_values = self.means.index.to_numpy()

        if self.opp is not None:
            self.means["estimate_rt"] = [self.estimate_execution_time(i) for i in self.means.index]
            self.means["estimate_perf"] = (total_ops / (self.means["estimate_rt"] / 1000)) / 1e6
            print(f"Actual ac values for {test_name}: ")
            self.means["estimate_pd"] = [self.estimate_power_draw(i) for i in self.means.index]
            print("")
            self.means["estimate_eff"] = self.means["estimate_perf"] / self.means["estimate_pd"]


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
    
    def plot(self, label=None, show_estimate=False):
        self.plot_runtime(show_estimate=show_estimate)
        if self.power:
            self.plot_power(show_estimate=show_estimate)
        if self.performance:
            self.plot_performance(show_estimate=show_estimate)
        if self.efficiency:
            self.plot_efficiency(show_estimate=show_estimate)
        if self.kernel_percent_peak:
            self.plot_kernel_percent_peak()
        if self.latency:
            self.plot_latency()

            
    def plot_power(self, label=None, legend_title=None, show_estimate=False):
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
        if (show_estimate):
            plt.errorbar(self.x_axis_values, self.means["estimate_pd"], color="black", label="Estimated Power Draw")
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

    def plot_runtime(self, label=None, legend_title=None, show_estimate=False):
        plt.clf()
        show_label = label is not None and legend_title is not None
        fig, ax1 = plt.subplots()
        title = plt.title(f'{self.graph_name}: {self.independent_variable_proper} V Runtime', fontsize=16, wrap=True)
        ax1.set_xlabel(f'{self.independent_variable_proper}', fontsize=14)
        ax1.set_ylabel('Runtime (ms)', fontsize=14)
        if self.y_max is not None:
            plt.ylim(top=self.y_max['runtime'])
        if self.y_min is not None:
            plt.ylim(bottom=self.y_min['runtime'])
        performance = self.means['total_exec_ms'].to_numpy()
        ax1.grid(True, alpha=0.4)
        ax1.plot(self.x_axis_values, performance, marker='o', label=label)
        
        if (show_estimate):
            plt.errorbar(self.x_axis_values, self.means["estimate_rt"], color="black", label="Estimated Runtime")
        for plot in self.additional_plots['performance']:
            if plot["label"] is not None:
                show_label = True
            plt.errorbar(plot["x_axis"], plot["y_axis"], yerr=plot["err"], color=plot["color"], label=plot["label"], marker=plot["marker"])
        if show_label:
            plt.legend(title=legend_title)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        title.set_y(1.05)
        plt.savefig(f"{self.graph_name_for_file}_runtime.png", dpi=400)
        plt.close()

    def plot_performance(self, label=None, legend_title=None, show_estimate=False):
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
        
        if (show_estimate):
            plt.errorbar(self.x_axis_values, self.means["estimate_perf"], color="black", label="Estimated Performance")
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

    def plot_efficiency(self, label=None, legend_title=None, show_estimate=False):
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
        plt.errorbar(self.x_axis_values, self.means["estimate_eff"], color="black", label="Estimated Efficiency")
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
    



    def _require_mean_column(self, column_name):
        if column_name not in self.means.columns:
            raise KeyError(
                f"Column {column_name!r} is required, but it is not present in self.means. "
                f"Available columns: {list(self.means.columns)}"
            )


    def _get_min_max_independent_values(self):
        if len(self.means.index) < 2:
            raise ValueError(
                "At least two independent-variable values are required for endpoint estimation."
            )

        try:
            x_values = self.means.index.to_numpy(dtype=float)
        except ValueError as exc:
            raise TypeError(
                f"Independent variable {self.independent_variable!r} must be numeric "
                "for frequency-based estimation."
            ) from exc

        min_pos = x_values.argmin()
        max_pos = x_values.argmax()

        min_key = self.means.index[min_pos]
        max_key = self.means.index[max_pos]

        min_freq = float(x_values[min_pos])
        max_freq = float(x_values[max_pos])

        if min_freq <= 0 or max_freq <= 0:
            raise ValueError("Frequencies must be positive.")

        if min_freq == max_freq:
            raise ValueError("Minimum and maximum frequencies are identical.")

        return min_key, max_key, min_freq, max_freq


    def _lookup_voltage(self, opp, freq):
        """
        Look up voltage for a frequency.

        This allows small int/float representation differences, e.g.
        600000000 vs 600000000.0.
        """
        if freq in opp:
            return float(opp[freq])

        freq_f = float(freq)

        for key, voltage in opp.items():
            try:
                key_f = float(key)
            except (TypeError, ValueError):
                continue

            if key_f == freq_f:
                return float(voltage)

        raise KeyError(
            f"Frequency {freq!r} was not found in the OPP table. "
            f"Available frequencies: {sorted(opp.keys())}"
        )


    def _lookup_mean_row_key(self, freq):
        """
        Find the matching self.means index key for a requested frequency.

        This is used because the grouped index may contain ints, floats,
        numpy scalar types, etc.
        """
        if freq in self.means.index:
            return freq

        freq_f = float(freq)

        for key in self.means.index:
            try:
                key_f = float(key)
            except (TypeError, ValueError):
                continue

            if key_f == freq_f:
                return key

        raise KeyError(
            f"Frequency {freq!r} was not found in measured data for "
            f"{self.independent_variable!r}. Available values: {list(self.means.index)}"
        )


    def estimate_execution_time(self, freq):
        """
        Estimate total execution time using a two-point inverse-frequency model:

            time_ms = a / freq + b

        The model is fit using the minimum and maximum measured independent-variable
        frequencies.
        """
        self._require_mean_column("total_exec_ms")

        min_key, max_key, min_freq, max_freq = self._get_min_max_independent_values()

        min_time_ms = float(self.means.loc[min_key, "total_exec_ms"])
        max_time_ms = float(self.means.loc[max_key, "total_exec_ms"])

        # Solve:
        #   min_time = a / min_freq + b
        #   max_time = a / max_freq + b
        denom = (1.0 / min_freq) - (1.0 / max_freq)

        if denom == 0:
            raise ValueError("Could not fit inverse-frequency model because denominator is zero.")

        a = (min_time_ms - max_time_ms) / denom
        b = min_time_ms - (a / min_freq)

        freq_f = float(freq)

        if freq_f <= 0:
            raise ValueError("freq must be positive.")

        return (a / freq_f) + b


    def estimate_power_draw(self, freq, opp=None):
        """
        Estimate run power using:

            power_w = Ac * V^2 * freq + idle_power_w

        Ac is estimated at the minimum and maximum measured frequencies using:

            Ac = (run_power_w - idle_power_w) / (V^2 * freq)

        Then the two Ac values are averaged.

        Requirements:
        - OPP table must contain min frequency, max frequency, and requested freq.
        - measured data must contain idle_power_w for requested freq.
        - measured data must contain run_power_w and idle_power_w for min/max freq.
        """
        if opp is None:
            opp = getattr(self, "opp", None)

        if opp is None:
            raise ValueError(
                "No OPP table was provided. Pass opp=... to estimate_power_draw(), "
                "or set self.opp when constructing the Graph."
            )

        self._require_mean_column("run_power_w")
        self._require_mean_column("idle_power_w")

        min_key, max_key, min_freq, max_freq = self._get_min_max_independent_values()

        min_voltage = self._lookup_voltage(opp, min_freq)
        max_voltage = self._lookup_voltage(opp, max_freq)

        min_run_power = float(self.means.loc[min_key, "run_power_w"])
        max_run_power = float(self.means.loc[max_key, "run_power_w"])

        idle_power = float(self.means["idle_power_w"].mean())
        min_idle_power = float(self.means.loc[min_key, "idle_power_w"])
        max_idle_power = float(self.means.loc[max_key, "idle_power_w"])

        min_dynamic_power = min_run_power - idle_power
        max_dynamic_power = max_run_power - idle_power

        voltage_range = max_freq - min_freq
        voltage_weight = (max_freq - freq) / voltage_range

        if (voltage_weight > 1) or (voltage_weight < 0):
            raise ValueError(
                "frequency should be between minimum and maximum frequencies"
            )

        if min_dynamic_power < 0 or max_dynamic_power < 0:
            raise ValueError(
                "Measured run_power_w is below idle_power_w at one of the endpoints. "
                "Cannot estimate a valid positive dynamic-power coefficient."
            )

        min_den = (min_voltage ** 2) * min_freq
        max_den = (max_voltage ** 2) * max_freq

        if min_den == 0 or max_den == 0:
            raise ValueError("Voltage and frequency must be nonzero for power estimation.")

        ac_min = min_dynamic_power / min_den
        ac_max = max_dynamic_power / max_den

        ac_avg = (ac_min + ac_max) / 2.0
        estimate_ac = ac_min * voltage_weight + ac_max * (1 - voltage_weight)

        freq_key = self._lookup_mean_row_key(freq)
        freq_f = float(freq)
        voltage = self._lookup_voltage(opp, freq_f)

        
        actual_power = float(self.means.loc[freq, "run_power_w"])
        cur_dynamic_power = actual_power - idle_power
        actual_den = (voltage ** 2) * freq
        actual_ac = cur_dynamic_power / actual_den
        print(f"({actual_ac}, {estimate_ac}), ", end="")
        

        # idle_power = float(self.means.loc[freq_key, "idle_power_w"])

        return estimate_ac * (voltage ** 2) * freq_f + idle_power
    


def normalize_graphs(graphs):

    global_max = {}
    global_min = {}

    for g in graphs:

        vals = g.means["total_exec_ms"]
        global_max["runtime"] = max(global_max.get("rintime", -float("inf")), vals.max())
        global_min["runtime"] = min(global_max.get("rintime", -float("inf")), vals.min())

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
