"""
graph_tool_v2_skeleton.py

A small refactor skeleton for benchmark CSV plotting.

Design goal:
- A benchmark CSV is just data.
- A Metric describes how to derive one column from that data.
- A View/PlotSpec describes how to visualize columns.
- The plotter does not know what "performance" means.

This is intentionally compact. Treat it as a starting point rather than a framework.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence, Optional, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MetricFn = Callable[[pd.DataFrame], pd.Series]
Aggregator = Literal["mean", "median", "min", "max"]
PlotKind = Literal["line", "scatter", "heatmap", "bubble"]


@dataclass(frozen=True)
class Metric:
    """A derived or existing measurement column.

    `fn` receives the full raw dataframe and returns a Series.
    Use this for runtime, ROI runtime, energy, EDP, throughput, FLOPS/W, etc.
    """

    name: str
    label: str
    unit: str
    fn: MetricFn
    higher_is_better: bool = True

    @property
    def axis_label(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label


@dataclass(frozen=True)
class Dimension:
    """A column used as an independent/grouping variable."""

    name: str
    label: str
    unit: str = ""

    @property
    def axis_label(self) -> str:
        return f"{self.label} ({self.unit})" if self.unit else self.label


@dataclass
class TrimSpec:
    """Drop N low/high rows inside each group based on a value column."""

    by: str = "total_exec_ms"
    group_by: Sequence[str] = field(default_factory=list)
    drop_lowest_n: int = 0
    drop_highest_n: int = 0
    min_runs_after_drop: int = 2


@dataclass
class BenchmarkData:
    """Owns raw CSV rows plus derived metric columns."""

    df: pd.DataFrame
    metrics: dict[str, Metric] = field(default_factory=dict)

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        test_name: Optional[str] = None,
        test_col: str = "TestName",
    ) -> "BenchmarkData":
        df = pd.read_csv(path)
        if test_name is not None and test_col in df.columns:
            df = df[df[test_col] == test_name].reset_index(drop=True)
        return cls(df=df)

    def with_metrics(self, metrics: Iterable[Metric]) -> "BenchmarkData":
        out = self.df.copy()
        metric_map = dict(self.metrics)
        for metric in metrics:
            out[metric.name] = metric.fn(out)
            metric_map[metric.name] = metric
        return BenchmarkData(out, metric_map)

    def trimmed(self, spec: TrimSpec) -> "BenchmarkData":
        group_by = list(spec.group_by)
        if not group_by:
            raise ValueError("TrimSpec.group_by must name at least one grouping column")

        missing = [c for c in group_by + [spec.by] if c not in self.df.columns]
        if missing:
            raise KeyError(f"Cannot trim; missing columns: {missing}")

        drop_indices: list[int] = []
        for _, group in self.df.groupby(group_by, sort=False, dropna=False):
            valid = group[group[spec.by].notna()].sort_values(spec.by, kind="mergesort")
            requested = spec.drop_lowest_n + spec.drop_highest_n
            if requested == 0:
                continue
            if len(valid) - requested < spec.min_runs_after_drop:
                raise ValueError(
                    f"Trimming would leave {len(valid) - requested} valid runs; "
                    f"minimum is {spec.min_runs_after_drop}"
                )

            low = valid.head(spec.drop_lowest_n).index.tolist()
            remaining = valid.drop(index=low)
            high = remaining.tail(spec.drop_highest_n).index.tolist()
            drop_indices.extend(low + high)

        return BenchmarkData(self.df.drop(index=drop_indices).reset_index(drop=True), self.metrics)

    def aggregate(
        self,
        group_by: Sequence[str],
        value_cols: Sequence[str],
        *,
        aggregator: Aggregator = "mean",
        include_std: bool = True,
    ) -> pd.DataFrame:
        missing = [c for c in list(group_by) + list(value_cols) if c not in self.df.columns]
        if missing:
            raise KeyError(f"Cannot aggregate; missing columns: {missing}")

        grouped = self.df.groupby(list(group_by), sort=True, dropna=False)
        if aggregator == "mean":
            center = grouped[list(value_cols)].mean(numeric_only=True)
        elif aggregator == "median":
            center = grouped[list(value_cols)].median(numeric_only=True)
        elif aggregator == "min":
            center = grouped[list(value_cols)].min(numeric_only=True)
        elif aggregator == "max":
            center = grouped[list(value_cols)].max(numeric_only=True)
        else:
            raise ValueError(f"Unsupported aggregator: {aggregator}")

        if not include_std:
            return center.reset_index()

        std = grouped[list(value_cols)].std(numeric_only=True).add_suffix("_std")
        return center.join(std).reset_index()


@dataclass(frozen=True)
class PlotSpec:
    kind: PlotKind
    x: str
    y: str
    title: str
    output: str | Path
    color_by: Optional[str] = None       # algorithm, problem_size, device, etc.
    size_by: Optional[str] = None        # for bubble plots, often power_w or problem_size
    facet_by: Optional[str] = None       # simple small-multiple hook; optional extension point
    yerr: Optional[str] = None
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None


class Plotter:
    """Generic renderer. It plots columns; it does not derive metrics."""

    def __init__(self, dimensions: dict[str, Dimension], metrics: dict[str, Metric]):
        self.dimensions = dimensions
        self.metrics = metrics

    def label_for(self, col: str) -> str:
        if col in self.metrics:
            return self.metrics[col].axis_label
        if col in self.dimensions:
            return self.dimensions[col].axis_label
        return col.replace("_", " ")

    def plot(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        if spec.kind == "line":
            self._line(df, spec)
        elif spec.kind == "scatter":
            self._scatter(df, spec)
        elif spec.kind == "heatmap":
            self._heatmap(df, spec)
        elif spec.kind == "bubble":
            self._bubble(df, spec)
        else:
            raise ValueError(f"Unsupported plot kind: {spec.kind}")

    def _finish(self, spec: PlotSpec) -> None:
        plt.title(spec.title, fontsize=15, wrap=True)
        plt.xlabel(spec.xlabel or self.label_for(spec.x), fontsize=13)
        plt.ylabel(spec.ylabel or self.label_for(spec.y), fontsize=13)
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(spec.output, dpi=400)
        plt.close()

    def _line(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        plt.figure()
        if spec.color_by is None:
            ordered = df.sort_values(spec.x)
            plt.errorbar(
                ordered[spec.x],
                ordered[spec.y],
                yerr=ordered[spec.yerr] if spec.yerr else None,
                marker="o",
                capsize=4 if spec.yerr else 0,
            )
        else:
            for label, part in df.groupby(spec.color_by, sort=True):
                ordered = part.sort_values(spec.x)
                plt.errorbar(
                    ordered[spec.x],
                    ordered[spec.y],
                    yerr=ordered[spec.yerr] if spec.yerr else None,
                    marker="o",
                    capsize=4 if spec.yerr else 0,
                    label=str(label),
                )
            plt.legend(title=self.label_for(spec.color_by))
        self._finish(spec)

    def _scatter(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        plt.figure()
        if spec.color_by is None:
            plt.scatter(df[spec.x], df[spec.y])
        else:
            for label, part in df.groupby(spec.color_by, sort=True):
                plt.scatter(part[spec.x], part[spec.y], label=str(label))
            plt.legend(title=self.label_for(spec.color_by))
        self._finish(spec)

    def _bubble(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        if spec.size_by is None:
            raise ValueError("Bubble plots require PlotSpec.size_by")
        plt.figure()
        sizes = df[spec.size_by].astype(float)
        sizes = 50 + 450 * (sizes - sizes.min()) / max(float(sizes.max() - sizes.min()), 1e-12)
        if spec.color_by is None:
            plt.scatter(df[spec.x], df[spec.y], s=sizes, alpha=0.65)
        else:
            for label, part in df.groupby(spec.color_by, sort=True):
                idx = part.index
                plt.scatter(part[spec.x], part[spec.y], s=sizes.loc[idx], alpha=0.65, label=str(label))
            plt.legend(title=self.label_for(spec.color_by))
        self._finish(spec)

    def _heatmap(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        if spec.color_by is None:
            raise ValueError("Heatmap uses PlotSpec.color_by as the second axis column")

        pivot = df.pivot(index=spec.color_by, columns=spec.x, values=spec.y)
        plt.figure()
        plt.imshow(pivot.to_numpy(), aspect="auto", origin="lower")
        plt.colorbar(label=self.label_for(spec.y))
        plt.xticks(np.arange(len(pivot.columns)), pivot.columns, rotation=45, ha="right")
        plt.yticks(np.arange(len(pivot.index)), pivot.index)
        plt.xlabel(spec.xlabel or self.label_for(spec.x), fontsize=13)
        plt.ylabel(self.label_for(spec.color_by), fontsize=13)
        plt.title(spec.title, fontsize=15, wrap=True)
        plt.tight_layout()
        plt.savefig(spec.output, dpi=400)
        plt.close()


# ---- Common metric builders -------------------------------------------------

def column_metric(name: str, label: str, unit: str, *, higher_is_better: bool = True) -> Metric:
    return Metric(name=name, label=label, unit=unit, fn=lambda df: df[name], higher_is_better=higher_is_better)


def runtime_ms(src: str = "total_exec_ms", name: str = "runtime_ms", label: str = "Runtime") -> Metric:
    return Metric(name, label, "ms", lambda df: df[src], higher_is_better=False)

def runtime_power(src: str = "run_power_w", name: str = "avg_runtime_power", label: str = "Average Power", unit: str = "W") -> Metric:
    return Metric(name, label, unit, lambda df: df[src], higher_is_better=False)

def energy_j(runtime_ms_col: str = "runtime_ms", power_w_col: str = "run_power_w") -> Metric:
    return Metric(
        "energy_j",
        "Energy to solution",
        "J",
        lambda df: (df[runtime_ms_col] / 1000.0) * df[power_w_col],
        higher_is_better=False,
    )


def edp(runtime_ms_col: str = "runtime_ms", energy_col: str = "energy_j") -> Metric:
    return Metric(
        "edp_j_s",
        "Energy-delay product",
        "J*s",
        lambda df: df[energy_col] * (df[runtime_ms_col] / 1000.0),
        higher_is_better=False,
    )


def throughput(work: float, runtime_ms_col: str = "runtime_ms", name: str = "throughput") -> Metric:
    return Metric(
        name,
        "Throughput",
        "work/s",
        lambda df: work / (df[runtime_ms_col] / 1000.0),
        higher_is_better=True,
    )


def per_watt(metric_col: str, power_w_col: str = "run_power_w", name: Optional[str] = None) -> Metric:
    out_name = name or f"{metric_col}_per_w"
    return Metric(
        out_name,
        f"{metric_col.replace('_', ' ')} per watt",
        "/W",
        lambda df: df[metric_col] / df[power_w_col],
        higher_is_better=True,
    )





# ---- Example usage ----------------------------------------------------------
# dims = {
#     "gpu_freq_mhz": Dimension("gpu_freq_mhz", "GPU frequency", "MHz"),
#     "cpu_freq_mhz": Dimension("cpu_freq_mhz", "CPU frequency", "MHz"),
#     "problem_size": Dimension("problem_size", "Problem size"),
#     "algorithm": Dimension("algorithm", "Algorithm"),
# }
#
# data = BenchmarkData.from_csv("results.csv").with_metrics([
#     runtime_ms("total_exec_ms"),
#     runtime_ms("kernel_ms", name="kernel_runtime_ms", label="Kernel runtime"),
#     column_metric("run_power_w", "Average power", "W"),
#     energy_j("runtime_ms", "run_power_w"),
#     edp("runtime_ms", "energy_j"),
# ])
#
# agg = data.aggregate(
#     group_by=["algorithm", "gpu_freq_mhz", "cpu_freq_mhz", "problem_size"],
#     value_cols=["runtime_ms", "kernel_runtime_ms", "run_power_w", "energy_j", "edp_j_s"],
# )
#
# plotter = Plotter(dims, data.metrics)
# plotter.plot(agg, PlotSpec(
#     kind="line",
#     x="gpu_freq_mhz",
#     y="runtime_ms",
#     yerr="runtime_ms_std",
#     color_by="algorithm",
#     title="Runtime vs GPU frequency",
#     output="runtime_by_gpu_freq.png",
# ))
# plotter.plot(agg, PlotSpec(
#     kind="heatmap",
#     x="gpu_freq_mhz",
#     color_by="cpu_freq_mhz",
#     y="runtime_ms",
#     title="Runtime across CPU/GPU DVFS points",
#     output="runtime_dvfs_heatmap.png",
# ))
# plotter.plot(agg, PlotSpec(
#     kind="bubble",
#     x="gpu_freq_mhz",
#     y="runtime_ms",
#     size_by="run_power_w",
#     color_by="algorithm",
#     title="Runtime vs frequency, bubble size = power",
#     output="runtime_power_bubble.png",
# ))
