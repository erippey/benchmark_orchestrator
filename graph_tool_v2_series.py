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
import math
from pathlib import Path
import textwrap
from typing import Any, Callable, Iterable, Sequence, Optional, Literal, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


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

    def where(self, **equals) -> "BenchmarkData":
        """Return a filtered BenchmarkData using equality filters.

        Example:
            openmp = data.where(Backend="OpenMP")

        For more complex filters, use `data.subset(lambda df: ...)`.
        """
        out = self.df
        for col, value in equals.items():
            if col not in out.columns:
                raise KeyError(f"Cannot filter; missing column: {col!r}")
            out = out[out[col] == value]
        return BenchmarkData(out.reset_index(drop=True), dict(self.metrics))
    
    def where_or(self, **equals) -> "BenchmarkData":
        """Return a filtered BenchmarkData where any equality filter matches.

        Example:
            cpu_or_gpu = data.where_or(Backend="OpenMP", Platform="CUDA")

        This keeps rows where:
            Backend == "OpenMP" OR Platform == "CUDA"

        For more complex filters, use `data.subset(lambda df: ...)`.
        """
        out = self.df

        if not equals:
            return BenchmarkData(out.reset_index(drop=True), dict(self.metrics))

        mask = pd.Series(False, index=out.index)

        for col, value in equals.items():
            if col not in out.columns:
                raise KeyError(f"Cannot filter; missing column: {col!r}")
            mask |= out[col] == value

        out = out[mask]
        return BenchmarkData(out.reset_index(drop=True), dict(self.metrics))

    def subset(self, predicate: Callable[[pd.DataFrame], pd.Series]) -> "BenchmarkData":
        """Return a filtered BenchmarkData using a boolean-mask function."""
        mask = predicate(self.df)
        return BenchmarkData(self.df[mask].reset_index(drop=True), dict(self.metrics))

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


@dataclass
class PlotSpec:
    kind: str
    x: str
    y: str
    title: str
    output: str

    yerr: Optional[str] = None
    series_by: Optional[Union[str, Sequence[str]]] = None
    size_by: Optional[str] = None

    xlabel: Optional[str] = None
    xlabel_fontsize: int = 12
    ylabel: Optional[str] = None
    ylabel_fontsize: int = 12

    xlim: tuple[int, int] = None
    ylim: tuple[int, int] = None

    figsize: tuple[float, float] = (5, 3)
    dpi: int = 400

    legend: str = "outside_right"
    legend_ncol: Optional[int] = None
    legend_fontsize: int = 8

    title_wrap: int = 72
    title_fontsize: int = 15

    # existing
    palette: str = "tab20"
    use_line_styles: bool = False

    # NEW: style semantics
    hue_by: Optional[Union[str, Sequence[str]]] = None
    shade_by: Optional[Union[str, Sequence[str]]] = None
    marker_by: Optional[Union[str, Sequence[str]]] = None
    linestyle_by: Optional[Union[str, Sequence[str]]] = None

    # explicit style control
    base_colors: Optional[dict[Any, Any]] = None
    shade_values: Optional[dict[Any, float]] = None
    category_orders: dict[str, list[Any]] = field(default_factory=dict)
    value_aliases: dict[str, dict[Any, str]] = field(default_factory=dict)

    highlight_highest_by: Optional[Union[str, Sequence[str]]] = None
    highlight_lowest_by: Optional[Union[str, Sequence[str]]] = None

    highlight_size: float = 95.0
    highlight_linewidth: float = 1.2
    highlight_color: str = "black"
    highlight_alpha: float = 0.6

    highlight_highest_label: Optional[str] = None
    highlight_lowest_label: Optional[str] = None



# --------------- helper for coloring ------------------

def adjust_tone(color, delta):
    """
    delta < 0  -> darken toward black
    delta = 0  -> original color
    delta > 0  -> lighten toward white

    Recommended range: about [-0.4, 0.5]
    """
    rgb = np.array(mcolors.to_rgb(color), dtype=float)

    if delta >= 0:
        out = rgb + (1.0 - rgb) * delta
    else:
        out = rgb * (1.0 + delta)

    out = np.clip(out, 0.0, 1.0)
    return tuple(out)

def pick_group_value(row_map, cols):
    if not cols:
        return None
    vals = tuple(row_map[c] for c in cols)
    return vals[0] if len(vals) == 1 else vals

def default_tone_levels(n):
    if n <= 1:
        return [0.0]
    if n == 2:
        return [-0.20, 0.25]
    if n == 3:
        return [-0.25, 0.05, 0.35]
    if n == 4:
        return [-0.30, -0.10, 0.15, 0.35]
    # fallback
    return np.linspace(-0.30, 0.40, n).tolist()


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

    @staticmethod
    def _as_list(value: Optional[Union[str, Sequence[str]]]) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    def _series_cols(self, spec: PlotSpec) -> list[str]:
        # `series_by` is the preferred name. `color_by` is kept for older calls.
        return self._as_list(spec.series_by if spec.series_by is not None else spec.color_by)

    def _series_label(self, cols: Sequence[str], key) -> str:
        if not isinstance(key, tuple):
            key = (key,)
        return ", ".join(f"{self.label_for(col)}={value}" for col, value in zip(cols, key))
    
    def _series_styles(self, series_keys, series_cols, spec):
        hue_cols = self._as_list(spec.hue_by)
        shade_cols = self._as_list(spec.shade_by)
        marker_cols = self._as_list(spec.marker_by)
        linestyle_cols = self._as_list(spec.linestyle_by)

        # If user did not specify hue_by, default to first series column
        if not hue_cols and series_cols:
            hue_cols = [series_cols[0]]

        # Rebuild row-like maps for each series key
        row_maps = {}
        for key in series_keys:
            if not isinstance(key, tuple):
                key_tuple = (key,)
            else:
                key_tuple = key
            row_maps[key] = dict(zip(series_cols, key_tuple))

        # Stable unique helper
        def stable_unique(values):
            out = []
            seen = set()
            for v in values:
                if v not in seen:
                    seen.add(v)
                    out.append(v)
            return out

        hue_keys = stable_unique(
            pick_group_value(row_maps[k], hue_cols) for k in series_keys
        )

        shade_keys = stable_unique(
            pick_group_value(row_maps[k], shade_cols) for k in series_keys
        ) if shade_cols else [None]

        marker_keys = stable_unique(
            pick_group_value(row_maps[k], marker_cols) for k in series_keys
        ) if marker_cols else [None]

        linestyle_keys = stable_unique(
            pick_group_value(row_maps[k], linestyle_cols) for k in series_keys
        ) if linestyle_cols else [None]

        # Base colors
        if spec.base_colors is not None:
            base_color_map = dict(spec.base_colors)
        else:
            cmap = plt.get_cmap(spec.palette)
            n = max(len(hue_keys), 1)
            base_color_map = {
                hk: cmap(i / max(n - 1, 1))
                for i, hk in enumerate(hue_keys)
            }

        # Shade values
        if spec.shade_values is not None:
            tone_map = dict(spec.shade_values)
        else:
            levels = default_tone_levels(len(shade_keys))
            tone_map = {sk: levels[i] for i, sk in enumerate(shade_keys)}

        marker_cycle = ["o", "s", "^", "D", "v", "P", "X", "*"]
        linestyle_cycle = ["-", "--", "-.", ":"]

        marker_map = {
            mk: marker_cycle[i % len(marker_cycle)]
            for i, mk in enumerate(marker_keys)
        }

        linestyle_map = {
            lk: linestyle_cycle[i % len(linestyle_cycle)]
            for i, lk in enumerate(linestyle_keys)
        }

        styles = {}
        for key in series_keys:
            row = row_maps[key]

            hue_key = pick_group_value(row, hue_cols)
            shade_key = pick_group_value(row, shade_cols)
            marker_key = pick_group_value(row, marker_cols)
            linestyle_key = pick_group_value(row, linestyle_cols)

            base = base_color_map[hue_key]
            delta = tone_map.get(shade_key, 0.0)
            color = adjust_tone(base, delta)

            styles[key] = {
                "color": color,
                "marker": marker_map.get(marker_key, "o"),
                "linestyle": linestyle_map.get(linestyle_key, "-") if spec.use_line_styles else "-",
            }

        return styles

    def _legend_title(self, cols: Sequence[str]) -> str:
        return " / ".join(self.label_for(col) for col in cols)
    
    def _apply_legend(self, fig, ax, spec):
        if spec.legend == "none":
            return

        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return

        # Shared legend kwargs
        legend_kwargs = {
            "fontsize": spec.legend_fontsize,
            "frameon": True,
        }

        # Legends outside the plot area
        if spec.legend == "outside_right":
            ax.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                **legend_kwargs,
            )
            return

        if spec.legend == "outside_left":
            ax.legend(
                handles,
                labels,
                loc="center right",
                bbox_to_anchor=(-0.02, 0.5),
                **legend_kwargs,
            )
            return

        if spec.legend == "bottom":
            ncol = spec.legend_ncol
            if ncol is None:
                ncol = min(4, max(1, len(labels)))

            ax.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.18),
                ncol=ncol,
                **legend_kwargs,
            )
            return

        if spec.legend == "top":
            ncol = spec.legend_ncol
            if ncol is None:
                ncol = min(4, max(1, len(labels)))

            ax.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                **legend_kwargs,
            )
            return

        # Legends inside the plot area
        loc_map = {
            "best": "best",

            "upper_left": "upper left",
            "upper_right": "upper right",
            "lower_left": "lower left",
            "lower_right": "lower right",

            "top_left": "upper left",
            "top_right": "upper right",
            "bottom_left": "lower left",
            "bottom_right": "lower right",

            "center": "center",
            "center_left": "center left",
            "center_right": "center right",
            "upper_center": "upper center",
            "lower_center": "lower center",
        }

        loc = loc_map.get(spec.legend)

        if loc is None:
            raise ValueError(
                f"Unknown legend location {spec.legend!r}. "
                f"Valid options are: none, outside_right, outside_left, bottom, top, "
                f"{', '.join(sorted(loc_map.keys()))}"
            )

        ax.legend(
            handles,
            labels,
            loc=loc,
            **legend_kwargs,
        )

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
        plt.title(spec.title, fontsize=spec.title_fontsize, wrap=True)
        plt.xlabel(spec.xlabel or self.label_for(spec.x), fontsize=spec.xlabel_fontsize)
        plt.ylabel(spec.ylabel or self.label_for(spec.y), fontsize=spec.ylabel_fontsize)
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(spec.output, dpi=spec.dpi)
        plt.close()

    def _finish(self, fig, ax, spec: PlotSpec) -> None:
        self._apply_legend(fig, ax, spec)
        ax.set_xlabel(spec.xlabel or self.label_for(spec.x), fontsize=spec.xlabel_fontsize)
        ax.set_ylabel(spec.ylabel or self.label_for(spec.y), fontsize=spec.ylabel_fontsize)
        if (spec.ylim is not None):
            ax.set_ylim(spec.ylim)
        if (spec.xlim is not None):
            ax.set_xlim(spec.xlim)
        ax.grid(True, alpha=0.35)
        # fig.tight_layout()
        plt.savefig(spec.output, dpi=spec.dpi)
        plt.close(fig)

    def _line(self, df, spec) -> None:

        fig, ax = plt.subplots(figsize=spec.figsize, layout="constrained")

        title = "\n".join(textwrap.wrap(spec.title, width=spec.title_wrap))
        fig.suptitle(title, fontsize=spec.title_fontsize)

        series_cols = self._as_list(spec.series_by)

        if series_cols:
            grouped = df.groupby(series_cols, dropna=False, sort=True)
            series_items = list(grouped)
        else:
            series_items = [(None, df)]

        styles = self._series_styles(
            [key for key, _ in series_items],
            series_cols,
            spec
        )

        for key, group in series_items:
            group = group.sort_values(spec.x)

            label = self._series_label(series_cols, key)

            style = styles[key]

            if spec.yerr and spec.yerr in group.columns:
                ax.errorbar(
                    group[spec.x],
                    group[spec.y],
                    yerr=group[spec.yerr],
                    label=label,
                    capsize=3,
                    linewidth=1.8,
                    markersize=4,
                    **style,
                )
            else:
                ax.plot(
                    group[spec.x],
                    group[spec.y],
                    label=label,
                    linewidth=1.8,
                    markersize=4,
                    **style,
                )

        self._apply_highlight_extrema(df, ax, spec)

        self._finish(fig, ax, spec)

    def _scatter(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        fig, ax = plt.subplots(figsize=spec.figsize, layout="constrained")

        title = "\n".join(textwrap.wrap(spec.title, width=spec.title_wrap))
        fig.suptitle(title, fontsize=14)

        series_cols = self._series_cols(spec)
        if series_cols:
            grouped = df.groupby(series_cols, dropna=False, sort=True)
            series_items = list(grouped)
        else:
            series_items = [(None, df)]

        styles = self._series_styles(
            [key for key, _ in series_items],
            series_cols,
            spec
        )

        for key, group in series_items:
            group = group.sort_values(spec.x)

            label = self._series_label(series_cols, key)

            style = styles[key]

            ax.scatter(group[spec.x], group[spec.y], label=label, **style)

        self._apply_highlight_extrema(df, ax, spec)

        self._finish(fig, ax, spec)
            

    def _bubble(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        if spec.size_by is None:
            raise ValueError("Bubble plots require PlotSpec.size_by")
        plt.figure()
        sizes = df[spec.size_by].astype(float)
        sizes = 50 + 450 * (sizes - sizes.min()) / max(float(sizes.max() - sizes.min()), 1e-12)
        series_cols = self._series_cols(spec)
        if not series_cols:
            plt.scatter(df[spec.x], df[spec.y], s=sizes, alpha=0.65)
        else:
            for key, part in df.groupby(series_cols, sort=True, dropna=False):
                idx = part.index
                plt.scatter(
                    part[spec.x],
                    part[spec.y],
                    s=sizes.loc[idx],
                    alpha=0.65,
                    label=self._series_label(series_cols, key),
                )
            plt.legend(title=self._legend_title(series_cols))
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

    # ---------- helpers for highlighting highest lowest point 
    def _groupby_cols(self, df, cols):
        """
        Like df.groupby(cols), but avoids awkward behavior for empty / single-column cases.
        """
        cols = self._as_list(cols)

        if not cols:
            return [(None, df)]

        if len(cols) == 1:
            return list(df.groupby(cols[0], dropna=False, sort=True))

        return list(df.groupby(cols, dropna=False, sort=True))


    def _highlight_scope_label(self, cols) -> str:
        cols = self._as_list(cols)

        if not cols:
            return "all data"

        return ", ".join(self.label_for(c) for c in cols)


    def _add_highlight_legend_entry(
        self,
        ax,
        *,
        label: str,
        color: str,
        size: float,
        linewidth: float,
    ) -> None:
        """
        Add an empty artist so the highlight explanation appears at the bottom
        of the normal legend.
        """
        ax.plot(
            [],
            [],
            linestyle="None",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=color,
            markeredgewidth=linewidth,
            markersize=math.sqrt(size),
            label=label,
        )


    def _highlight_extrema(
        self,
        ax,
        df,
        spec,
        *,
        by,
        mode: str,
        color: str,
        size: float,
        linewidth: float,
        alpha: float,
    ) -> int:
        """
        Circle the highest or lowest y-value within each group defined by `by`.

        mode: "max" or "min"
        """
        by_cols = self._as_list(by)

        required = [spec.x, spec.y, *by_cols]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise KeyError(f"Cannot highlight extrema; missing columns: {missing}")

        count = 0

        for _, group in self._groupby_cols(df, by_cols):
            valid = group.dropna(subset=[spec.x, spec.y])

            if valid.empty:
                continue

            if mode == "max":
                idx = valid[spec.y].idxmax()
            elif mode == "min":
                idx = valid[spec.y].idxmin()
            else:
                raise ValueError(f"Unsupported extrema highlight mode: {mode}")

            row = valid.loc[idx]

            ax.scatter(
                [row[spec.x]],
                [row[spec.y]],
                s=size,
                marker="o",
                facecolors="none",
                edgecolors=color,
                linewidths=linewidth,
                alpha=alpha,
                zorder=10,
                label="_nolegend_",
            )

            count += 1

        return count
    
    def _apply_highlight_extrema(self, df : pd.DataFrame, ax, spec: PlotSpec):
        # Highlight lowest y-value within each requested group.
        if spec.highlight_lowest_by is not None:
            count = self._highlight_extrema(
                ax,
                df,
                spec,
                by=spec.highlight_lowest_by,
                mode="min",
                color=spec.highlight_color,
                size=spec.highlight_size,
                linewidth=spec.highlight_linewidth,
                alpha=spec.highlight_alpha,
            )

            if count > 0:
                scope = self._highlight_scope_label(spec.highlight_lowest_by)
                label = (
                    spec.highlight_lowest_label
                    or f"circled: lowest {self.label_for(spec.y)} within {scope}"
                )

                self._add_highlight_legend_entry(
                    ax,
                    label=label,
                    color=spec.highlight_color,
                    size=spec.highlight_size,
                    linewidth=spec.highlight_linewidth,
                )

        # Highlight highest y-value within each requested group.
        if spec.highlight_highest_by is not None:
            highest_size = spec.highlight_size * 1.35

            count = self._highlight_extrema(
                ax,
                df,
                spec,
                by=spec.highlight_highest_by,
                mode="max",
                color=spec.highlight_color,
                size=highest_size,
                linewidth=spec.highlight_linewidth,
                alpha=spec.highlight_alpha,
            )

            if count > 0:
                scope = self._highlight_scope_label(spec.highlight_highest_by)
                label = (
                    spec.highlight_highest_label
                    or f"circled: highest {self.label_for(spec.y)} within {scope}"
                )

                self._add_highlight_legend_entry(
                    ax,
                    label=label,
                    color=spec.highlight_color,
                    size=highest_size,
                    linewidth=spec.highlight_linewidth,
                )


# ---- Common metric builders -------------------------------------------------

def column_metric(name: str, label: str, unit: str, *, higher_is_better: bool = True) -> Metric:
    return Metric(name=name, label=label, unit=unit, fn=lambda df: df[name], higher_is_better=higher_is_better)


def runtime_ms(src: str = "total_exec_ms", name: str = "runtime_ms", label: str = "Runtime") -> Metric:
    return Metric(name, label, "ms", lambda df: df[src], higher_is_better=False)


def energy_j(runtime_ms_col: str = "runtime_ms", power_w_col: str = "run_power_w") -> Metric:
    return Metric(
        "energy_j",
        "Energy to solution",
        "J",
        lambda df: (df[runtime_ms_col] / 1000.0) * df[power_w_col],
        higher_is_better=False,
    )

def runtime_power(src: str = "run_power_w", name: str = "avg_runtime_power", label: str = "Average Power", unit: str = "W") -> Metric:
    return Metric(name, label, unit, lambda df: df[src], higher_is_better=False)

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


# ---- Relative metric builders ----------------------------------------------

def add_relative_to_group_best(
    df: pd.DataFrame,
    value_col: str,
    group_by: Sequence[str],
    *,
    higher_is_better: bool,
    out_col: Optional[str] = None,
    as_percent: bool = False,
) -> pd.DataFrame:
    """
    Add a relative score column normalized within each group.

    For higher-is-better metrics:
        relative = current / group_max

    For lower-is-better metrics:
        relative = group_min / current

    So the best point in each group is 1.0, and worse points are < 1.0.
    """
    if out_col is None:
        out_col = f"rel_{value_col}"

    missing = [c for c in list(group_by) + [value_col] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for relative normalization: {missing}")

    out = df.copy()

    grouped = out.groupby(list(group_by), dropna=False)[value_col]

    if higher_is_better:
        reference = grouped.transform("max")
        out[out_col] = out[value_col] / reference
    else:
        reference = grouped.transform("min")
        out[out_col] = reference / out[value_col]

    if as_percent:
        out[out_col] *= 100.0

    return out


def add_relative_to_group_x(
    df: pd.DataFrame,
    value_col: str,
    group_by: Sequence[str],
    x_col: str,
    x_value: str,
    *,
    higher_is_better: bool,
    out_col: Optional[str] = None,
    as_percent: bool = False,
) -> pd.DataFrame:
    """
    Add a relative score column normalized within each group.

    x_col is the name of the column used to select the baseline.

    The selected baseline is chosen via x_value. x_value can be:
        - "min": use the row with the minimum x_col value within each group
        - "max": use the row with the maximum x_col value within each group
        - a specific value, e.g. "1800", 1800, "Serial", etc.

    For higher-is-better metrics:
        relative = current / selected_baseline

    For lower-is-better metrics:
        relative = selected_baseline / current

    So the baseline point in each group is 1.0. Values better than the
    baseline may be > 1.0, and values worse than the baseline may be < 1.0.
    """
    if out_col is None:
        out_col = f"rel_{value_col}"

    needed_cols = list(group_by) + [value_col, x_col]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns for relative normalization: {missing}")

    out = df.copy()
    group_cols = list(group_by)

    grouped = out.groupby(group_cols, dropna=False)

    if x_value == "min":
        # Index of baseline row per group: row with minimum x_col.
        baseline_idx = grouped[x_col].idxmin()

    elif x_value == "max":
        # Index of baseline row per group: row with maximum x_col.
        baseline_idx = grouped[x_col].idxmax()

    else:
        # Select rows whose x_col matches the requested baseline value.
        # This tries both direct comparison and string comparison so that
        # x_value="1800" can still match numeric 1800 columns.
        mask = (out[x_col] == x_value) | (out[x_col].astype(str) == str(x_value))
        candidates = out.loc[mask]

        if candidates.empty:
            raise ValueError(
                f"No rows found where {x_col!r} matches baseline value {x_value!r}"
            )

        # If multiple rows match within a group, take the first one.
        baseline_idx = (
            candidates
            .groupby(group_cols, dropna=False)
            .head(1)
            .index
        )

    # Build a DataFrame containing one baseline value per group.
    baseline = out.loc[baseline_idx, group_cols + [value_col]].rename(
        columns={value_col: "__baseline_value"}
    )

    # Merge baseline values back onto every row in the same group.
    out = out.merge(
        baseline,
        on=group_cols,
        how="left",
        validate="many_to_one",
    )

    missing_baseline = out["__baseline_value"].isna()
    if missing_baseline.any():
        missing_groups = (
            out.loc[missing_baseline, group_cols]
            .drop_duplicates()
            .to_dict("records")
        )
        raise ValueError(
            f"Some groups do not have a baseline for {x_col}={x_value!r}: "
            f"{missing_groups}"
        )

    if higher_is_better:
        out[out_col] = out[value_col] / out["__baseline_value"]
    else:
        out[out_col] = out["__baseline_value"] / out[value_col]

    if as_percent:
        out[out_col] *= 100.0

    out = out.drop(columns="__baseline_value")

    return out