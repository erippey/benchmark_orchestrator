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

from dataclasses import dataclass, field, replace
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
PlotKind = Literal["line", "scatter", "heatmap", "bubble", "bar"]


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
    color_by: Optional[str] = None

    # Bar-chart layout. For a hierarchical bar chart, use:
    #   bar_group_by=outer x-axis grouping, e.g. device_backend
    #   bar_subgroup_by=middle grouping, e.g. algorithm
    #   x=leaf label repeated on each bar, e.g. problem_size
    bar_group_by: Optional[Union[str, Sequence[str]]] = None
    bar_subgroup_by: Optional[Union[str, Sequence[str]]] = None
    bar_label_mode: str = "hierarchical"  # hierarchical, flat, leaf, none

    bar_width: float = 0.82
    bar_group_gap: float = 1.10
    bar_subgroup_gap: float = 0.35
    bar_error_capsize: float = 3.0
    bar_edgecolor: Optional[Any] = None
    bar_linewidth: float = 0.0
    bar_alpha: float = 0.95

    bar_value_labels: bool = False
    bar_value_fmt: str = "{:.3g}"
    bar_value_rotation: float = 0.0
    bar_value_padding: float = 3.0
    bar_value_fontsize: int = 8

    bar_tick_label_rotation: float = 0.0
    bar_tick_label_fontsize: int = 9
    bar_subgroup_label_fontsize: int = 9
    bar_group_label_fontsize: int = 10
    bar_multilevel_bottom: float = 0.24
    bar_show_group_separators: bool = True
    bar_show_group_labels: bool = True
    bar_show_subgroup_labels: bool = True

    xscale: str = "linear"
    xbase: int = 10
    xlabel: Optional[str] = ""
    xlabel_fontsize: int = 12
    yscale: str = "linear"
    ybase: int = 10
    ylabel: Optional[str] = ""
    ylabel_fontsize: int = 12

    xlim: tuple[int, int] = None
    ylim: tuple[int, int] = None

    figsize: tuple[float, float] = (5, 3)
    dpi: int = 400

    legend: str = "outside_right"
    legend_ncol: Optional[int] = None
    legend_fontsize: int = 8
    legend_style: Optional[Literal["full", "values"]] = "full"
    legend_separator: Optional[str] = "-"

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

    # Bar-only non-color semantics. Matplotlib calls these hatches.
    # Good for grayscale printing and colorblind-accessible figures.
    pattern_by: Optional[Union[str, Sequence[str]]] = None

    # explicit style control
    base_colors: Optional[dict[Any, Any]] = None
    shade_values: Optional[dict[Any, float]] = None
    pattern_values: Optional[dict[Any, str]] = None
    marker_values: Optional[dict[Any, str]] = None
    pattern_cycle: Sequence[str] = field(
        default_factory=lambda: ["", "//", "\\", "xx", "++", "--", "oo", "..", "**"]
    )
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
    
    def _series_styles(self, series_keys, series_cols, spec, *, include_pattern: bool = False):
        hue_cols = self._as_list(spec.hue_by)
        shade_cols = self._as_list(spec.shade_by)
        marker_cols = self._as_list(spec.marker_by)
        linestyle_cols = self._as_list(spec.linestyle_by)
        pattern_cols = self._as_list(spec.pattern_by) if include_pattern else []

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

        pattern_keys = stable_unique(
            pick_group_value(row_maps[k], pattern_cols) for k in series_keys
        ) if pattern_cols else [None]

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

        if spec.marker_values is not None:
            marker_map = spec.marker_values
        else:
            marker_map = {
                mk: marker_cycle[i % len(marker_cycle)]
                for i, mk in enumerate(marker_keys)
            }

        linestyle_map = {
            lk: linestyle_cycle[i % len(linestyle_cycle)]
            for i, lk in enumerate(linestyle_keys)
        }

        if spec.pattern_values is not None:
            pattern_map = dict(spec.pattern_values)
        else:
            pattern_cycle = list(spec.pattern_cycle)
            if not pattern_cycle:
                pattern_cycle = [""]
            pattern_map = {
                pk: pattern_cycle[i % len(pattern_cycle)]
                for i, pk in enumerate(pattern_keys)
            }

        styles = {}
        for key in series_keys:
            row = row_maps[key]

            hue_key = pick_group_value(row, hue_cols)
            shade_key = pick_group_value(row, shade_cols)
            marker_key = pick_group_value(row, marker_cols)
            linestyle_key = pick_group_value(row, linestyle_cols)
            pattern_key = pick_group_value(row, pattern_cols)

            base = base_color_map[hue_key]
            delta = tone_map.get(shade_key, 0.0)
            color = adjust_tone(base, delta)

            style = {
                "color": color,
                "marker": marker_map.get(marker_key, "o"),
                "linestyle": linestyle_map.get(linestyle_key, "-") if spec.use_line_styles else "-",
            }

            if include_pattern:
                style["hatch"] = pattern_map.get(pattern_key, "")

            styles[key] = style

        return styles

    def _legend_title(self, cols: Sequence[str]) -> str:
        return " / ".join(self.label_for(col) for col in cols)
    
    def _apply_legend(self, fig, ax, spec):
        if spec.legend == "none":
            return

        handles, labels = ax.get_legend_handles_labels()
        if not handles:
            return
        
        formatted_labels = []
        
        for handle, label in zip(handles, labels):
                if not label or label.startswith("_"):
                    continue

                formatted_labels.append(self._format_legend_label(
                    label,
                    style=spec.legend_style,
                    sep=spec.legend_separator,
                ))

        


        # Shared legend kwargs
        legend_kwargs = {
            "fontsize": spec.legend_fontsize,
            "frameon": True,
        }

        # Legends outside the plot area
        if spec.legend == "outside_right":
            ax.legend(
                handles,
                formatted_labels,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                **legend_kwargs,
            )
            return

        if spec.legend == "outside_left":
            ax.legend(
                handles,
                formatted_labels,
                loc="center right",
                bbox_to_anchor=(-0.02, 0.5),
                **legend_kwargs,
            )
            return

        if spec.legend == "bottom":
            ncol = spec.legend_ncol
            if ncol is None:
                ncol = min(4, max(1, len(formatted_labels)))

            ax.legend(
                handles,
                formatted_labels,
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
                formatted_labels,
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
            formatted_labels,
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
        elif spec.kind == "bar":
            self._bar(df, spec)
        else:
            raise ValueError(f"Unsupported plot kind: {spec.kind}")

    def _format_axes(self, ax, spec: PlotSpec) -> None:
        """Apply common axis formatting without saving or closing the figure.

        This is intentionally separate from `_finish()` so the same plot
        rendering code can be used for both normal single-plot figures and
        multi-panel figures.
        """
        if (spec.xlabel is None):
            xlabel = None
        elif len(spec.xlabel) == 0:
            xlabel = self.label_for(spec.x)
        else:
            xlabel = spec.xlabel

        if (spec.ylabel is None):
            ylabel = None
        elif len(spec.ylabel) == 0:
            ylabel = self.label_for(spec.y)
        else:
            ylabel = spec.ylabel

        if spec.kind != "bar":
            if (xlabel):
                ax.set_xlabel(xlabel, fontsize=spec.xlabel_fontsize)
            if spec.xlim is not None:
                ax.set_xlim(spec.xlim)
            if spec.xscale == "log":
                ax.set_xscale(spec.xscale, base=spec.xbase)
            else:
                ax.set_xscale(spec.xscale)

        if (ylabel):
            ax.set_ylabel(ylabel, fontsize=spec.ylabel_fontsize)
        if spec.yscale == "log":
            ax.set_yscale(spec.yscale, base=spec.ybase)
        else:
            ax.set_yscale(spec.yscale)
        if spec.ylim is not None:
            ax.set_ylim(spec.ylim)

        if spec.kind == "bar":
            ax.grid(True, axis="y", alpha=0.35)
            ax.xaxis.grid(False)
        else:
            ax.grid(True, alpha=0.35)

    def _finish(self, fig, ax, spec: PlotSpec) -> None:
        self._apply_legend(fig, ax, spec)
        self._format_axes(ax, spec)
        plt.savefig(spec.output, dpi=spec.dpi, bbox_inches="tight")
        plt.close(fig)


    def _format_legend_label(
        self,
        label: str,
        *,
        style: Literal["full", "values"] = "full",
        sep: str = "-",
    ) -> str:
        """Format a legend label.

        Examples
        --------
        full:
            "Algorithm=BFS, Variant=OpenMP"

        values:
            "BFS-OpenMP"
        """
        if style == "full":
            return label

        if style == "values":
            # Match comma-separated key=value pairs.
            # Example: "Algorithm=BFS, Variant=OpenMP"
            parts = [part.strip() for part in label.split(",")]

            values: list[str] = []
            for part in parts:
                if "=" not in part:
                    # Fall back gracefully for labels that are not key=value.
                    values.append(part)
                    continue

                _, value = part.split("=", 1)
                values.append(value.strip())

            return sep.join(values)

        raise ValueError(f"Unknown legend label style: {style}")

    def _dedupe_legend(
        self,
        axes,
        *,
        label_style: Literal["full", "values"] = "full",
        label_sep: str = "-",
    ) -> tuple[list[Any], list[str]]:
        """Collect unique legend entries from one or more axes.

        Matplotlib creates duplicate legend entries when each panel draws the
        same series. This helper keeps the first handle for each formatted label
        and ignores internal/private labels such as `_nolegend_`.
        """
        by_label: dict[str, Any] = {}

        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()

            for handle, label in zip(handles, labels):
                if not label or label.startswith("_"):
                    continue

                formatted_label = self._format_legend_label(
                    label,
                    style=label_style,
                    sep=label_sep,
                )

                if formatted_label not in by_label:
                    by_label[formatted_label] = handle

        return list(by_label.values()), list(by_label.keys())

    def _apply_figure_legend(
        self,
        fig,
        axes,
        *,
        legend: str,
        legend_ncol: Optional[int],
        legend_fontsize: int,
        legend_label_style: Literal["full", "values"] = "full",
        legend_label_sep: str = "-",
    ) -> None:
        """Apply one merged legend to an entire multi-panel figure."""
        if legend == "none":
            return

        handles, labels = self._dedupe_legend(axes,
                            label_style=legend_label_style, 
                            label_sep=legend_label_sep)
        if not handles:
            return

        legend_kwargs = {
            "fontsize": legend_fontsize,
            "frameon": True,
        }

        if legend == "right":
            fig.legend(
                handles,
                labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                **legend_kwargs,
            )
            return

        if legend == "left":
            fig.legend(
                handles,
                labels,
                loc="center right",
                bbox_to_anchor=(-0.01, 0.5),
                **legend_kwargs,
            )
            return

        if legend == "bottom":
            ncol = legend_ncol
            if ncol is None:
                ncol = min(4, max(1, len(labels)))

            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, -0.02),
                ncol=ncol,
                **legend_kwargs,
            )
            return

        if legend == "top":
            ncol = legend_ncol
            if ncol is None:
                ncol = min(4, max(1, len(labels)))

            fig.legend(
                handles,
                labels,
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                ncol=ncol,
                **legend_kwargs,
            )
            return

        raise ValueError("figure legend must be one of: right, left, bottom, top, none")
    
    def _normalize_column_labels(self, labels, ncols: int) -> dict[int, str]:
        """
        Normalize a column-label input into {column_index: label}.

        Accepted forms:
            None
            "same label for every column"
            ["label col 0", "label col 1", None, ...]
            {0: "label col 0", 1: "label col 1"}
        """
        if labels is None:
            return {}

        if isinstance(labels, str):
            return {i: labels for i in range(ncols)}

        if isinstance(labels, dict):
            return {
                int(col): label
                for col, label in labels.items()
                if label is not None
            }

        out = {}
        for i, label in enumerate(labels):
            if i >= ncols:
                break
            if label is not None:
                out[i] = label

        return out


    def _add_column_labels(
        self,
        fig,
        axes_grid,
        *,
        nrows: int,
        ncols: int,
        n_panels: int,
        column_xlabels=None,
        column_ylabels=None,
        xlabel_fontsize: int = 12,
        ylabel_fontsize: int = 12,
        xlabel_pad: float = 0.045,
        ylabel_pad: float = 0.055,
    ) -> None:
        """
        Add one x/y label per subplot column.

        Labels are positioned from the visible axes in each column, so this also
        works for incomplete grids.
        """
        xlabels = self._normalize_column_labels(column_xlabels, ncols)
        ylabels = self._normalize_column_labels(column_ylabels, ncols)

        if not xlabels and not ylabels:
            return

        # Make sure constrained_layout has computed final-ish axes positions.
        fig.canvas.draw()

        for col in range(ncols):
            col_axes = []

            for row in range(nrows):
                flat_idx = row * ncols + col
                if flat_idx >= n_panels:
                    continue

                ax = axes_grid[row][col]
                if ax.get_visible():
                    col_axes.append(ax)

            if not col_axes:
                continue

            boxes = [ax.get_position() for ax in col_axes]

            left = min(box.x0 for box in boxes)
            right = max(box.x1 for box in boxes)
            bottom = min(box.y0 for box in boxes)
            top = max(box.y1 for box in boxes)

            x_center = 0.5 * (left + right)
            y_center = 0.5 * (bottom + top)

            if col in xlabels:
                fig.text(
                    x_center,
                    bottom - xlabel_pad,
                    xlabels[col],
                    ha="center",
                    va="top",
                    fontsize=xlabel_fontsize,
                )

            if col in ylabels:
                fig.text(
                    left - ylabel_pad,
                    y_center,
                    ylabels[col],
                    ha="right",
                    va="center",
                    rotation="vertical",
                    fontsize=ylabel_fontsize,
                )

    def _draw_on_axes(self, ax, df: pd.DataFrame, spec: PlotSpec) -> None:
        """Draw a PlotSpec onto an existing axes without saving/closing."""

        if spec.kind == "line":
            self._draw_line(ax, df, spec)
        elif spec.kind == "scatter":
            self._draw_scatter(ax, df, spec)
        elif spec.kind == "bubble":
            self._draw_bubble(ax, df, spec)
        elif spec.kind == "bar":
            self._draw_bar(ax, df, spec)
        else:
            raise ValueError(
                f"plot_panels currently supports line, scatter, bubble, and bar plots; "
                f"got {spec.kind!r}"
            )

    def plot_panels(
        self,
        panels: Sequence[tuple[pd.DataFrame, PlotSpec]],
        *,
        output: str,
        title: Optional[str] = None,
        figsize: Optional[tuple[float, float]] = None,
        dpi: Optional[int] = None,
        ncols: Optional[int] = None,
        sharex: bool = False,
        sharey: bool = False,
        legend: str = "right",
        legend_style: Literal["full", "values"] = "full",
        legend_seperator: str = "-",
        legend_ncol: Optional[int] = None,
        legend_fontsize: Optional[int] = None,
        title_fontsize: Optional[int] = None,
        title_wrap: int = 92,
        hide_repeated_ylabels: bool = True,

        column_gap: Optional[float] = None,
        row_gap: Optional[float] = None,
        layout_w_pad: Optional[float] = None,
        layout_h_pad: Optional[float] = None,

        shared_xlabel: Optional[str] = None,
        shared_ylabel: Optional[str] = None,
        shared_xlabel_fontsize: Optional[int] = None,
        shared_ylabel_fontsize: Optional[int] = None,
        hide_panel_xlabels_when_shared: bool = True,
        hide_panel_ylabels_when_shared: bool = True,

        column_xlabels: Optional[Union[str, Sequence[Optional[str]], dict[int, str]]] = None,
        column_ylabels: Optional[Union[str, Sequence[Optional[str]], dict[int, str]]] = None,
        column_xlabel_fontsize: Optional[int] = None,
        column_ylabel_fontsize: Optional[int] = None,
        column_xlabel_pad: float = 0.045,
        column_ylabel_pad: float = 0.055,
        hide_panel_xlabels_with_column_labels: bool = True,
        hide_panel_ylabels_with_column_labels: bool = True,
    ) -> None:
        """Render multiple PlotSpecs into one figure with one merged legend.

        Example:
            plotter.plot_panels(
                [(gpu_df, gpu_spec), (cpu_df, cpu_spec)],
                output="graphs/power_cpu_gpu.png",
                title="Average Power Draw vs Operating Frequency",
                figsize=(10, 4),
                sharey=True,
                legend="bottom",
                legend_ncol=4,
            )

        This intentionally reuses the same style/highlight logic as `plot()`,
        but does not call `_finish()` for each panel.
        """
        panels = list(panels)
        if not panels:
            raise ValueError("plot_panels requires at least one panel")

        n_panels = len(panels)

        if ncols is None:
            ncols = n_panels
        if ncols <= 0:
            raise ValueError("ncols must be positive")

        nrows = math.ceil(n_panels / ncols)

        if figsize is None:
            first_spec = panels[0][1]
            panel_w, panel_h = first_spec.figsize
            figsize = (panel_w * ncols, panel_h * nrows)

        if dpi is None:
            dpi = panels[0][1].dpi

        if legend_fontsize is None:
            legend_fontsize = panels[0][1].legend_fontsize

        if title_fontsize is None:
            title_fontsize = panels[0][1].title_fontsize

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=figsize,
            sharex=sharex,
            sharey=sharey,
            layout="constrained",
        )

        if (
            column_gap is not None
            or row_gap is not None
            or layout_w_pad is not None
            or layout_h_pad is not None
        ):
            wspace = 0.02 if column_gap is None else column_gap
            hspace = 0.02 if row_gap is None else row_gap
            w_pad = 0.04167 if layout_w_pad is None else layout_w_pad
            h_pad = 0.04167 if layout_h_pad is None else layout_h_pad

            # Newer Matplotlib layout-engine API.
            engine = fig.get_layout_engine()
            if engine is not None and hasattr(engine, "set"):
                engine.set(
                    wspace=wspace,
                    hspace=hspace,
                    w_pad=w_pad,
                    h_pad=h_pad,
                )
            else:
                # Older Matplotlib fallback.
                fig.set_constrained_layout_pads(
                    wspace=wspace,
                    hspace=hspace,
                    w_pad=w_pad,
                    h_pad=h_pad,
                )

        axes_arr = np.atleast_1d(axes).ravel().tolist()
        axes_grid = np.asarray(axes, dtype=object).reshape(nrows, ncols)

        for ax, (df, spec) in zip(axes_arr, panels):
            # Panel specs should never create their own legends or save files.
            # The figure-level legend and output path are handled here.
            panel_spec = replace(spec, output=output)

            panel_title = "\n".join(textwrap.wrap(panel_spec.title, width=panel_spec.title_wrap))
            ax.set_title(panel_title, fontsize=panel_spec.title_fontsize)

            if shared_xlabel is not None and hide_panel_xlabels_when_shared:
                panel_spec.xlabel = None

            if shared_ylabel is not None and hide_panel_ylabels_when_shared:
                panel_spec.ylabel = None

            if column_xlabels is not None and hide_panel_xlabels_with_column_labels:
                panel_spec.xlabel = None

            if column_ylabels is not None and hide_panel_ylabels_with_column_labels:
                panel_spec.ylabel = None

            self._draw_on_axes(ax, df, panel_spec)
            self._format_axes(ax, panel_spec)
            self._apply_legend(fig, ax, panel_spec)

        # Hide any unused axes in an incomplete grid.
        for ax in axes_arr[n_panels:]:
            ax.set_visible(False)

        if hide_repeated_ylabels and sharey:
            for i, ax in enumerate(axes_arr[:n_panels]):
                if i % ncols != 0:
                    ax.set_ylabel("")

        self._apply_figure_legend(
            fig,
            axes_arr[:n_panels],
            legend=legend,
            legend_ncol=legend_ncol,
            legend_fontsize=legend_fontsize,
            legend_label_style=legend_style,
            legend_label_sep=legend_seperator,
        )

        if shared_xlabel is not None:
            if shared_xlabel_fontsize is None:
                shared_xlabel_fontsize = panels[0][1].xlabel_fontsize

            fig.supxlabel(shared_xlabel, fontsize=shared_xlabel_fontsize)

        if shared_ylabel is not None:
            if shared_ylabel_fontsize is None:
                shared_ylabel_fontsize = panels[0][1].ylabel_fontsize

            fig.supylabel(shared_ylabel, fontsize=shared_ylabel_fontsize)

        if column_xlabel_fontsize is None:
            column_xlabel_fontsize = panels[0][1].xlabel_fontsize

        if column_ylabel_fontsize is None:
            column_ylabel_fontsize = panels[0][1].ylabel_fontsize

        self._add_column_labels(
            fig,
            axes_grid,
            nrows=nrows,
            ncols=ncols,
            n_panels=n_panels,
            column_xlabels=column_xlabels,
            column_ylabels=column_ylabels,
            xlabel_fontsize=column_xlabel_fontsize,
            ylabel_fontsize=column_ylabel_fontsize,
            xlabel_pad=column_xlabel_pad,
            ylabel_pad=column_ylabel_pad,
        )

        if title is not None:
            wrapped = "\n".join(textwrap.wrap(title, width=title_wrap))
            fig.suptitle(wrapped, fontsize=title_fontsize)

        fig.savefig(output, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def _line(self, df, spec) -> None:

        fig, ax = plt.subplots(figsize=spec.figsize, layout="constrained")

        title = "\n".join(textwrap.wrap(spec.title, width=spec.title_wrap))
        fig.suptitle(title, fontsize=spec.title_fontsize)

        self._draw_line(ax, df, spec)
        self._finish(fig, ax, spec)

    def _draw_line(self, ax, df: pd.DataFrame, spec: PlotSpec) -> None:
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

    def _scatter(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        fig, ax = plt.subplots(figsize=spec.figsize, layout="constrained")

        title = "\n".join(textwrap.wrap(spec.title, width=spec.title_wrap))
        fig.suptitle(title, fontsize=spec.title_fontsize)

        self._draw_scatter(ax, df, spec)
        self._finish(fig, ax, spec)

    def _draw_scatter(self, ax, df: pd.DataFrame, spec: PlotSpec) -> None:
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
            

    # ---------- bar-chart helpers ------------------------------------------------

    def _unique_preserve_order(self, values):
        out = []
        seen = set()
        for value in values:
            key = self._hashable_key(value)
            if key not in seen:
                seen.add(key)
                out.append(value)
        return out

    @staticmethod
    def _hashable_key(value):
        if isinstance(value, tuple):
            return tuple(Plotter._hashable_key(v) for v in value)
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    def _row_key(self, row, cols: Sequence[str]):
        vals = tuple(row[col] for col in cols)
        if not vals:
            return None
        return vals[0] if len(vals) == 1 else vals

    def _display_value(self, col: str, value, spec: PlotSpec) -> str:
        alias_map = spec.value_aliases.get(col, {})
        return str(alias_map.get(value, value))

    def _key_label(self, cols: Sequence[str], key, spec: PlotSpec, *, include_names: bool = False) -> str:
        cols = list(cols)
        if not cols:
            return ""
        if not isinstance(key, tuple):
            key = (key,)

        parts = []
        for col, value in zip(cols, key):
            value_label = self._display_value(col, value, spec)
            if include_names:
                parts.append(f"{self.label_for(col)}={value_label}")
            else:
                parts.append(value_label)
        return " / ".join(parts)

    def _category_rank_map(self, df: pd.DataFrame, col: str, spec: PlotSpec) -> dict[Any, int]:
        explicit = list(spec.category_orders.get(col, []))
        seen = list(pd.unique(df[col]))

        ordered = []
        used = set()
        for value in explicit + seen:
            key = self._hashable_key(value)
            if key not in used:
                used.add(key)
                ordered.append(value)

        return {value: i for i, value in enumerate(ordered)}

    def _sort_by_category_orders(self, df: pd.DataFrame, cols: Sequence[str], spec: PlotSpec) -> pd.DataFrame:
        if not cols:
            return df

        out = df.copy()
        tmp_cols = []
        for i, col in enumerate(cols):
            rank = self._category_rank_map(out, col, spec)
            tmp = f"__plot_sort_{i}"
            out[tmp] = out[col].map(rank).astype(float)
            tmp_cols.append(tmp)

        out = out.sort_values(tmp_cols, kind="mergesort").drop(columns=tmp_cols)
        return out

    def _format_bar_value(self, value, fmt: str) -> str:
        if pd.isna(value):
            return ""
        try:
            return fmt.format(value)
        except Exception:
            return format(value, fmt)

    def _bar_style_spec(self, spec: PlotSpec, group_cols: list[str], subgroup_cols: list[str]) -> PlotSpec:
        """Return a shallow spec copy with useful default bar styling semantics."""
        style_spec = replace(spec)

        # If the caller did not specify hue/shade, use the hierarchy in a way
        # that usually reads naturally: algorithm gets color, leaf size gets tone.
        if style_spec.hue_by is None:
            if subgroup_cols:
                style_spec.hue_by = subgroup_cols[0]
            elif group_cols:
                style_spec.hue_by = group_cols[0]
            else:
                style_spec.hue_by = spec.x

        if style_spec.shade_by is None and subgroup_cols and spec.x not in self._as_list(style_spec.hue_by):
            style_spec.shade_by = spec.x

        return style_spec

    def _bar(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        fig, ax = plt.subplots(figsize=spec.figsize, layout="constrained")

        title = "\n".join(textwrap.wrap(spec.title, width=spec.title_wrap))
        fig.suptitle(title, fontsize=spec.title_fontsize)

        self._draw_bar(ax, df, spec, adjust_figure_bottom=True)

        # Hierarchical bar labels usually make a conventional xlabel redundant.
        finish_spec = self._bar_finish_spec(spec)
        self._finish(fig, ax, finish_spec)

    def _bar_finish_spec(self, spec: PlotSpec) -> PlotSpec:
        """Return the axis-formatting spec used after drawing a bar chart."""
        mode = spec.bar_label_mode.lower()
        return replace(
            spec,
            xlabel=spec.xlabel if spec.xlabel is not None else (" " if mode == "hierarchical" else None),
        )

    def _draw_bar(
        self,
        ax,
        df: pd.DataFrame,
        spec: PlotSpec,
        *,
        adjust_figure_bottom: bool = False,
    ) -> None:
        """Draw a bar PlotSpec onto an existing axes without saving/closing.

        This is the panel-friendly half of `_bar()`, matching `_draw_line()`,
        `_draw_scatter()`, and `_draw_bubble()`.
        """
        group_cols = self._as_list(spec.bar_group_by)
        subgroup_cols = self._as_list(spec.bar_subgroup_by)
        leaf_col = spec.x
        hierarchy_cols = [*group_cols, *subgroup_cols, leaf_col]

        required = [leaf_col, spec.y, *group_cols, *subgroup_cols]
        if spec.yerr is not None:
            required.append(spec.yerr)
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Cannot create bar plot; missing columns: {missing}")

        plot_df = df.dropna(subset=[spec.y]).copy()
        if plot_df.empty:
            raise ValueError("Cannot create bar plot from an empty dataframe")

        plot_df = self._sort_by_category_orders(plot_df, hierarchy_cols, spec).reset_index(drop=True)

        positions = []
        group_positions: dict[Any, list[float]] = {}
        subgroup_positions: dict[Any, list[float]] = {}

        prev_group_key = object()
        prev_subgroup_key = object()
        cursor = 0.0

        for _, row in plot_df.iterrows():
            group_key = self._row_key(row, group_cols)
            subgroup_key = self._row_key(row, [*group_cols, *subgroup_cols])

            if positions:
                if group_cols and group_key != prev_group_key:
                    cursor += spec.bar_group_gap
                elif subgroup_cols and subgroup_key != prev_subgroup_key:
                    cursor += spec.bar_subgroup_gap

            x_pos = cursor
            positions.append(x_pos)
            group_positions.setdefault(group_key, []).append(x_pos)
            subgroup_positions.setdefault(subgroup_key, []).append(x_pos)

            prev_group_key = group_key
            prev_subgroup_key = subgroup_key
            cursor += 1.0

        plot_df["__bar_xpos"] = positions

        # Determine style keys. series_by overrides the default legend/style grouping.
        style_spec = self._bar_style_spec(spec, group_cols, subgroup_cols)
        style_cols = self._unique_preserve_order(
            [
                *self._as_list(style_spec.series_by),
                *self._as_list(style_spec.hue_by),
                *self._as_list(style_spec.shade_by),
                *self._as_list(style_spec.pattern_by),
            ]
        )

        style_keys = [self._row_key(row, style_cols) for _, row in plot_df.iterrows()]
        unique_style_keys = self._unique_preserve_order(style_keys)
        styles = self._series_styles(
            unique_style_keys,
            style_cols,
            style_spec,
            include_pattern=True,
        )

        legend_cols = self._as_list(style_spec.series_by)
        if not legend_cols:
            legend_cols = self._unique_preserve_order(
                [
                    *self._as_list(style_spec.hue_by),
                    *self._as_list(style_spec.shade_by),
                    *self._as_list(style_spec.pattern_by),
                ]
            )

        seen_legend_labels = set()
        ax.set_axisbelow(True)

        for i, row in plot_df.iterrows():
            style_key = style_keys[i]
            style = styles[style_key]

            yerr = None
            if spec.yerr is not None and pd.notna(row[spec.yerr]):
                yerr = row[spec.yerr]

            legend_key = self._row_key(row, legend_cols)
            legend_label = self._key_label(legend_cols, legend_key, spec, include_names=True)
            if not legend_label:
                legend_label = None
            elif legend_label in seen_legend_labels:
                legend_label = "_nolegend_"
            else:
                seen_legend_labels.add(legend_label)

            # Hatches are drawn using the patch edge color in Matplotlib.
            # If a pattern is requested and no explicit edge color was supplied,
            # use a thin black edge so the pattern survives grayscale printing.
            hatch = style.get("hatch", "")
            edgecolor = spec.bar_edgecolor
            linewidth = spec.bar_linewidth
            if hatch and edgecolor is None:
                edgecolor = "black"
                linewidth = max(linewidth, 0.35)

            bars = ax.bar(
                row["__bar_xpos"],
                row[spec.y],
                width=spec.bar_width,
                yerr=yerr,
                capsize=spec.bar_error_capsize if yerr is not None else 0,
                color=style["color"],
                edgecolor=edgecolor,
                linewidth=linewidth,
                hatch=hatch,
                alpha=spec.bar_alpha,
                label=legend_label,
                zorder=3,
            )

            if spec.bar_value_labels:
                ax.bar_label(
                    bars,
                    labels=[self._format_bar_value(row[spec.y], spec.bar_value_fmt)],
                    padding=spec.bar_value_padding,
                    fontsize=spec.bar_value_fontsize,
                    rotation=spec.bar_value_rotation,
                )

        # X labels.
        mode = spec.bar_label_mode.lower()
        if mode not in {"hierarchical", "flat", "leaf", "none"}:
            raise ValueError("bar_label_mode must be one of: hierarchical, flat, leaf, none")

        ax.set_xticks(plot_df["__bar_xpos"])
        if mode == "none":
            ax.set_xticklabels([])
        elif mode == "flat":
            labels = [
                "\n".join(self._display_value(col, row[col], spec) for col in hierarchy_cols)
                for _, row in plot_df.iterrows()
            ]
            ax.set_xticklabels(
                labels,
                rotation=spec.bar_tick_label_rotation,
                ha="right" if spec.bar_tick_label_rotation else "center",
                fontsize=spec.bar_tick_label_fontsize,
            )
        else:
            labels = [self._display_value(leaf_col, row[leaf_col], spec) for _, row in plot_df.iterrows()]
            ax.set_xticklabels(
                labels,
                rotation=spec.bar_tick_label_rotation,
                ha="right" if spec.bar_tick_label_rotation else "center",
                fontsize=spec.bar_tick_label_fontsize,
            )

        if mode == "hierarchical":
            # The normal tick label is the leaf, for example variant.
            # Group/subgroup labels are optional display layers.
            y_level = -0.12

            show_subgroup_labels = bool(subgroup_cols and spec.bar_show_subgroup_labels)
            show_group_labels = bool(group_cols and spec.bar_show_group_labels)

            if show_subgroup_labels:
                for subgroup_key, xs in subgroup_positions.items():
                    label_key = subgroup_key

                    if group_cols:
                        # Strip the group part so subgroup label is not repeated as
                        # "BFS / CPU"; we only want "CPU".
                        if isinstance(subgroup_key, tuple):
                            label_key = subgroup_key[len(group_cols):]
                            if len(label_key) == 1:
                                label_key = label_key[0]

                    ax.text(
                        float(np.mean(xs)),
                        y_level,
                        self._key_label(subgroup_cols, label_key, spec),
                        ha="center",
                        va="top",
                        fontsize=spec.bar_subgroup_label_fontsize,
                        transform=ax.get_xaxis_transform(),
                        clip_on=False,
                    )

                y_level -= 0.12

            if group_cols:
                prev_last = None

                for group_key, xs in group_positions.items():
                    if show_group_labels:
                        ax.text(
                            float(np.mean(xs)),
                            y_level,
                            self._key_label(group_cols, group_key, spec),
                            ha="center",
                            va="top",
                            fontsize=spec.bar_group_label_fontsize,
                            transform=ax.get_xaxis_transform(),
                            clip_on=False,
                        )

                    # Keep separators independent from whether the group label is printed.
                    if spec.bar_show_group_separators and prev_last is not None:
                        boundary = (prev_last + xs[0]) / 2.0
                        ax.axvline(boundary, linewidth=0.8, alpha=0.35, zorder=0)

                    prev_last = xs[-1]

            # In single-plot mode we can reserve bottom margin exactly like the
            # original `_bar()` implementation. In panel mode, use plot_panels'
            # row_gap/layout_h_pad controls instead so one subplot does not
            # globally distort the whole figure.
            if adjust_figure_bottom:
                ax.figure.subplots_adjust(bottom=spec.bar_multilevel_bottom)

        # Bar extrema highlighting uses numeric positions, not the categorical leaf values.
        if spec.highlight_lowest_by is not None or spec.highlight_highest_by is not None:
            highlight_spec = replace(spec, x="__bar_xpos")
            self._apply_highlight_extrema(plot_df, ax, highlight_spec)

    def _bubble(self, df: pd.DataFrame, spec: PlotSpec) -> None:
        fig, ax = plt.subplots(figsize=spec.figsize, layout="constrained")
        title = "\n".join(textwrap.wrap(spec.title, width=spec.title_wrap))
        fig.suptitle(title, fontsize=spec.title_fontsize)

        self._draw_bubble(ax, df, spec)
        self._finish(fig, ax, spec)

    def _draw_bubble(self, ax, df: pd.DataFrame, spec: PlotSpec) -> None:
        if spec.size_by is None:
            raise ValueError("Bubble plots require PlotSpec.size_by")

        sizes = df[spec.size_by].astype(float)
        sizes = 50 + 450 * (sizes - sizes.min()) / max(float(sizes.max() - sizes.min()), 1e-12)
        series_cols = self._series_cols(spec)

        if not series_cols:
            ax.scatter(df[spec.x], df[spec.y], s=sizes, alpha=0.65)
        else:
            for key, part in df.groupby(series_cols, sort=True, dropna=False):
                idx = part.index
                ax.scatter(
                    part[spec.x],
                    part[spec.y],
                    s=sizes.loc[idx],
                    alpha=0.65,
                    label=self._series_label(series_cols, key),
                )

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

            if count > 0 and spec.highlight_lowest_label is not None:
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

            if count > 0 and spec.highlight_highest_label is not None:
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

def make_algorithm_stats_table(
    agg: pd.DataFrame,
    device: Literal["CPU", "GPU"],
    *,
    include_test_name: bool = False,
    decimals: int = 3,
) -> pd.DataFrame:
    """
    Build a compact algorithm/frequency summary table.

    CPU rows become:
        Algorithm, device, cpu_frequency_mhz,
        serial_kernel_runtime_ms, serial_run_power_w,
        openmp_kernel_runtime_ms, openmp_run_power_w,
        pocl_kernel_runtime_ms, pocl_run_power_w

    GPU rows become:
        Algorithm, device, gpu_frequency_mhz,
        opencl_kernel_runtime_ms, opencl_run_power_w
    """

    device = device.upper()

    if device not in {"CPU", "GPU"}:
        raise ValueError("device must be 'CPU' or 'GPU'")

    variant_order = {
        "CPU": ["Serial", "OpenMP", "PoCL"],
        "GPU": ["clvk", "OpenCL"],
    }[device]

    metrics = ["kernel_runtime", "run_power_w"]

    required = [
        "Algorithm",
        "frequency_component",
        "operating_frequency_mhz",
        "variant",
        *metrics,
    ]

    missing = [c for c in required if c not in agg.columns]
    if missing:
        raise KeyError(f"Cannot build summary table; missing columns: {missing}")

    work = agg.copy()

    work = work[
        work["frequency_component"].eq(device)
        & work["variant"].isin(variant_order)
    ].copy()

    if work.empty:
        return pd.DataFrame()

    index_cols = ["Algorithm", "frequency_component", "operating_frequency_mhz"]

    if include_test_name and "test_name" in work.columns:
        index_cols.insert(1, "test_name")

    table = work.pivot_table(
        index=index_cols,
        columns="variant",
        values=metrics,
        aggfunc="mean",
    )

    # Convert from metric-major columns:
    #   kernel_runtime / Serial
    # to variant-major columns:
    #   Serial / kernel_runtime
    table = table.swaplevel(0, 1, axis=1)

    ordered_cols = [
        (variant, metric)
        for variant in variant_order
        for metric in metrics
        if (variant, metric) in table.columns
    ]

    table = table[ordered_cols]

    metric_names = {
        "kernel_runtime": "kernel_runtime_ms",
        "run_power_w": "run_power_w",
    }

    table.columns = [
        f"{variant.lower()}_{metric_names[metric]}"
        for variant, metric in table.columns
    ]

    table = table.reset_index()

    freq_name = "cpu_frequency_mhz" if device == "CPU" else "gpu_frequency_mhz"

    table = table.rename(
        columns={
            "frequency_component": "device",
            "operating_frequency_mhz": freq_name,
        }
    )

    # Put columns in the exact readable order.
    front_cols = ["Algorithm"]
    if include_test_name and "test_name" in table.columns:
        front_cols.append("test_name")

    front_cols += ["device", freq_name]

    remaining_cols = [c for c in table.columns if c not in front_cols]
    table = table[front_cols + remaining_cols]

    numeric_cols = table.select_dtypes(include="number").columns
    table[numeric_cols] = table[numeric_cols].round(decimals)

    return table.sort_values(front_cols).reset_index(drop=True)