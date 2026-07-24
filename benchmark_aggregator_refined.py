#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import Dict, Iterable, Mapping, Optional


PI_MANAGED_KEYS = [
    "TestName",
    "Device",
    "Governor",
    "arm_freq",
    "arm_freq_min",
    "core_freq",
    "core_freq_min",
    "v3d_freq",
    "v3d_freq_min",
    "gpu_freq",
    "gpu_freq_min",
    "over_voltage",
    "over_voltage_min",
]

NANO_MANAGED_KEYS = [
    "TestName",
    "Device",
    "Governor",
    "Config_Name",
    "Config_ID",
    "cores_online",
    "cpu_freq_max",
    "cpu_freq_min",
    "gpu_freq_max",
    "gpu_freq_min",
    "emc_max_freq",
]

OPI_MANAGED_KEYS = [
    "TestName",
    "Device",
    "cpu_governor",
    "cpu_freq",
    "cpu_freq_min",
    "cpu_freq_max",

    # Backward-compatible generic governor.
    "governor",

    # Per-policy CPU controls
    "policy0_freq",
    "policy0_freq_min",
    "policy0_freq_max",
    "policy4_freq",
    "policy4_freq_min",
    "policy4_freq_max",
    "policy6_freq",
    "policy6_freq_min",
    "policy6_freq_max",

    # GPU controls
    "gpu_governor",
    "gpu_freq",
    "gpu_freq_min",
    "gpu_freq_max",

    # DMC controls
    "dmc_governor",
    "dmc_freq",
    "dmc_freq_min",
    "dmc_freq_max",
]

# dict.fromkeys preserves the first-seen order, unlike set(...).
ALL_MANAGED_KEYS = list(dict.fromkeys(
    PI_MANAGED_KEYS + NANO_MANAGED_KEYS + OPI_MANAGED_KEYS
))

RUN_METADATA_KEYS = [
    "Platform",
    "Device",
    "Algorithm",
    "Threads",
    "Signal_Length",
    "Filter_Length",
    "Backend",
    "N_FFT",
    "Banks",
    "Channels",
]

POWER_KEYS = [
    "idle_power_w",
    "run_power_w",
]


@dataclass(frozen=True)
class InputFileNames:
    """File names expected inside each run directory.

    A value of None disables aggregation for that file type.
    """

    run_metadata: Optional[str] = "metadata.txt"
    config_metadata: Optional[str] = "config_metadata.txt"
    idle_power: Optional[str] = "idle_power.csv"
    run_power: Optional[str] = "run_power.csv"
    benchmark_log: Optional[str] = None
    runtime_stats: Optional[str] = None


@dataclass(frozen=True)
class RuntimeStatSpec:
    column: str
    value_kind: str  # "count", "number", or "duration_ms"


DEFAULT_BENCHMARK_EVENT_COLUMNS = {
    "Total Execution": "total_exec_ms",
    "Convolution": "conv_avg_ms",
}

DEFAULT_RUNTIME_STAT_SPECS = {
    "Invocations": RuntimeStatSpec("invocations", "count"),
    "Deadlines Missed": RuntimeStatSpec("deadlines_missed", "count"),
    "Missed periods": RuntimeStatSpec("missed_periods", "count"),
    "Average execution": RuntimeStatSpec("average_execution_ms", "duration_ms"),
    "Maximum execution": RuntimeStatSpec("maximum_execution_ms", "duration_ms"),
    "Maximum start lateness": RuntimeStatSpec(
        "maximum_start_lateness_ms", "duration_ms"
    ),
}

NUMBER_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
NUMBER_AND_UNIT_RE = re.compile(
    rf"^\s*(?P<number>{NUMBER_PATTERN})(?:\s+(?P<unit>\S+))?\s*$"
)
BENCHMARK_EVENT_RE = re.compile(r"^Event\s+\d+\s*:\s*(.+?)\s*$")
BENCHMARK_METRIC_RE = re.compile(
    r"^(Runs|Avg time|Min time|Max time|Std Dev)\s*:\s*(.+?)\s*$"
)


class BenchmarkAggregator:

    def __init__(
        self,
        root,
        additional_benchmark_events: Optional[Iterable[str]] = None,
        config_keys: Optional[Iterable[str]] = None,
        *,
        input_files: Optional[InputFileNames] = None,
        benchmark_event_columns: Optional[Mapping[str, str]] = None,
        runtime_stat_specs: Optional[Mapping[str, RuntimeStatSpec]] = None,
        idle_sample_count: int = 60,
        run_power_tail_samples: int = 5,
    ):
        self.root = Path(root)
        self.rows = []

        self.config_managed_keys = list(
            PI_MANAGED_KEYS if config_keys is None else config_keys
        )
        self.input_files = input_files or InputFileNames()
        self.idle_sample_count = idle_sample_count
        self.run_power_tail_samples = run_power_tail_samples

        self.benchmark_event_columns = dict(
            DEFAULT_BENCHMARK_EVENT_COLUMNS
            if benchmark_event_columns is None
            else benchmark_event_columns
        )

        # Additional benchmark event names become stable snake_case CSV columns.
        for event_name in additional_benchmark_events or []:
            self.benchmark_event_columns.setdefault(
                event_name,
                f"{self._to_snake_case(event_name)}_avg_ms",
            )

        self.runtime_stat_specs = dict(
            DEFAULT_RUNTIME_STAT_SPECS
            if runtime_stat_specs is None
            else runtime_stat_specs
        )

        self._runtime_specs_by_normalized_name = {
            self._normalize_label(name): spec
            for name, spec in self.runtime_stat_specs.items()
        }

    # -----------------------------
    # General helpers
    # -----------------------------

    @staticmethod
    def _normalize_label(value: str) -> str:
        return " ".join(value.strip().lower().split())

    @staticmethod
    def _to_snake_case(value: str) -> str:
        value = re.sub(r"[^0-9A-Za-z]+", "_", value.strip())
        return value.strip("_").lower()

    @staticmethod
    def _parse_scalar(value: str):
        value = value.strip()

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            return value

    @staticmethod
    def _parse_number_and_unit(value: str):
        match = NUMBER_AND_UNIT_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"Expected a numeric value, got {value!r}")

        number = float(match.group("number"))
        unit = match.group("unit")
        return number, unit

    @staticmethod
    def _duration_to_ms(value: float, unit: Optional[str]) -> float:
        # Existing benchmark and runtime-stat formats use milliseconds. Treat a
        # missing unit as milliseconds while accepting common alternatives.
        if unit is None:
            return value

        normalized = unit.strip().lower().rstrip(".,")
        factors = {
            "s": 1000.0,
            "sec": 1000.0,
            "second": 1000.0,
            "seconds": 1000.0,
            "ms": 1.0,
            "msec": 1.0,
            "millisecond": 1.0,
            "milliseconds": 1.0,
            "us": 1.0e-3,
            "µs": 1.0e-3,
            "microsecond": 1.0e-3,
            "microseconds": 1.0e-3,
            "ns": 1.0e-6,
            "nanosecond": 1.0e-6,
            "nanoseconds": 1.0e-6,
        }

        try:
            return value * factors[normalized]
        except KeyError as exc:
            raise ValueError(f"Unsupported duration unit: {unit!r}") from exc

    @staticmethod
    def _resolve_run_file(
        run_dir: Path, filename: Optional[str]
    ) -> Optional[Path]:
        return None if filename is None else run_dir / filename

    # -----------------------------
    # Directory discovery
    # -----------------------------

    def walk_runs(self):
        if not self.root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.root}")

        if self.root.is_dir() and self.root.name.startswith("run"):
            yield self.root

        for path in sorted(self.root.rglob("*")):
            if path.is_dir() and path.name.startswith("run"):
                yield path

    # -----------------------------
    # Run metadata parser
    # -----------------------------

    def parse_run_metadata(self, path: Optional[Path]) -> Dict[str, object]:
        columns: Dict[str, object] = {}

        if path is None or not path.exists():
            return columns

        key_map = {
            self._normalize_label(key.replace("_", " ")): key
            for key in RUN_METADATA_KEYS
        }

        with path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith(("#", "//")) or ":" not in line:
                    continue

                key, value = line.split(":", 1)
                output_key = key_map.get(
                    self._normalize_label(key.replace("_", " "))
                )
                if output_key is not None:
                    columns[output_key] = self._parse_scalar(value)

        return columns

    # -----------------------------
    # Configuration metadata parser
    # -----------------------------

    def parse_config_metadata(self, path: Optional[Path]) -> Dict[str, object]:
        values: Dict[str, object] = {
            key: 0 for key in self.config_managed_keys
        }

        if path is None or not path.exists():
            return values

        with path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()
                if not line or line.startswith(("#", "//")) or ":" not in line:
                    continue

                key, value = line.split(":", 1)
                key = key.strip()

                if key in values:
                    values[key] = self._parse_scalar(value)

        return values

    # -----------------------------
    # Benchmark-framework log parser
    # -----------------------------

    def parse_benchmark_log(self, path: Optional[Path]) -> Dict[str, dict]:
        """Parse the Event/Avg time benchmark-framework text format."""

        events: Dict[str, dict] = {}

        if path is None or not path.exists():
            return events

        current_event = None
        metric_columns = {
            "Runs": "runs",
            "Avg time": "avg_time_ms",
            "Min time": "min_time_ms",
            "Max time": "max_time_ms",
            "Std Dev": "std_dev_ms",
        }

        with path.open("r", encoding="utf-8") as file:
            for raw_line in file:
                line = raw_line.strip()

                event_match = BENCHMARK_EVENT_RE.fullmatch(line)
                if event_match is not None:
                    current_event = event_match.group(1)
                    events[current_event] = {}
                    continue

                if current_event is None:
                    continue

                metric_match = BENCHMARK_METRIC_RE.fullmatch(line)
                if metric_match is None:
                    continue

                metric_name, raw_value = metric_match.groups()
                output_column = metric_columns[metric_name]
                number, unit = self._parse_number_and_unit(raw_value)

                if metric_name == "Runs":
                    if not number.is_integer():
                        raise ValueError(
                            f"Runs must be an integer in {path}: {raw_value!r}"
                        )
                    events[current_event][output_column] = int(number)
                else:
                    events[current_event][output_column] = (
                        self._duration_to_ms(number, unit)
                    )

        return events

    # -----------------------------
    # Runtime-statistics parser
    # -----------------------------

    def parse_runtime_stats(self, path: Optional[Path]) -> Dict[str, object]:
        """Parse selected key/value runtime statistics into normalized columns."""

        values: Dict[str, object] = {}

        if path is None or not path.exists():
            return values

        with path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                line = raw_line.strip()
                if not line or line.startswith(("#", "//")) or ":" not in line:
                    continue

                input_name, raw_value = line.split(":", 1)
                spec = self._runtime_specs_by_normalized_name.get(
                    self._normalize_label(input_name)
                )
                if spec is None:
                    continue

                try:
                    number, unit = self._parse_number_and_unit(raw_value)

                    if spec.value_kind == "count":
                        if unit is not None or not number.is_integer():
                            raise ValueError("expected a unitless integer")
                        parsed_value = int(number)
                    elif spec.value_kind == "number":
                        if unit is not None:
                            raise ValueError("expected a unitless number")
                        parsed_value = number
                    elif spec.value_kind == "duration_ms":
                        parsed_value = self._duration_to_ms(number, unit)
                    else:
                        raise ValueError(
                            f"unknown value kind {spec.value_kind!r}"
                        )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid runtime statistic at {path}:{line_number}: "
                        f"{line!r} ({exc})"
                    ) from exc

                values[spec.column] = parsed_value

        return values

    # -----------------------------
    # Power helpers
    # -----------------------------

    def average_first_n_watts(self, path: Path, count: int) -> float:
        watts = []

        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for index, row in enumerate(reader):
                if index >= count:
                    break
                watts.append(float(row["watts"]))

        return sum(watts) / len(watts) if watts else 0.0

    def average_watts_excluding_tail(
        self, path: Path, tail_count: int
    ) -> float:
        rows = []

        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows.append({
                    "timestamp": datetime.fromisoformat(row["timestamp"]),
                    "watts": float(row["watts"]),
                })

        rows.sort(key=lambda row: row["timestamp"])

        if tail_count > 0:
            rows = rows[:-tail_count]

        watts = [row["watts"] for row in rows]
        return sum(watts) / len(watts) if watts else 0.0

    # -----------------------------
    # Run aggregation
    # -----------------------------

    def process_run(self, run_dir: Path) -> Dict[str, object]:
        row: Dict[str, object] = {"run_dir": str(run_dir)}

        # Apply configuration first so explicit per-run metadata wins when
        # both sources contain a shared field such as Device or TestName.
        config_metadata_path = self._resolve_run_file(
            run_dir, self.input_files.config_metadata
        )
        row.update(self.parse_config_metadata(config_metadata_path))

        run_metadata_path = self._resolve_run_file(
            run_dir, self.input_files.run_metadata
        )
        row.update(self.parse_run_metadata(run_metadata_path))

        benchmark_log_path = self._resolve_run_file(
            run_dir, self.input_files.benchmark_log
        )
        benchmark_events = self.parse_benchmark_log(benchmark_log_path)

        for event_name, output_column in self.benchmark_event_columns.items():
            average_ms = benchmark_events.get(event_name, {}).get("avg_time_ms")
            if average_ms is not None:
                row[output_column] = average_ms

        runtime_stats_path = self._resolve_run_file(
            run_dir, self.input_files.runtime_stats
        )
        row.update(self.parse_runtime_stats(runtime_stats_path))

        idle_power_path = self._resolve_run_file(
            run_dir, self.input_files.idle_power
        )
        if idle_power_path is not None and idle_power_path.exists():
            row["idle_power_w"] = self.average_first_n_watts(
                idle_power_path, self.idle_sample_count
            )

        run_power_path = self._resolve_run_file(
            run_dir, self.input_files.run_power
        )
        if run_power_path is not None and run_power_path.exists():
            row["run_power_w"] = self.average_watts_excluding_tail(
                run_power_path, self.run_power_tail_samples
            )

        return row

    # -----------------------------
    # Master aggregation
    # -----------------------------

    def aggregate(self):
        self.rows.clear()

        for run_dir in self.walk_runs():
            try:
                self.rows.append(self.process_run(run_dir))
            except Exception as exc:
                print(f"Skipping {run_dir}: {exc}", file=sys.stderr)

        return self.rows

    # -----------------------------
    # CSV output
    # -----------------------------

    def _output_fieldnames(self):
        configured_columns = (
            ["run_dir"]
            + RUN_METADATA_KEYS
            + self.config_managed_keys
            + list(self.benchmark_event_columns.values())
            + [spec.column for spec in self.runtime_stat_specs.values()]
            + POWER_KEYS
        )

        fieldnames = list(dict.fromkeys(configured_columns))
        known = set(fieldnames)

        # Preserve any future parser-produced fields without making the writer
        # fail because a new column was not manually added here.
        for row in self.rows:
            for key in row:
                if key not in known:
                    fieldnames.append(key)
                    known.add(key)

        return fieldnames

    def write_csv(self, output_path):
        if not self.rows:
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self._output_fieldnames())
            writer.writeheader()
            writer.writerows(self.rows)

        return True


def optional_filename(value: str) -> Optional[str]:
    """Allow 'none' or '-' to disable a default input file."""

    if value.strip().lower() in {"none", "null", "-"}:
        return None
    return value


def parse_benchmark_event_argument(value: str):
    """Parse EVENT or EVENT=COLUMN from --benchmark-event."""

    if "=" in value:
        event_name, column = value.split("=", 1)
        event_name = event_name.strip()
        column = column.strip()
        if not event_name or not column:
            raise argparse.ArgumentTypeError(
                "Expected EVENT or EVENT=COLUMN"
            )
        return event_name, column

    event_name = value.strip()
    if not event_name:
        raise argparse.ArgumentTypeError("Benchmark event name cannot be empty")

    column = f"{BenchmarkAggregator._to_snake_case(event_name)}_avg_ms"
    return event_name, column


def build_argument_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate metadata, benchmark logs, runtime statistics, and "
            "power logs from run directories into one CSV."
        )
    )

    parser.add_argument(
        "-i", "--input-dir",
        default="bench_logs/upols_stream_tests",
        help="Root directory containing run* directories",
    )
    parser.add_argument(
        "-o", "--output-file",
        default="aggregated_csv/streaming_conv_results.csv",
        help="Aggregated CSV output path",
    )

    parser.add_argument(
        "--run-metadata-file",
        type=optional_filename,
        default="metadata.txt",
        help="Run metadata filename relative to each run directory",
    )
    parser.add_argument(
        "--config-metadata-file",
        type=optional_filename,
        default="config_metadata.txt",
        help="Configuration metadata filename relative to each run directory",
    )
    parser.add_argument(
        "--idle-power-file",
        type=optional_filename,
        default="idle_power.csv",
        help="Idle-power CSV filename relative to each run directory",
    )
    parser.add_argument(
        "--run-power-file",
        type=optional_filename,
        default="run_power.csv",
        help="Run-power CSV filename relative to each run directory",
    )
    parser.add_argument(
        "--benchmark-log-file",
        type=optional_filename,
        default=None,
        help=(
            "Benchmark-framework log filename. Disabled unless supplied; "
            "for the current layout, pass stdout_dump.txt"
        ),
    )
    parser.add_argument(
        "--runtime-stats-file",
        type=optional_filename,
        default=None,
        help=(
            "Key/value runtime-statistics filename. Disabled unless supplied"
        ),
    )
    parser.add_argument(
        "--benchmark-event",
        action="append",
        type=parse_benchmark_event_argument,
        default=[],
        metavar="EVENT[=COLUMN]",
        help=(
            "Additional benchmark event to aggregate by average time. "
            "Repeat as needed; COLUMN defaults to snake_case_event_avg_ms"
        ),
    )

    return parser


def main(argv=None) -> int:
    args = build_argument_parser().parse_args(argv)

    benchmark_event_columns = dict(DEFAULT_BENCHMARK_EVENT_COLUMNS)
    benchmark_event_columns.update(dict(args.benchmark_event))

    input_files = InputFileNames(
        run_metadata=args.run_metadata_file,
        config_metadata=args.config_metadata_file,
        idle_power=args.idle_power_file,
        run_power=args.run_power_file,
        benchmark_log=args.benchmark_log_file,
        runtime_stats=args.runtime_stats_file,
    )

    aggregator = BenchmarkAggregator(
        args.input_dir,
        config_keys=ALL_MANAGED_KEYS,
        input_files=input_files,
        benchmark_event_columns=benchmark_event_columns,
    )
    aggregator.aggregate()

    if not aggregator.write_csv(args.output_file):
        print("No run directories were aggregated; no CSV was written.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
