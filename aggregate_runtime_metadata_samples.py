#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


RUN_RE = re.compile(r"^run[_-]?(\d+)$", re.IGNORECASE)


def natural_key(path: Path) -> list[object]:
    """
    Sort paths naturally so run2 comes before run10.
    """
    text = str(path)
    parts = re.split(r"(\d+)", text)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def run_number(run_dir: Path) -> int | None:
    m = RUN_RE.match(run_dir.name)
    if not m:
        return None
    return int(m.group(1))


def is_valid_run(
    run_dir: Path,
    *,
    metadata_name: str,
    require_stdout: bool,
) -> bool:
    metadata_path = run_dir / metadata_name
    if not metadata_path.is_file():
        return False

    if require_stdout and not (run_dir / "stdout_dump.txt").is_file():
        return False

    return True


def discover_run_dirs(root: Path) -> dict[Path, list[Path]]:
    """
    Returns:
        independent_var_dir -> [run dirs]

    Expected shape:

        root/.../<date>/<test_name>/<independent_var_val>/run1/
        root/.../<date>/<test_name>/<independent_var_val>/run2/

    This does not care whether there is a test-category directory before
    the date. It only relies on run directories being inside an
    independent-variable directory.
    """
    by_independent_var: dict[Path, list[Path]] = defaultdict(list)

    for p in root.rglob("*"):
        if not p.is_dir():
            continue

        if run_number(p) is None:
            continue

        independent_var_dir = p.parent
        by_independent_var[independent_var_dir].append(p)

    return by_independent_var


def choose_first_and_last_valid_runs(
    run_dirs: list[Path],
    *,
    metadata_name: str,
    require_stdout: bool,
) -> list[Path]:
    """
    Select up to two valid runs:
      - first valid run when searching upward from run1
      - last valid run when searching downward from the largest runN

    If only one valid run exists, it is returned once.
    """
    sorted_runs = sorted(run_dirs, key=lambda p: run_number(p) or 0)

    first_valid: Path | None = None
    for run_dir in sorted_runs:
        if is_valid_run(
            run_dir,
            metadata_name=metadata_name,
            require_stdout=require_stdout,
        ):
            first_valid = run_dir
            break

    last_valid: Path | None = None
    for run_dir in reversed(sorted_runs):
        if is_valid_run(
            run_dir,
            metadata_name=metadata_name,
            require_stdout=require_stdout,
        ):
            last_valid = run_dir
            break

    selected: list[Path] = []

    if first_valid is not None:
        selected.append(first_valid)

    if last_valid is not None and last_valid != first_valid:
        selected.append(last_valid)

    return selected


def copy_csv_with_repeated_header(
    out_f,
    csv_path: Path,
) -> None:
    """
    Copy the source CSV exactly as text. This preserves the header row for
    every runtime_metadata_samples.csv file, which is what you requested.
    """
    text = csv_path.read_text(encoding="utf-8", errors="replace")

    out_f.write(text)

    if not text.endswith("\n"):
        out_f.write("\n")


def aggregate_runtime_metadata(
    root: Path,
    output: Path,
    *,
    metadata_name: str = "runtime_metadata_samples.csv",
    require_stdout: bool = True,
    dry_run: bool = False,
) -> None:
    by_independent_var = discover_run_dirs(root)

    selected_by_test: dict[Path, list[Path]] = defaultdict(list)
    skipped_independent_vars: list[Path] = []

    for independent_var_dir, run_dirs in sorted(
        by_independent_var.items(),
        key=lambda item: natural_key(item[0]),
    ):
        selected_runs = choose_first_and_last_valid_runs(
            run_dirs,
            metadata_name=metadata_name,
            require_stdout=require_stdout,
        )

        if not selected_runs:
            skipped_independent_vars.append(independent_var_dir)
            continue

        # independent_var_dir is:
        #   .../<test_name>/<independent_var_val>
        #
        # Therefore the test directory is its parent.
        test_dir = independent_var_dir.parent

        selected_by_test[test_dir].extend(selected_runs)

    if dry_run:
        print("Selected runtime metadata files:\n")
        for test_dir, run_dirs in sorted(selected_by_test.items(), key=lambda item: natural_key(item[0])):
            print(f"[{test_dir.name}]")
            for run_dir in sorted(run_dirs, key=natural_key):
                print(f"  {run_dir / metadata_name}")
            print()

        if skipped_independent_vars:
            print("Skipped independent-variable directories with no valid runs:\n")
            for p in skipped_independent_vars:
                print(f"  {p}")

        return

    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f)

        for test_dir, run_dirs in sorted(selected_by_test.items(), key=lambda item: natural_key(item[0])):
            test_name = test_dir.name

            # Single-cell row containing just the individual test name.
            writer.writerow([test_name])

            for run_dir in sorted(run_dirs, key=natural_key):
                csv_path = run_dir / metadata_name
                copy_csv_with_repeated_header(out_f, csv_path)

    print(f"Wrote: {output}")
    print(f"Tests included: {len(selected_by_test)}")
    print(f"Runtime metadata files included: {sum(len(v) for v in selected_by_test.values())}")

    if skipped_independent_vars:
        print()
        print(f"Skipped independent-variable directories with no valid runs: {len(skipped_independent_vars)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate selected runtime_metadata_samples.csv files from benchmark run directories."
    )

    parser.add_argument(
        "root",
        type=Path,
        help="Root test directory to search.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("aggregated_csv/aggregated_runtime_metadata_samples.csv"),
        help="Output CSV path.",
    )

    parser.add_argument(
        "--metadata-name",
        default="runtime_metadata_samples.csv",
        help="Runtime metadata CSV filename.",
    )

    parser.add_argument(
        "--no-require-stdout",
        action="store_true",
        help="Allow runs even if stdout_dump.txt is missing.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected files without writing the aggregate CSV.",
    )

    args = parser.parse_args()

    aggregate_runtime_metadata(
        root=args.root,
        output=args.output,
        metadata_name=args.metadata_name,
        require_stdout=not args.no_require_stdout,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()