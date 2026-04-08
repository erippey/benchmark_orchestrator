import json
import csv
from pathlib import Path


class RunLogger:

    def __init__(self, root_dir):

        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def new_run_dir(self, test_name, tag, date):

        today = date

        base = self.root / today / f"{test_name}" / f"{tag}"
        base.mkdir(parents=True, exist_ok=True)

        run_id = 1
        while (base / f"run{run_id}").exists():
            run_id += 1

        run_dir = base / f"run{run_id}"
        run_dir.mkdir()

        return run_dir

    def write_metadata(self, run_dir, data):

        with open(run_dir / "metadata.json", "w") as f:
            json.dump(data, f, indent=2)

    def open_power_log(self, run_dir, name):

        path = run_dir / name

        f = open(path, "w", newline="")
        writer = csv.writer(f)

        writer.writerow(["timestamp", "watts", "volts", "amps"])

        return f, writer