

import csv
import re
from pathlib import Path


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

ALL_MANAGED_KEYS = [
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
    "Config_Name",
    "Config_ID",
    "cores_online",
    "cpu_freq_max",
    "cpu_freq_min",
    "gpu_freq_max",
    "gpu_freq_min",
    "emc_max_freq",
]

RUN_METADATA_KEYS = [
    "Signal_Length",
    "Filter_Length",
    "Backend",
    "Platform",
    "N_FFT",
    "Batch_Size",
    "Algorithm_Data_Type",
    "Sample_Format",
]


PERFORMANCE_KEYS = [
    "conv_avg_ms",
    "total_exec_ms",
]

POWER_KEYS = [
    "idle_power_w",
    "run_power_w",
]


class BenchmarkAggregator:

    def __init__(self, root, additional_keys=[], config_keys=PI_MANAGED_KEYS):
        self.root = Path(root)
        self.rows = []

        self.config_managed_keys = config_keys
        self.additional_keys = additional_keys

    # -----------------------------
    # Directory discovery
    # -----------------------------

    def walk_runs(self):

        def walk_iter(current_dir):
            for sub_dir in current_dir.iterdir():
                if not sub_dir.is_dir():
                    continue
                
                if sub_dir.name.startswith("run"):
                    yield sub_dir

                yield from walk_iter(sub_dir)

        
        yield from walk_iter(self.root)



    # -----------------------------
    # Metadata parser
    # -----------------------------

    def parse_run_metadata(self, path):
        columns = {}

        if not path.exists():
            return columns

        key_map = {
            "Length": "Signal_Length",
            "FilterLength": "Filter_Length",
            "Backend": "Backend",
            "Platform": "Platform",
            "N_FFT": "N_FFT",
            "BatchSize": "Batch_Size",
            "Algorithmdatatype": "Algorithm_Data_Type",
            "Sampleformat": "Sample_Format",
        }

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if ":" not in line:
                    continue

                key, val = line.replace(" ", "").split(":", 1)

                if key in key_map:
                    out_key = key_map[key]

                    try:
                        columns[out_key] = int(val)
                    except ValueError:
                        columns[out_key] = val

        return columns
                        

    # -----------------------------
    # config.txt parser
    # -----------------------------

    def parse_config_metadata(self, path):

        values = {k: 0 for k in self.config_managed_keys}

        if not path.exists():
            return values

        with open(path) as f:
            for line in f:
                line = line.strip()

                if line.startswith("#"):
                    continue
                    
                if line.startswith("//"):
                    continue

                if ":" not in line:
                    continue

                key, val = line.replace(" ","").split(":", 1)

                if key in values:
                    try:
                        values[key] = int(val)
                    except:
                        values[key] = val

        return values

    # -----------------------------
    # stdout event parser
    # -----------------------------

    def parse_stdout(self, path):

        events = {}
        total_exec_ms = None

        with open(path) as f:
            lines = f.readlines()

        current = None

        for line in lines:

            line = line.strip()

            m = re.match(r"Event\s+\d+:\s+(.+)", line)

            if m:
                current = m.group(1)
                events[current] = {}
                continue

            if not current:
                continue

            if "Runs" in line:
                events[current]["runs"] = int(line.split(":")[1])

            elif "Avg time" in line:
                val = float(line.split(":")[1].split()[0])
                events[current]["avg_time_ms"] = val

            elif "Min time" in line:
                val = line.split(":")[1].split()[0]
                events[current]["min_time_ms"] = val

            elif "Max time" in line:
                val = line.split(":")[1].split()[0]
                events[current]["max_time_ms"] = val

            elif "Std Dev" in line:
                val = float(line.split(":")[1].split()[0])
                events[current]["std_dev_ms"] = val

        if "Total Execution" in events:
            total_exec_ms = events["Total Execution"]["avg_time_ms"]

        return events, total_exec_ms

    # -----------------------------
    # power helpers
    # -----------------------------

    def avg_first_n_watts(self, path, n):

        watts = []

        with open(path) as f:
            reader = csv.DictReader(f)

            for i, row in enumerate(reader):
                if i >= n:
                    break

                watts.append(float(row["watts"]))

        return sum(watts) / len(watts) if watts else 0

    # -----------------------------
    # run aggregation
    # -----------------------------

    def process_run(self, run_dir):

        row = {"run_dir": str(run_dir)}

        # metadata
        meta_file = run_dir / "stdout_dump.txt"
        run_md = self.parse_run_metadata(run_dir / "metadata.txt")
        row.update(run_md)

        # config
        config_md = self.parse_config_metadata(run_dir / "config_metadata.txt")
        row.update(config_md)

        # stdout events
        events, total_exec_ms = self.parse_stdout(meta_file)

        if "Convolution" in events:
            row["conv_avg_ms"] = events["Convolution"]["avg_time_ms"]

        if total_exec_ms:
            row["total_exec_ms"] = total_exec_ms

        if self.additional_keys:
            for key in self.additional_keys:
                if key in events:
                    row[key] = events[key]["avg_time_ms"]

        # idle power
        idle_csv = run_dir / "idle_power.csv"
        if idle_csv.exists():
            row["idle_power_w"] = self.avg_first_n_watts(idle_csv, 60)

        # run power
        if total_exec_ms:
            run_seconds = round(total_exec_ms / 1000)

            run_power_csv = run_dir / "run_power.csv"

            if run_power_csv.exists():
                row["run_power_w"] = self.avg_first_n_watts(
                    run_power_csv, run_seconds
                )

        return row

    # -----------------------------
    # master aggregation
    # -----------------------------

    def aggregate(self):

        for run_dir in self.walk_runs():
            try:
                row = self.process_run(run_dir)
                self.rows.append(row)
            except Exception as e:
                print("Skipping", run_dir, e)

    # -----------------------------
    # write CSV
    # -----------------------------

    def write_csv(self, out):

        if not self.rows:
            return
        
        fieldnames = (
            ["run_dir"]
            + RUN_METADATA_KEYS
            + self.config_managed_keys
            + PERFORMANCE_KEYS
            + self.additional_keys
            + POWER_KEYS
        ) 


        with open(out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)


# -----------------------------
# usage
# -----------------------------

if __name__ == "__main__":

    ADDITIONAL_KEYS = [
        "Forward FFT Execution Time",
        "Complex Multiply Execution Time",
        "Inverse FFT Execution Time",
        "Overlap Add"
    ]



    agg = BenchmarkAggregator("bench_logs", ADDITIONAL_KEYS, ALL_MANAGED_KEYS)

    agg.aggregate()

    agg.write_csv("aggregated_csv/aggregated_results.csv")