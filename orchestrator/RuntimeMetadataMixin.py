import csv
import re
import shlex
import statistics
import time
from pathlib import Path


class RuntimeMetadataMixin:
    MAX_CPU_POLICIES = 8

    COMMON_RUNTIME_COLUMNS = (
        ["sample_index", "timestamp_ns", "timestamp_s"]
        + [f"cpu_policy{i}_cur_khz" for i in range(MAX_CPU_POLICIES)]
        + [f"cpu_policy{i}_min_khz" for i in range(MAX_CPU_POLICIES)]
        + [f"cpu_policy{i}_max_khz" for i in range(MAX_CPU_POLICIES)]
        + [
            "gpu_cur_khz",
            "gpu_min_khz",
            "gpu_max_khz",
            "core_cur_khz",
            "memory_cur_khz",
            "memory_min_khz",
            "memory_max_khz",
            "npu_cur_khz",

            "cpu_temp_c",
            "gpu_temp_c",
            "soc_temp_c",
            "board_temp_c",
            "npu_temp_c",

            "voltage_cpu_mv",
            "voltage_gpu_mv",
            "voltage_core_mv",
            "voltage_memory_mv",
            "voltage_soc_mv",
            "voltage_npu_mv",

            "power_cpu_mw",
            "power_gpu_mw",
            "power_soc_mw",
            "power_memory_mw",
            "power_total_mw",

            "throttle_raw",
        ]
    )

    def __init__(self, device_manager):
        self.device_manager = device_manager

    @property
    def device_name(self):
        return self.device_manager.device_name

    @property
    def client(self):
        return self.device_manager.client

    def _run_shell(
        self,
        cmd,
        *,
        as_root=False,
        timeout=5,
        safe_to_retry=True,
    ):
        """
        Run a small metadata command on the remote DUT.

        This expects self.client.run(...) to return:

            code, stdout, stderr
        """

        if self.client is None:
            return ""

        try:
            code, out, err = self.client.run(
                cmd,
                as_root=as_root,
                async_run=False,
                retries=2,
                timeout=timeout,
                safe_to_retry=safe_to_retry,
            )

            if code != 0:
                return ""

            return out.strip()

        except Exception:
            return ""

    def _read_text(self, path, *, as_root=False):
        quoted_path = shlex.quote(str(path))
        return self._run_shell(
            f"cat {quoted_path} 2>/dev/null || true",
            as_root=as_root,
        ).strip()

    def _glob(self, pattern):
        out = self._run_shell(
            f'for p in {pattern}; do [ -e "$p" ] && printf "%s\\n" "$p"; done'
        )

        return [
            line.strip()
            for line in out.splitlines()
            if line.strip()
        ]

    def _read_first_existing_text(self, paths):
        for path in paths:
            value = self._read_text(path)
            if value != "":
                return value

        return ""

    @staticmethod
    def _safe_key(name):
        name = str(name).strip().lower()
        name = re.sub(r"[^a-z0-9]+", "_", name)
        name = name.strip("_")
        return name or "unknown"

    @staticmethod
    def _first_number(text):
        if text is None:
            return None

        match = re.search(r"[-+]?\d+(?:\.\d+)?", str(text))
        if not match:
            return None

        value = float(match.group(0))
        if value.is_integer():
            return int(value)

        return value

    def _read_number_file(self, path, default=0):
        value = self._first_number(self._read_text(path))
        return default if value is None else value

    def _read_cpufreq_khz(self, path):
        return int(self._read_number_file(path, 0))

    def _read_devfreq_khz(self, path):
        raw = int(self._read_number_file(path, 0))

        if raw <= 0:
            return 0

        # Most devfreq values are Hz. CPUFreq values are usually kHz.
        # If it is clearly Hz, convert to kHz.
        if raw >= 10_000_000:
            return raw // 1000

        return raw

    def save_runtime_metadata(
        self,
        params,
        root_dir,
        date,
        *,
        phase="runtime",
        delay_s=2.0,
        duration_s=6.0,
        interval_s=0.25,
    ):
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)

        if delay_s > 0:
            time.sleep(delay_s)

        samples = []
        start = time.monotonic()
        sample_index = 0

        while True:
            sample = self._sample_runtime_metadata_once()
            sample["sample_index"] = sample_index
            samples.append(sample)

            sample_index += 1

            if time.monotonic() - start >= duration_s:
                break

            time.sleep(interval_s)

        self._write_runtime_metadata_text(
            params=params,
            root_dir=root_dir,
            date=date,
            phase=phase,
            delay_s=delay_s,
            duration_s=duration_s,
            interval_s=interval_s,
            samples=samples,
        )

        self._write_runtime_metadata_csv(
            root_dir=root_dir,
            samples=samples,
        )

    def _sample_runtime_metadata_once(self):
        sample = {
            "timestamp_ns": time.time_ns(),
            "timestamp_s": time.time(),
        }

        sample.update(self._sample_cpu_policies())
        sample.update(self._sample_generic_devfreq())
        sample.update(self._sample_generic_thermals())
        sample.update(self._sample_generic_regulator_voltages())
        sample.update(self._sample_device_specific_runtime_metadata())

        for key in self.COMMON_RUNTIME_COLUMNS:
            sample.setdefault(key, 0)

        return sample

    def _sample_cpu_policies(self):
        data = {}

        for i in range(self.MAX_CPU_POLICIES):
            data[f"cpu_policy{i}_cur_khz"] = 0
            data[f"cpu_policy{i}_min_khz"] = 0
            data[f"cpu_policy{i}_max_khz"] = 0
            data[f"cpu_policy{i}_governor"] = ""
            data[f"cpu_policy{i}_affected_cpus"] = ""

        policy_paths = self._glob("/sys/devices/system/cpu/cpufreq/policy*")

        def policy_num(path):
            match = re.search(r"policy(\d+)$", path)
            return int(match.group(1)) if match else 9999

        for policy_path in sorted(policy_paths, key=policy_num):
            idx = policy_num(policy_path)

            cur = self._read_first_existing_text([
                f"{policy_path}/scaling_cur_freq",
                f"{policy_path}/scaling_curr_freq",
                f"{policy_path}/cpuinfo_cur_freq",
            ])

            data[f"cpu_policy{idx}_cur_khz"] = int(self._first_number(cur) or 0)
            data[f"cpu_policy{idx}_min_khz"] = self._read_cpufreq_khz(
                f"{policy_path}/scaling_min_freq"
            )
            data[f"cpu_policy{idx}_max_khz"] = self._read_cpufreq_khz(
                f"{policy_path}/scaling_max_freq"
            )
            data[f"cpu_policy{idx}_governor"] = self._read_text(
                f"{policy_path}/scaling_governor"
            )
            data[f"cpu_policy{idx}_affected_cpus"] = self._read_text(
                f"{policy_path}/affected_cpus"
            )

        return data

    def _sample_generic_devfreq(self):
        data = {
            "gpu_cur_khz": 0,
            "gpu_min_khz": 0,
            "gpu_max_khz": 0,
            "gpu_governor": "",
            "gpu_source": "",

            "memory_cur_khz": 0,
            "memory_min_khz": 0,
            "memory_max_khz": 0,
            "memory_governor": "",
            "memory_source": "",

            "npu_cur_khz": 0,
            "npu_source": "",
        }

        devfreq_paths = self._glob("/sys/class/devfreq/*")

        for path in devfreq_paths:
            name = path.split("/")[-1]
            safe = self._safe_key(name)
            lower = name.lower()

            cur = self._read_devfreq_khz(f"{path}/cur_freq")
            min_freq = self._read_devfreq_khz(f"{path}/min_freq")
            max_freq = self._read_devfreq_khz(f"{path}/max_freq")
            governor = self._read_text(f"{path}/governor")

            data[f"devfreq_{safe}_cur_khz"] = cur
            data[f"devfreq_{safe}_min_khz"] = min_freq
            data[f"devfreq_{safe}_max_khz"] = max_freq
            data[f"devfreq_{safe}_governor"] = governor

            is_gpu = any(token in lower for token in [
                "gpu", "mali", "v3d", "g3d", "3d", "rgx"
            ])

            is_memory = any(token in lower for token in [
                "dmc", "emc", "dram", "ddr", "memory", "mem"
            ])

            is_npu = any(token in lower for token in [
                "npu", "rknpu", "dla"
            ])

            if is_gpu and cur:
                data["gpu_cur_khz"] = cur
                data["gpu_min_khz"] = min_freq
                data["gpu_max_khz"] = max_freq
                data["gpu_governor"] = governor
                data["gpu_source"] = name

            if is_memory and cur:
                data["memory_cur_khz"] = cur
                data["memory_min_khz"] = min_freq
                data["memory_max_khz"] = max_freq
                data["memory_governor"] = governor
                data["memory_source"] = name

            if is_npu and cur:
                data["npu_cur_khz"] = cur
                data["npu_source"] = name

        return data

    def _sample_generic_thermals(self):
        data = {
            "cpu_temp_c": 0,
            "gpu_temp_c": 0,
            "soc_temp_c": 0,
            "board_temp_c": 0,
            "npu_temp_c": 0,
        }

        for zone in self._glob("/sys/class/thermal/thermal_zone*"):
            zone_type = self._read_text(f"{zone}/type") or zone.split("/")[-1]
            safe = self._safe_key(zone_type)
            temp_millic = self._read_number_file(f"{zone}/temp", 0)

            if not temp_millic:
                continue

            temp_c = round(float(temp_millic) / 1000.0, 3)
            data[f"temp_{safe}_c"] = temp_c

            lower = zone_type.lower()

            if "cpu" in lower:
                data["cpu_temp_c"] = temp_c
            elif "gpu" in lower:
                data["gpu_temp_c"] = temp_c
            elif "npu" in lower:
                data["npu_temp_c"] = temp_c
            elif "board" in lower:
                data["board_temp_c"] = temp_c
            elif any(token in lower for token in ["soc", "thermal", "tboard"]):
                data["soc_temp_c"] = temp_c

        return data

    def _sample_generic_regulator_voltages(self):
        data = {
            "voltage_cpu_mv": 0,
            "voltage_gpu_mv": 0,
            "voltage_core_mv": 0,
            "voltage_memory_mv": 0,
            "voltage_soc_mv": 0,
            "voltage_npu_mv": 0,
        }

        for reg in self._glob("/sys/class/regulator/regulator*"):
            name = self._read_text(f"{reg}/name", as_root=True)
            microvolts = self._read_number_file(f"{reg}/microvolts", 0)

            if not name or not microvolts:
                continue

            safe = self._safe_key(name)
            mv = round(float(microvolts) / 1000.0, 3)
            lower = name.lower()

            data[f"voltage_regulator_{safe}_mv"] = mv

            if "cpu" in lower:
                data["voltage_cpu_mv"] = max(data["voltage_cpu_mv"], mv)
            elif "gpu" in lower:
                data["voltage_gpu_mv"] = max(data["voltage_gpu_mv"], mv)
            elif "npu" in lower:
                data["voltage_npu_mv"] = max(data["voltage_npu_mv"], mv)
            elif any(token in lower for token in ["sdram", "ddr", "dram", "mem"]):
                data["voltage_memory_mv"] = max(data["voltage_memory_mv"], mv)
            elif any(token in lower for token in ["core", "logic"]):
                data["voltage_core_mv"] = max(data["voltage_core_mv"], mv)
            elif "soc" in lower:
                data["voltage_soc_mv"] = max(data["voltage_soc_mv"], mv)

        return data

    def _sample_device_specific_runtime_metadata(self):
        return {}

    def _write_runtime_metadata_csv(self, root_dir, samples):
        all_keys = set()

        for sample in samples:
            all_keys.update(sample.keys())

        common = [
            key
            for key in self.COMMON_RUNTIME_COLUMNS
            if key in all_keys
        ]

        extras = sorted(
            key
            for key in all_keys
            if key not in common
        )

        columns = common + extras

        with open(root_dir / "runtime_metadata_samples.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()

            for sample in samples:
                row = {}

                for key in columns:
                    value = sample.get(key, 0)

                    if value is None or value == "":
                        value = 0

                    row[key] = value

                writer.writerow(row)

    def _write_runtime_metadata_text(
        self,
        params,
        root_dir,
        date,
        phase,
        delay_s,
        duration_s,
        interval_s,
        samples,
    ):
        lines = []

        lines.append(f"Date: {date}")
        lines.append(f"Device: {self.device_name}")
        lines.append(f"Test Name: {params.get('name', '<unset>')}")
        lines.append(f"Executable: {params.get('executable', '<unset>')}")
        lines.append(
            f"Governor: {params.get('governor', params.get('cpu_governor', '<unset>'))}"
        )
        lines.append(
            f"Independent Variable: {params.get('independant_var', params.get('independent_var', '<unset>'))}"
        )
        lines.append("")
        lines.append("Runtime Sampling:")
        lines.append(f"Phase: {phase}")
        lines.append(f"Delay Before Sampling Seconds: {delay_s}")
        lines.append(f"Sampling Duration Seconds: {duration_s}")
        lines.append(f"Sampling Interval Seconds: {interval_s}")
        lines.append(f"Samples: {len(samples)}")
        lines.append("")
        lines.append("Runtime Metadata Summary:")
        lines.append("Format: key: last | avg | min | max")
        lines.append("")

        all_keys = sorted(
            key
            for sample in samples
            for key in sample.keys()
            if key not in {"timestamp_ns", "timestamp_s", "sample_index"}
        )

        for key in all_keys:
            values = [
                sample.get(key)
                for sample in samples
                if key in sample
            ]

            numeric_values = []

            for value in values:
                try:
                    if value == "" or value is None:
                        continue

                    numeric_values.append(float(value))
                except (TypeError, ValueError):
                    pass

            if numeric_values:
                last = numeric_values[-1]
                avg = statistics.fmean(numeric_values)
                min_v = min(numeric_values)
                max_v = max(numeric_values)

                lines.append(
                    f"{key}: "
                    f"last={last:g}, "
                    f"avg={avg:g}, "
                    f"min={min_v:g}, "
                    f"max={max_v:g}"
                )
            else:
                last = values[-1] if values else ""
                lines.append(f"{key}: {last}")

        with open(root_dir / "runtime_metadata.txt", "w") as f:
            for line in lines:
                f.write(f"{line}\n")