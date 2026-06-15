import time
import re

from orchestrator.RuntimeMetadataMixin import RuntimeMetadataMixin


known_configurations_1 = [
    {
        "ID": 0,
        "NAME": "15W",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 1,
        "NAME": "25W",
        "cores_online": 6,
        "cpu_freq_max": 1344000,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 918000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 3199000000
    },
    {
        "ID": 2,
        "NAME": "MAXN_SUPER",
        "cores_online": 6,
        "cpu_freq_max": -1,
        "cpu_freq_min": 729600,
        "gpu_freq_max": -1,
        "gpu_freq_min": 0,
        "emc_max_freq": -1
    },
    {
        "ID": 3,
        "NAME": "7W",
        "cores_online": 4,
        "cpu_freq_max": 960000,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 408000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 4,
        "NAME": "729MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 5,
        "NAME": "806MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 806400,
        "cpu_freq_min": 806400,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 6,
        "NAME": "883MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 883200,
        "cpu_freq_min": 883200,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 7,
        "NAME": "960MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 960000,
        "cpu_freq_min": 960000,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 8,
        "NAME": "1036MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 1036800,
        "cpu_freq_min": 1036800,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 9,
        "NAME": "1190MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 1190400,
        "cpu_freq_min": 1190400,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 10,
        "NAME": "1267MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 1267200,
        "cpu_freq_min": 1267200,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 11,
        "NAME": "1344MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 1344000,
        "cpu_freq_min": 1344000,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 12,
        "NAME": "1420MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 1420800,
        "cpu_freq_min": 1420800,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 13,
        "NAME": "1497MHz_CPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 14,
        "NAME": "306MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 306000000,
        "gpu_freq_min": 306000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 15,
        "NAME": "408MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 408000000,
        "gpu_freq_min": 408000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 16,
        "NAME": "510MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 510000000,
        "gpu_freq_min": 510000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 17,
        "NAME": "612MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 612000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 18,
        "NAME": "714MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 714000000,
        "gpu_freq_min": 714000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 19,
        "NAME": "816MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 816000000,
        "gpu_freq_min": 816000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 20,
        "NAME": "918MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 918000000,
        "gpu_freq_min": 918000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 21,
        "NAME": "1020MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 1020000000,
        "gpu_freq_min": 1020000000,
        "emc_max_freq": 2133000000
    }
]

known_configurations_2 = [
    {
        "ID": 0,
        "NAME": "15W",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 1,
        "NAME": "25W",
        "cores_online": 6,
        "cpu_freq_max": 1344000,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 918000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 3199000000
    },
    {
        "ID": 2,
        "NAME": "MAXN_SUPER",
        "cores_online": 6,
        "cpu_freq_max": -1,
        "cpu_freq_min": 729600,
        "gpu_freq_max": -1,
        "gpu_freq_min": 0,
        "emc_max_freq": -1
    },
    {
        "ID": 3,
        "NAME": "7W",
        "cores_online": 4,
        "cpu_freq_max": 960000,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 408000000,
        "gpu_freq_min": 0,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 4,
        "NAME": "306MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 306000000,
        "gpu_freq_min": 306000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 5,
        "NAME": "408MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 408000000,
        "gpu_freq_min": 408000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 6,
        "NAME": "510MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 510000000,
        "gpu_freq_min": 510000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 7,
        "NAME": "612MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 612000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 8,
        "NAME": "714MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 714000000,
        "gpu_freq_min": 714000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 9,
        "NAME": "816MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 816000000,
        "gpu_freq_min": 816000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 10,
        "NAME": "918MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 918000000,
        "gpu_freq_min": 918000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 11,
        "NAME": "1020MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 1497600,
        "cpu_freq_min": 1497600,
        "gpu_freq_max": 1020000000,
        "gpu_freq_min": 1020000000,
        "emc_max_freq": 2133000000
    }
]




class NanoDeviceManager:

    def __init__(self, config, ssh_client = None):
        self.managed_keys = [
            "ID",
            "cores_online",
            "cpu_freq_max",
            "cpu_freq_min",
            "gpu_freq_max",
            "gpu_freq_min",
            "emc_max_freq"
        ]

        self.client = ssh_client

        self.cfg = config

        self.device_name = "Jetson Orin Nano Super"

        self.runtime_manager = JetsonRuntimeMetadataMixin(self)

    def set_ssh_client(self, ssh_client):
        self.client = ssh_client

    def save_metadata(self, params, root_dir, date):
        metadata = []
        known_cfg = self.resolve_known_configuration(params["config"])

        metadata.append(f"Date: {date}")
        metadata.append(f"Device: {self.device_name}")
        metadata.append(f"Test Name: {params['name']}")
        metadata.append(f"Executable: {params['executable']}")
        metadata.append(f"Governor: {params['governor']}")
        metadata.append("")

        metadata.append(f"Independant Vairable: {params['independant_var']}")
        metadata.append(self.format_config_metadata_block(known_cfg))

        with open(root_dir / "config_metadata.txt", "w") as meta_file:
            for line in metadata:
                meta_file.write(f"{line}\n")

    def save_runtime_metadata(self, params, root_dir, date):
        self.runtime_manager.save_runtime_metadata(params, root_dir, date)



    def reboot_and_reconnect(self):
        self.client.reboot()

        try:
            self.client.close()
        except AttributeError:
            pass
        print("Waiting for reboot")

        self.client.wait_for_ssh(
            self.cfg["retry"]["ssh_wait_timeout_sec"]
        )

    def _still_connected(self):
        try:
            code, _, _ = self.client.run("true", as_root=False)
            return code == 0
        except Exception:
            return False


    def apply_config(self, params):

        known_cfg = self.resolve_known_configuration(params["config"])

        code, _, err = self.client.run(f"nvpmodel -m {known_cfg['ID']} --force", as_root=True)
        
        time.sleep(2)

        if not self._still_connected():
            self.reboot_and_reconnect()

        code, out, err = self.client.run("nvpmodel -q", as_root=True)

        if known_cfg["NAME"] in out and str(known_cfg["ID"]) in out:
            return self.format_config_metadata_block(known_cfg)
        else:
            raise Exception(f"Failed to set nvpmodel: expected {known_cfg["name"]}, found\n{out}")
    

    def resolve_known_configuration(self, run_config):
        """
        Find exactly one known configuration whose managed fields match run_config.
        Extra keys in run_config are ignored.
        """
        matches = []

        for known in known_configurations_2:
            same = True
            for key in self.managed_keys:
                if run_config.get(key) is None:
                    continue
                if run_config.get(key) != known.get(key):
                    same = False
                    break
            if same:
                matches.append(known)

        if len(matches) == 0:
            raise ValueError(
                f"No known configuration matched run config: {run_config}"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Multiple known configurations matched run config: {run_config}. "
                f"Matches: {[m['NAME'] for m in matches]}"
            )

        return matches[0]
    
    def format_config_metadata_block(self, known_cfg):
        lines = []
        lines.append(f"Config_Name: {known_cfg['NAME']}")
        lines.append(f"Config_ID: {known_cfg['ID']}")

        for key in self.managed_keys:
            value = known_cfg.get(key, 0)
            lines.append(f"{key}: {value}")

        return "\n".join(lines)
    

class JetsonRuntimeMetadataMixin(RuntimeMetadataMixin):
    def _sample_device_specific_runtime_metadata(self):
        data = {}

        line = self._run_shell(
            "command -v tegrastats >/dev/null 2>&1 && "
            "timeout 2s tegrastats --interval 100 --count 1 2>/dev/null | head -n 1 || true"
        )

        if line:
            data["jetson_tegrastats_line"] = line
            data.update(self._parse_tegrastats_line(line))

        data.update(self._sample_jetson_ina3221())

        return data

    def _parse_tegrastats_line(self, line):
        data = {}

        # Example forms:
        # GR3D_FREQ 99%@918
        # EMC_FREQ 12%@2133
        match = re.search(r"GR3D_FREQ\s+(\d+)%@(\d+)", line)
        if match:
            data["gpu_util_pct"] = int(match.group(1))
            data["gpu_cur_khz"] = int(match.group(2)) * 1000
            data["gpu_source"] = "tegrastats GR3D_FREQ"

        match = re.search(r"EMC_FREQ\s+(\d+)%@(\d+)", line)
        if match:
            data["memory_util_pct"] = int(match.group(1))
            data["memory_cur_khz"] = int(match.group(2)) * 1000
            data["memory_source"] = "tegrastats EMC_FREQ"

        # CPU [0%@729,3%@729,...]
        match = re.search(r"CPU\s+\[([^\]]+)\]", line)
        if match:
            cpu_entries = match.group(1).split(",")

            for idx, entry in enumerate(cpu_entries):
                freq_match = re.search(r"@(\d+)", entry)
                util_match = re.search(r"(\d+)%", entry)

                if freq_match:
                    data[f"jetson_cpu_core{idx}_cur_khz"] = int(freq_match.group(1)) * 1000

                if util_match:
                    data[f"jetson_cpu_core{idx}_util_pct"] = int(util_match.group(1))

        # Temps: CPU@42C GPU@39C SOC0@40C etc.
        for name, temp in re.findall(r"([A-Za-z0-9_]+)@([0-9.]+)C", line):
            safe = self._safe_key(name)
            temp_c = float(temp)
            data[f"temp_{safe}_c"] = temp_c

            lower = name.lower()

            if "cpu" in lower:
                data["cpu_temp_c"] = temp_c
            elif "gpu" in lower:
                data["gpu_temp_c"] = temp_c
            elif "soc" in lower:
                data["soc_temp_c"] = temp_c
            elif "board" in lower or "tboard" in lower:
                data["board_temp_c"] = temp_c

        # Power rails, when present:
        # VDD_CPU_GPU_CV 1234mW/1567mW
        # VDD_SOC 567mW/600mW
        total_power = 0

        for rail, inst_mw, avg_mw in re.findall(
            r"\b([A-Z0-9_]+)\s+(\d+)mW/(\d+)mW",
            line,
        ):
            safe = self._safe_key(rail)
            inst_mw = int(inst_mw)
            avg_mw = int(avg_mw)

            data[f"power_{safe}_instant_mw"] = inst_mw
            data[f"power_{safe}_avg_mw"] = avg_mw

            total_power += inst_mw

            lower = rail.lower()

            if "cpu" in lower:
                data["power_cpu_mw"] = max(data.get("power_cpu_mw", 0), inst_mw)

            if "gpu" in lower:
                data["power_gpu_mw"] = max(data.get("power_gpu_mw", 0), inst_mw)

            if "soc" in lower:
                data["power_soc_mw"] = max(data.get("power_soc_mw", 0), inst_mw)

            if "ddr" in lower or "mem" in lower:
                data["power_memory_mw"] = max(data.get("power_memory_mw", 0), inst_mw)

        if total_power:
            data["power_total_mw"] = total_power

        return data

    def _sample_jetson_ina3221(self):
        """
        Optional Jetson power/voltage rail reader.

        This is intentionally broad because Jetson Nano, Xavier, Orin, and carrier
        boards expose these sensors differently.
        """

        data = {}

        out = self._run_shell(r'''
for d in /sys/bus/i2c/drivers/ina3221x/*/iio_device /sys/bus/i2c/devices/*/iio:device*; do
    [ -d "$d" ] || continue

    dev="$(basename "$d")"

    for f in "$d"/in_voltage*_input "$d"/in_current*_input "$d"/in_power*_input; do
        [ -e "$f" ] || continue

        base="$(basename "$f")"
        channel="$(echo "$base" | sed -E 's/in_(voltage|current|power)([0-9]+).*/\2/')"

        label=""
        [ -e "$d/${base%_input}_label" ] && label="$(cat "$d/${base%_input}_label")"
        [ -z "$label" ] && [ -e "$d/in_voltage${channel}_label" ] && label="$(cat "$d/in_voltage${channel}_label")"

        value="$(cat "$f" 2>/dev/null)"
        printf "%s|%s|%s|%s\n" "$dev" "$label" "$base" "$value"
    done
done
''')

        for line in out.splitlines():
            parts = line.split("|")
            if len(parts) != 4:
                continue

            dev, label, field, value = parts
            label = label or dev
            safe_label = self._safe_key(label)
            safe_field = self._safe_key(field)

            number = self._first_number(value)
            if number is None:
                continue

            key = f"jetson_ina_{safe_label}_{safe_field}"
            data[key] = number

            lower = label.lower()

            if "cpu" in lower and "voltage" in field:
                data["voltage_cpu_mv"] = max(data.get("voltage_cpu_mv", 0), number)
            elif "gpu" in lower and "voltage" in field:
                data["voltage_gpu_mv"] = max(data.get("voltage_gpu_mv", 0), number)
            elif "soc" in lower and "voltage" in field:
                data["voltage_soc_mv"] = max(data.get("voltage_soc_mv", 0), number)
            elif ("ddr" in lower or "mem" in lower) and "voltage" in field:
                data["voltage_memory_mv"] = max(data.get("voltage_memory_mv", 0), number)

        return data

