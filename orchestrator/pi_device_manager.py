import re
import tempfile

from orchestrator.RuntimeMetadataMixin import RuntimeMetadataMixin



class CM5DeviceManager:

    def __init__(self, config, ssh_client = None):
        self.managed_keys = [
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

        self.client = ssh_client

        self.cfg = config

        self.device_name = "Raspberry Pi Compute Module 5"

        self.runtime_manager = RaspberryPiRuntimeMetadataMixin(self)

    def set_ssh_client(self, ssh_client):
        self.client = ssh_client

    def save_metadata(self, params, root_dir, date):
        metadata = []

        metadata.append(f"Date: {date}")
        metadata.append(f"Device: {self.device_name}")
        metadata.append(f"Test Name: {params['name']}")
        metadata.append(f"Executable: {params['executable']}")
        metadata.append(f"Governor: {params['governor']}")
        metadata.append("")

        metadata.append(f"Independant Vairable: {params['independant_var']}")
        
        for var_name in self.managed_keys:
            value = params['config'].get(var_name)

            if value:
                metadata.append(f"{var_name}: {value}")
            else:
                metadata.append(f"{var_name}: 0")

        with open(root_dir / "config_metadata.txt", "w") as meta_file:
            for line in metadata:
                meta_file.write(f"{line}\n")

    def save_runtime_metadata(self, params, root_dir, date):
        self.runtime_manager.save_runtime_metadata(params, root_dir, date)


    def reboot_and_reconnect(self):
        self.client.reboot()

        self.client.close()
        print("Waiting for reboot")

        self.client.wait_for_ssh(
            self.cfg["retry"]["ssh_wait_timeout_sec"]
        )


    def apply_config(self, params):

        cfg_path = self.cfg["dut"]["config_txt"]

        cmd = f"cp {cfg_path} {cfg_path}.copy"

        code, text, err = self.client.run(cmd, as_root=True)

        if code != 0:
            raise Exception(f"Failed to make copy of config.txt: {err}")
        
        code, text, err = self.client.run(f"cat {cfg_path}", as_root=True)

        if code != 0:
            raise Exception(f"Failed to read config.txt: {err}")

        new_text = self.update_config(text, params["config"])

        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(new_text.encode())
            local_path = tmp.name

        remote_tmp = "/tmp/config.txt.new"

        sftp = self.client.user.open_sftp()
        sftp.put(local_path, remote_tmp)
        sftp.close()

        cmd = f"mv {remote_tmp} {cfg_path}"

        code, text, err = self.client.run(cmd, as_root=True)

        if code != 0:
            raise Exception(f"Failed to install config.txt: {err}")
        
        print("Rebooting DUT")
        self.reboot_and_reconnect()

        # new OS cant find cpufreq-set / cpufrequitls
        # code, text, err = self.client.run(f"cpufreq-set -g {params["governor"]}", as_root=True)
        if (params.get('governor') is not None):
            code , text, err = self.client.run(f"echo {params["governor"]} > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor", as_root=True)
        else: 
            raise Exception("No governor set")

        if code != 0:
            raise Exception(f"Unable to set governor: {err}")
        
        return new_text

    def update_config(self, text, params):
        

        lines = text.splitlines()
        output = []
        seen_managed = set()

        for line in lines:
            stripped = line.strip()
            handled = False

            for key in self.managed_keys:
                active_prefix = f"{key}="
                commented_prefix = f"#{key}="

                if stripped.startswith(active_prefix) or stripped.startswith(commented_prefix):
                    if key in seen_managed:
                        handled = True
                        break

                    seen_managed.add(key)

                    if key in params:
                        output.append(f"{key}={params[key]}")
                    else:
                        output.append(f"#{key}=")
                    handled = True
                    break

            if not handled:
                output.append(line)

        for key in self.managed_keys:
            if key in params and key not in seen_managed:
                output.append(f"{key}={params[key]}")
            elif key not in params and key not in seen_managed:
                # optional: append commented default marker for visibility
                # output.append(f"#{key}=")
                pass

        return "\n".join(output) + "\n"
    


class RaspberryPiRuntimeMetadataMixin(RuntimeMetadataMixin):

    def _vcgencmd(self, args):
        return self._run_shell(
            f"command -v vcgencmd >/dev/null 2>&1 && "
            f"vcgencmd {args} 2>/dev/null || true",
            timeout=5,
        )

    @staticmethod
    def _parse_vcgencmd_clock_khz(text):
        match = re.search(r"frequency\(\d+\)=(\d+)", text or "")
        if not match:
            return 0

        return int(match.group(1)) // 1000

    @staticmethod
    def _parse_vcgencmd_voltage_mv(text):
        match = re.search(r"volt=([0-9.]+)V", text or "")
        if not match:
            return 0

        return round(float(match.group(1)) * 1000.0, 3)

    @staticmethod
    def _parse_vcgencmd_temp_c(text):
        match = re.search(r"temp=([0-9.]+)", text or "")
        if not match:
            return 0

        return float(match.group(1))

    def _sample_device_specific_runtime_metadata(self):
        data = {}

        clock_aliases = {
            "arm": "pi_clock_arm_khz",
            "core": "core_cur_khz",
            "v3d": "gpu_cur_khz",
            "emmc": "pi_clock_emmc_khz",
            "isp": "pi_clock_isp_khz",
            "h264": "pi_clock_h264_khz",
            "hevc": "pi_clock_hevc_khz",
        }

        for clock_name, key in clock_aliases.items():
            khz = self._parse_vcgencmd_clock_khz(
                self._vcgencmd(f"measure_clock {clock_name}")
            )

            if khz:
                data[key] = khz

                if clock_name == "v3d":
                    data["gpu_source"] = "vcgencmd measure_clock v3d"
                elif clock_name == "core":
                    data["core_source"] = "vcgencmd measure_clock core"

        voltage_aliases = {
            "core": "voltage_core_mv",
            "sdram_c": "pi_voltage_sdram_c_mv",
            "sdram_i": "pi_voltage_sdram_i_mv",
            "sdram_p": "pi_voltage_sdram_p_mv",
        }

        memory_voltages = []

        for rail, key in voltage_aliases.items():
            mv = self._parse_vcgencmd_voltage_mv(
                self._vcgencmd(f"measure_volts {rail}")
            )

            if mv:
                data[key] = mv

                if rail == "core":
                    data["voltage_core_mv"] = mv
                elif rail.startswith("sdram"):
                    memory_voltages.append(mv)

        if memory_voltages:
            data["voltage_memory_mv"] = max(memory_voltages)

        temp_c = self._parse_vcgencmd_temp_c(
            self._vcgencmd("measure_temp")
        )

        if temp_c:
            data["soc_temp_c"] = temp_c
            data["cpu_temp_c"] = temp_c

        throttled = self._vcgencmd("get_throttled")
        match = re.search(r"throttled=0x([0-9a-fA-F]+)", throttled or "")

        if match:
            data["throttle_raw"] = int(match.group(1), 16)
            data["pi_throttled_hex"] = f"0x{match.group(1)}"

        return data