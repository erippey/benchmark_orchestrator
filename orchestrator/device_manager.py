import tempfile



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

        code, text, err = self.client.run(f"cpufreq-set -g {params["governor"]}", as_root=True)

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