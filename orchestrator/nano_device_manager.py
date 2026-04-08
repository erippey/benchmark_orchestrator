import time


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
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 306000000,
        "gpu_freq_min": 306000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 15,
        "NAME": "408MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 408000000,
        "gpu_freq_min": 408000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 16,
        "NAME": "510MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 510000000,
        "gpu_freq_min": 510000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 17,
        "NAME": "612MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 612000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 18,
        "NAME": "714MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 714000000,
        "gpu_freq_min": 714000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 19,
        "NAME": "816MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 816000000,
        "gpu_freq_min": 816000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 20,
        "NAME": "918MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 918000000,
        "gpu_freq_min": 918000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 21,
        "NAME": "1020MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
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
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 306000000,
        "gpu_freq_min": 306000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 5,
        "NAME": "408MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 408000000,
        "gpu_freq_min": 408000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 6,
        "NAME": "510MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 510000000,
        "gpu_freq_min": 510000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 7,
        "NAME": "612MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 612000000,
        "gpu_freq_min": 612000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 8,
        "NAME": "714MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 714000000,
        "gpu_freq_min": 714000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 9,
        "NAME": "816MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 816000000,
        "gpu_freq_min": 816000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 10,
        "NAME": "918MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
        "gpu_freq_max": 918000000,
        "gpu_freq_min": 918000000,
        "emc_max_freq": 2133000000
    },
    {
        "ID": 11,
        "NAME": "1020MHz_GPU",
        "cores_online": 6,
        "cpu_freq_max": 729600,
        "cpu_freq_min": 729600,
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
                if not run_config.get(key):
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

