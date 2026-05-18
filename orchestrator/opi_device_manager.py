import tempfile

import shlex
import tempfile
from pathlib import Path


AVAILABLE_CPU_FREQS_LITTLE = [
    408000, 600000, 816000, 1008000,
    1200000, 1416000, 1608000, 1800000,
]

AVAILABLE_CPU_FREQS_BIG = [
    408000, 600000, 816000, 1008000,
    1200000, 1416000, 1608000, 1800000,
    2016000, 2208000, 2256000,
]

AVAILABLE_GPU_FREQS = [
    300000000, 400000000, 500000000, 600000000,
    700000000, 800000000, 900000000, 1000000000,
]


RK3588_FREQ_BIN = "/usr/local/bin/rk3588_freq"


# These are user-facing config keys your benchmark configs may contain.
USER_KEYS = [
    # Global CPU controls
    "cpu_governor",
    "cpu_freq",
    "cpu_freq_min",
    "cpu_freq_max",

    # Backward-compatible generic governor.
    # I would treat this as CPU governor unless you explicitly add gpu_governor.
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

    # DMC controls, optional for later
    "dmc_governor",
    "dmc_freq",
    "dmc_freq_min",
    "dmc_freq_max",
]


# These are not user config keys. These are what you read back for metadata.
MANAGED_KEY_PATHS = {
    # CPU policy0, A55 cluster
    "policy0_scaling_governor": "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor",
    "policy0_scaling_cur_freq": "/sys/devices/system/cpu/cpufreq/policy0/scaling_cur_freq",
    "policy0_scaling_min_freq": "/sys/devices/system/cpu/cpufreq/policy0/scaling_min_freq",
    "policy0_scaling_max_freq": "/sys/devices/system/cpu/cpufreq/policy0/scaling_max_freq",
    "policy0_cpuinfo_min_freq": "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_min_freq",
    "policy0_cpuinfo_max_freq": "/sys/devices/system/cpu/cpufreq/policy0/cpuinfo_max_freq",

    # CPU policy4, A76 pair
    "policy4_scaling_governor": "/sys/devices/system/cpu/cpufreq/policy4/scaling_governor",
    "policy4_scaling_cur_freq": "/sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq",
    "policy4_scaling_min_freq": "/sys/devices/system/cpu/cpufreq/policy4/scaling_min_freq",
    "policy4_scaling_max_freq": "/sys/devices/system/cpu/cpufreq/policy4/scaling_max_freq",
    "policy4_cpuinfo_min_freq": "/sys/devices/system/cpu/cpufreq/policy4/cpuinfo_min_freq",
    "policy4_cpuinfo_max_freq": "/sys/devices/system/cpu/cpufreq/policy4/cpuinfo_max_freq",

    # CPU policy6, A76 pair
    "policy6_scaling_governor": "/sys/devices/system/cpu/cpufreq/policy6/scaling_governor",
    "policy6_scaling_cur_freq": "/sys/devices/system/cpu/cpufreq/policy6/scaling_cur_freq",
    "policy6_scaling_min_freq": "/sys/devices/system/cpu/cpufreq/policy6/scaling_min_freq",
    "policy6_scaling_max_freq": "/sys/devices/system/cpu/cpufreq/policy6/scaling_max_freq",
    "policy6_cpuinfo_min_freq": "/sys/devices/system/cpu/cpufreq/policy6/cpuinfo_min_freq",
    "policy6_cpuinfo_max_freq": "/sys/devices/system/cpu/cpufreq/policy6/cpuinfo_max_freq",

    # GPU devfreq
    "gpu_governor": "/sys/class/devfreq/fb000000.gpu/governor",
    "gpu_cur_freq": "/sys/class/devfreq/fb000000.gpu/cur_freq",
    "gpu_min_freq": "/sys/class/devfreq/fb000000.gpu/min_freq",
    "gpu_max_freq": "/sys/class/devfreq/fb000000.gpu/max_freq",

    # DMC devfreq
    "dmc_governor": "/sys/class/devfreq/dmc/governor",
    "dmc_cur_freq": "/sys/class/devfreq/dmc/cur_freq",
    "dmc_min_freq": "/sys/class/devfreq/dmc/min_freq",
    "dmc_max_freq": "/sys/class/devfreq/dmc/max_freq",
}

MANAGED_KEYS = list(MANAGED_KEY_PATHS.keys())


class OPI5DeviceManager:

    def __init__(self, config, ssh_client=None):
        self.managed_keys = MANAGED_KEYS
        self.client = ssh_client
        self.cfg = config
        self.device_name = "Orange Pi 5 Ultra"

    def set_ssh_client(self, ssh_client):
        self.client = ssh_client

    def _quote_cmd(self, argv):
        return " ".join(shlex.quote(str(x)) for x in argv)

    def _run_checked(self, argv, as_root):
        """
        Run a command on the DUT through the SSH client.

        argv is a list, not a string. This function quotes it into a shell-safe
        command string because the SSH client likely expects a command string.
        """
        cmd = self._quote_cmd(argv)
        code, out, err = self.client.run(cmd, as_root=as_root)

        if code != 0:
            raise RuntimeError(
                f"Command failed with exit code {code}\n"
                f"Command: {cmd}\n"
                f"stdout:\n{out}\n"
                f"stderr:\n{err}"
            )

        return out

    def _normalize_user_params(self, params):
        """
        Pull only known user config keys from params.

        Also handles backward compatibility:
          governor -> cpu_governor
        """
        normalized = {}

        for key in USER_KEYS:
            if key not in params['config']:
                continue

            value = params['config'][key]

            if value is None:
                continue

            if isinstance(value, str) and value.strip() == "":
                continue

            normalized[key] = value

        # Backward compatibility with your existing benchmark configs.
        # I would not automatically use this for GPU because CPU and GPU do
        # not necessarily support the same governor names.
        if "governor" in params and "cpu_governor" not in normalized:
            value = params["governor"]
            if value is not None and not (isinstance(value, str) and value.strip() == ""):
                normalized["cpu_governor"] = value

        return normalized

    def _has_any(self, cfg, keys):
        return any(k in cfg for k in keys)

    def _add_freq_args(self, argv, cfg, key_prefix, cli_prefix):
        """
        Convert config keys like:

            gpu_freq
            gpu_freq_min
            gpu_freq_max

        into:

            --gpu-freq
            --gpu-min
            --gpu-max

        For CPU, key_prefix='cpu' and cli_prefix='cpu'.
        For policy controls, key_prefix='policy0' and cli_prefix='cpu'.
        """
        fixed_key = f"{key_prefix}_freq"
        min_key = f"{key_prefix}_freq_min"
        max_key = f"{key_prefix}_freq_max"

        has_fixed = fixed_key in cfg
        has_min = min_key in cfg
        has_max = max_key in cfg

        if has_fixed and (has_min or has_max):
            raise ValueError(
                f"Use either {fixed_key} or {min_key}/{max_key}, not both."
            )

        if has_fixed:
            argv.extend([f"--{cli_prefix}-freq", cfg[fixed_key]])

        if has_min:
            argv.extend([f"--{cli_prefix}-min", cfg[min_key]])

        if has_max:
            argv.extend([f"--{cli_prefix}-max", cfg[max_key]])

    def _append_metadata_reads(self, metadata):
        for var_name in self.managed_keys:
            path = MANAGED_KEY_PATHS.get(var_name)

            if path is None:
                continue

            cmd = f"cat {shlex.quote(path)}"
            code, out, err = self.client.run(cmd)

            if code == 0:
                metadata.append(f"{var_name}: {out.strip()}")
            else:
                metadata.append(f"{var_name}: <unavailable>")

    def save_metadata(self, params, root_dir, date):
        metadata = []

        metadata.append(f"Date: {date}")
        metadata.append(f"Device: {self.device_name}")
        metadata.append(f"Test Name: {params['name']}")
        metadata.append(f"Executable: {params['executable']}")
        metadata.append(f"Governor: {params.get('governor', params.get('cpu_governor', '<unset>'))}")
        metadata.append("")

        metadata.append(f"Independent Variable: {params['independant_var']}")
        metadata.append("")

        metadata.append("Applied User Config:")
        user_cfg = self._normalize_user_params(params)
        for key in sorted(user_cfg):
            metadata.append(f"{key}: {user_cfg[key]}")

        metadata.append("")
        metadata.append("Observed Device State:")
        self._append_metadata_reads(metadata)

        with open(root_dir / "config_metadata.txt", "w") as meta_file:
            for line in metadata:
                meta_file.write(f"{line}\n")

    def apply_config(self, params):
        """
        Apply Orange Pi 5 frequency/governor config using rk3588_freq.

        Supported params include:

            governor              -> treated as cpu_governor
            cpu_governor
            cpu_freq
            cpu_freq_min
            cpu_freq_max

            policy0_freq
            policy0_freq_min
            policy0_freq_max
            policy4_freq
            policy4_freq_min
            policy4_freq_max
            policy6_freq
            policy6_freq_min
            policy6_freq_max

            gpu_governor
            gpu_freq
            gpu_freq_min
            gpu_freq_max

            dmc_governor
            dmc_freq
            dmc_freq_min
            dmc_freq_max

        Notes:
          - cpu_freq sets all CPU policies.
          - policyN_freq sets a specific cpufreq policy.
          - fixed *_freq maps to min=max inside rk3588_freq.
          - *_freq_min and *_freq_max allow DVFS within a range.
        """
        cfg = self._normalize_user_params(params)

        commands = []
        applied = []

        # ------------------------------------------------------------
        # Validate ambiguous CPU configs.
        # ------------------------------------------------------------
        global_cpu_keys = [
            "cpu_freq",
            "cpu_freq_min",
            "cpu_freq_max",
        ]

        policy_cpu_keys = []
        for policy in ("policy0", "policy4", "policy6"):
            policy_cpu_keys.extend([
                f"{policy}_freq",
                f"{policy}_freq_min",
                f"{policy}_freq_max",
            ])

        has_global_cpu_freq = self._has_any(cfg, global_cpu_keys)
        has_policy_cpu_freq = self._has_any(cfg, policy_cpu_keys)

        if has_global_cpu_freq and has_policy_cpu_freq:
            raise ValueError(
                "Do not mix global CPU frequency keys with per-policy CPU "
                "frequency keys in the same config. Use either cpu_freq* or "
                "policy0/policy4/policy6_freq*."
            )

        # ------------------------------------------------------------
        # CPU governor.
        #
        # Apply this globally once. Frequency commands can then focus only
        # on min/max/fixed frequency.
        # ------------------------------------------------------------
        if "cpu_governor" in cfg:
            argv = [
                RK3588_FREQ_BIN,
                "set",
                "--cpu-target",
                "all",
                "--cpu-governor",
                cfg["cpu_governor"],
            ]

            commands.append(argv)
            applied.append(("cpu_governor", cfg["cpu_governor"]))

        # ------------------------------------------------------------
        # Global CPU frequency controls.
        # ------------------------------------------------------------
        if has_global_cpu_freq:
            argv = [
                RK3588_FREQ_BIN,
                "set",
                "--cpu-target",
                "all",
            ]

            self._add_freq_args(argv, cfg, "cpu", "cpu")

            commands.append(argv)

            for key in global_cpu_keys:
                if key in cfg:
                    applied.append((key, cfg[key]))

        # ------------------------------------------------------------
        # Per-policy CPU frequency controls.
        # ------------------------------------------------------------
        for policy in ("policy0", "policy4", "policy6"):
            keys = [
                f"{policy}_freq",
                f"{policy}_freq_min",
                f"{policy}_freq_max",
            ]

            if not self._has_any(cfg, keys):
                continue

            argv = [
                RK3588_FREQ_BIN,
                "set",
                "--cpu-policies",
                policy,
            ]

            self._add_freq_args(argv, cfg, policy, "cpu")

            commands.append(argv)

            for key in keys:
                if key in cfg:
                    applied.append((key, cfg[key]))

        # ------------------------------------------------------------
        # GPU controls.
        # ------------------------------------------------------------
        gpu_keys = [
            "gpu_freq",
            "gpu_freq_min",
            "gpu_freq_max",
        ]

        has_gpu_freq = self._has_any(cfg, gpu_keys)

        if "gpu_governor" in cfg or has_gpu_freq:
            argv = [
                RK3588_FREQ_BIN,
                "set",
            ]

            if "gpu_governor" in cfg:
                argv.extend(["--gpu-governor", cfg["gpu_governor"]])
                applied.append(("gpu_governor", cfg["gpu_governor"]))

            if has_gpu_freq:
                self._add_freq_args(argv, cfg, "gpu", "gpu")

                for key in gpu_keys:
                    if key in cfg:
                        applied.append((key, cfg[key]))

            commands.append(argv)

        # ------------------------------------------------------------
        # DMC controls.
        #
        # Optional for later. This is here so the manager shape is ready,
        # but you do not have to use it yet.
        # ------------------------------------------------------------
        dmc_keys = [
            "dmc_freq",
            "dmc_freq_min",
            "dmc_freq_max",
        ]

        has_dmc_freq = self._has_any(cfg, dmc_keys)

        if "dmc_governor" in cfg or has_dmc_freq:
            argv = [
                RK3588_FREQ_BIN,
                "set",
            ]

            if "dmc_governor" in cfg:
                argv.extend(["--dmc-governor", cfg["dmc_governor"]])
                applied.append(("dmc_governor", cfg["dmc_governor"]))

            if has_dmc_freq:
                self._add_freq_args(argv, cfg, "dmc", "dmc")

                for key in dmc_keys:
                    if key in cfg:
                        applied.append((key, cfg[key]))

            commands.append(argv)

        # ------------------------------------------------------------
        # Run commands.
        # ------------------------------------------------------------
        command_text = []

        for argv in commands:
            cmd = self._quote_cmd(argv)
            command_text.append(cmd)
            self._run_checked(argv, True)

        # ------------------------------------------------------------
        # Return text for benchmark metadata/logging.
        # ------------------------------------------------------------
        new_text = []

        new_text.append("Requested User Config:")
        if applied:
            for key, value in applied:
                new_text.append(f"{key}: {value}")
        else:
            new_text.append("<none>")

        new_text.append("")
        new_text.append("Commands:")
        if command_text:
            for cmd in command_text:
                new_text.append(cmd)
        else:
            new_text.append("<none>")

        return "\n".join(new_text) + "\n"