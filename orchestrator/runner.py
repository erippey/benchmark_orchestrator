import time

from .wattsup import WattsUpMeter
from .stability import StabilityDetector
from .ssh_client import DUTClient
from .logger import RunLogger
from .config_loader import expand_tests, config_tag
from .update_config import update_config


class BenchmarkRunner:

    def __init__(self, config):

        self.cfg = config

        self.meter = WattsUpMeter(
            port=config["host"]["wattsup_com_port"]
        )

        stab = config["stability"]

        self.stability = StabilityDetector(
            window_size=stab["window_size"],
            required_consecutive=stab["required_consecutive_windows"],
            max_range_watts=stab["max_range_watts"],
            max_step_delta_watts=stab["max_step_delta_watts"]
        )

        self.logger = RunLogger(config["host"]["log_root"])

        dut = config["dut"]

        self.client = DUTClient(
            dut["hostname"],
            dut["username"],
            root_access=dut["root_access"]
        )

        self.tests = expand_tests(config)

    def wait_for_idle(self):

        self.stability.clear()

        while True:

            sample = self.meter.read_sample()

            if self.stability.update(sample.watts):
                return

    def measure_idle(self, writer, seconds=60):

        for _ in range(seconds):

            s = self.meter.read_sample()

            writer.writerow([s.timestamp, s.watts, s.volts, s.amps])

    def apply_config(self, params):

        cfg_path = self.cfg["dut"]["config_txt"]

        code, text, _ = self.client.run(f"cat {cfg_path}", as_root=True)

        new_text = update_config(text, params)

        cmd = f"echo '{new_text}' > {cfg_path}"

        self.client.run(cmd, as_root=True)

    def run(self):

        self.meter.start_stream(self.cfg["host"]["sample_interval_sec"])

        for test in self.tests:

            tag = config_tag(test["config"], test["tag_fields"])

            for iteration in range(test["iterations"]):

                run_dir = self.logger.new_run_dir(test["name"], tag)

                attempt = 1

                while attempt <= self.cfg["retry"]["max_attempts"]:

                    print("Running", test["name"], tag, "attempt", attempt)

                    try:

                        self.client.connect()

                        self.apply_config(test["config"])

                        self.client.reboot()

                        print("Waiting for reboot")

                        self.client.wait_for_ssh(
                            self.cfg["retry"]["ssh_wait_timeout_sec"]
                        )

                        print("Waiting for idle stability")

                        self.wait_for_idle()

                        f, writer = self.logger.open_power_log(run_dir, "idle_power.csv")

                        self.measure_idle(writer)

                        f.close()

                        exe = test["executable"]
                        build = self.cfg["dut"]["build_dir"]

                        cmd = f"cd {build} && {exe}"

                        self.client.run(cmd)

                        f, writer = self.logger.open_power_log(run_dir, "run_power.csv")

                        start = time.time()

                        while True:

                            sample = self.meter.read_sample()

                            writer.writerow([
                                sample.timestamp,
                                sample.watts,
                                sample.volts,
                                sample.amps
                            ])

                            if time.time() - start > 30:
                                break

                        f.close()

                        print("Run completed")

                        break

                    except Exception as e:

                        print("Failure:", e)

                        attempt += 1

                        time.sleep(self.cfg["retry"]["cooldown_sec"])