import sys
import time
import tempfile
from datetime import datetime

from .wattsup import WattsUpMeter
from .stability import StabilityDetector
from .ssh_client import DUTClient
from .logger import RunLogger
from .config_loader import expand_tests, config_tag


class BenchmarkRunner:

    def __init__(self, config, device_manager):

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

        self.date = datetime.now().strftime("%Y-%m-%d")

        self.device_manager = device_manager
        self.device_manager.set_ssh_client(self.client)

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


    def run(self):

        self.meter.start_stream(self.cfg["host"]["sample_interval_sec"])

        comm_file = self.cfg["dut"].get("comm_file")

        if not comm_file:
            print("comm_file must be defined in config", file=sys.stderr)
            return

        for test in self.tests:

            tag = config_tag(test["config"], test["tag_fields"])

            for iteration in range(test["iterations"]):

                run_dir = self.logger.new_run_dir(test["name"], tag, self.date)

                attempt = 1

                while attempt <= self.cfg["retry"]["max_attempts"]:

                    print("Running", test["name"], tag, "attempt", attempt)
                    
                    self.device_manager.save_metadata(test, run_dir, self.date)

                    try:

                        self.client.connect()


                        config_txt = self.device_manager.apply_config(test)
                        with open(f"{run_dir}/config.txt", "w") as config_file:
                            config_file.write(config_txt)


                        time.sleep(self.cfg["retry"]["cooldown_sec"])

                        print("Waiting for idle stability")

                        self.wait_for_idle()

                        print("Idle power achieved, measuring idle power")

                        f, writer = self.logger.open_power_log(run_dir, "idle_power.csv")

                        self.measure_idle(writer)

                        f.close()

                        print(f"Idle power measurements finished, running {test['executable']}")

                        # remove trigger file
                        self.client.run(f"rm -f {comm_file}")

                        exe = test["executable"]
                        build = self.cfg["dut"]["build_dir"]

                        # build remote log directory
                        remote_log_dir = f"{build}/bench_logs/{run_dir.name}"

                        self.client.run(f"mkdir -p {remote_log_dir}")

                        args = f"--log_dir {remote_log_dir} --comm_file {comm_file}"

                        # start benchmark in background
                        cmd = (
                            f"(cd {build} && "
                            f"setsid nohup {exe} {args} "
                            f"> {remote_log_dir}/stdout_dump.txt 2>&1 "
                            f"< /dev/null & "
                            f"echo $! > {remote_log_dir}/run.pid)"
                        )

                        code, text, err = self.client.run(cmd, async_run=True)
                        
                        code, pid, _ = self.client.run(f"cat {remote_log_dir}/run.pid")
                        pid = pid.strip()
                        if not pid:
                            raise Exception("Failed to capture benchmark PID")

                        # wait for warmup stabilization
                        print("waiting for executable to reach stable power draw...")

                        self.wait_for_idle()

                        print("Stable power achieved, beginnining recording performance and power draw...")

                        # trigger benchmark
                        self.client.run(f"touch {comm_file}")

                        # start logging run power
                        f, writer = self.logger.open_power_log(run_dir, "run_power.csv")

                        test_start = time.time()
                        last_check = test_start

                        while True:

                            sample = self.meter.read_sample()

                            writer.writerow([
                                sample.timestamp,
                                sample.watts,
                                sample.volts,
                                sample.amps
                            ])

                            now = time.time()

                            if now - last_check > 5:
                                # check if benchmark finished
                                code, out, _ = self.client.run(
                                    f"test -f {comm_file} && cat {comm_file}"
                                )

                                if code == 0 and "done" in out:
                                    break

                                code, out, err = self.client.run(
                                    f"kill -0 {pid}"
                                )
                                alive = (code == 0)

                                if not alive:
                                    raise Exception("Benchmark has died unexpectedly")
                                
                                if now - test_start > test["max_runtime_sec"]:
                                    code, out, _ = self.client.run(
                                        f("kill {pid}")
                                    )
                                    raise Exception("Benchmark timeout")

                                last_check = time.time()

                        f.close()

                        print("Run completed")

                        self.client.fetch_directory(remote_log_dir, run_dir)

                        break

                    except Exception as e:

                        print("Failure:", e)

                        attempt += 1

                        time.sleep(self.cfg["retry"]["cooldown_sec"])