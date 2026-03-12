import sys
import time
import tempfile

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
        print("Ranges and steps: ", end='', flush=True)
        
        while True:
            
            for i in range(10):
                sample = self.meter.read_sample()

                if self.stability.update(sample.watts):
                    print('')
                    return
                
            rnge, step = self.stability.get_range_step()

            print(f'[{rnge}, {step}], ', end='', flush=True)



    def measure_idle(self, writer, seconds=60):

        for _ in range(seconds):

            s = self.meter.read_sample()

            writer.writerow([s.timestamp, s.watts, s.volts, s.amps])

    def apply_config(self, params):

        cfg_path = self.cfg["dut"]["config_txt"]

        cmd = f"cp {cfg_path} {cfg_path}.copy"
        code, text, err = self.client.run(cmd, as_root=True)

        if code != 0:
            raise Exception(f"Failed to make copy of config.txt: {err}")

        code, text, err = self.client.run(f"cat {cfg_path}", as_root=True)

        if code != 0:
            raise Exception(f"Failed to read config.txt: {err}")

        new_text = update_config(text, params)

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
        
        return new_text
        


    def run(self):

        self.meter.start_stream(self.cfg["host"]["sample_interval_sec"])

        comm_file = self.cfg["dut"].get("comm_file")

        if not comm_file:
            print("comm_file must be defined in config", file=sys.stderr)
            return

        for test in self.tests:

            tag = config_tag(test["config"], test["tag_fields"])

            for iteration in range(test["iterations"]):

                run_dir = self.logger.new_run_dir(test["name"], tag)

                attempt = 1

                while attempt <= self.cfg["retry"]["max_attempts"]:

                    print("Running", test["name"], tag, "attempt", attempt)

                    try:

                        self.client.connect()

                        config_txt = self.apply_config(test["config"])

                        with open(f"{run_dir}/config.txt", "w") as config_file:
                            config_file.write(config_txt)

                        print("Rebooting DUT")
                        self.client.reboot()

                        print("Waiting for reboot")

                        self.client.wait_for_ssh(
                            self.cfg["retry"]["ssh_wait_timeout_sec"]
                        )

                        # reconnect cleanly
                        self.client.close()
                        self.client.connect()

                        code, text, err = self.client.run(f"cpufreq-set -g {test["governor"]}", as_root=True)

                        time.sleep(self.cfg["retry"]["cooldown_sec"])

                        if code  != 0:
                            raise Exception(f"Unable to set governor: {err}")

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
                        self.wait_for_idle()

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