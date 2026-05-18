import sys
import time
import tempfile
from datetime import datetime

from .ssh_client import DUTClient
from .logger import RunLogger
from .config_loader import expand_tests, config_tag


class BenchmarkRunnerUnmeasured:

    def __init__(self, config, device_manager):

        self.cfg = config


        self.logger = RunLogger(config["host"]["log_root"])

        dut = config["dut"]
        ip = config.get("backup_ip") # backup ip may or may not exist

        self.client = DUTClient(
            dut["hostname"],
            dut["username"],
            root_access=dut["root_access"],
            backup_ip=ip
        )

        self.tests = expand_tests(config)

        self.date = datetime.now().strftime("%Y-%m-%d")

        self.device_manager = device_manager
        self.device_manager.set_ssh_client(self.client)


    def run(self):

        does_device_need_root=True


        comm_file = self.cfg["dut"].get("comm_file")

        if not comm_file:
            print("comm_file must be defined in config", file=sys.stderr)
            return

        for test in self.tests:

            tag = config_tag(test["config"], test["tag_fields"])

            config_txt = ""

            for i in range(self.cfg["retry"]["max_attempts"]):
                try: 
                    self.client.connect()
                    
                    config_txt = self.device_manager.apply_config(test)

                    break

                except Exception as e:
                    print("Failure:", e)
                    time.sleep(self.cfg["retry"]["cooldown_sec"])

            for iteration in range(test["iterations"]):

                run_dir = self.logger.new_run_dir(test["name"], tag, self.date)

                attempt = 1

                while attempt <= self.cfg["retry"]["max_attempts"]:

                    print("Running", test["name"], tag, "attempt", attempt)
                    
                    self.device_manager.save_metadata(test, run_dir, self.date)

                    try:

                        with open(f"{run_dir}/config.txt", "w") as config_file:
                            config_file.write(config_txt)

                        print("Connected to DUT, starting idle cooldown time")


                        time.sleep(self.cfg["retry"]["cooldown_sec"])



                        # remove trigger file
                        self.client.run(f"rm -f {comm_file}", as_root=does_device_need_root, safe_to_retry=False)

                        exe = test["executable"]
                        build = self.cfg["dut"]["build_dir"]

                        # build remote log directory
                        remote_log_dir = f"{build}/bench_logs/{test['name']}/{run_dir.name}"

                        self.client.run(f"mkdir -p {remote_log_dir}")

                        args = f"--log_dir {remote_log_dir} --comm_file {comm_file}"

                        print("Beginning Executable")

                        # start benchmark in background
                        cmd = (
                            f"(cd {build} && "
                            f"setsid nohup {exe} {args} "
                            f"> {remote_log_dir}/stdout_dump.txt 2>&1 "
                            f"< /dev/null & "
                            f"echo $! > {remote_log_dir}/run.pid)"
                        )

                        code, text, err = self.client.run(cmd, async_run=True, as_root=does_device_need_root)
                        
                        code, pid, _ = self.client.run(f"cat {remote_log_dir}/run.pid", as_root=does_device_need_root)
                        pid = pid.strip()
                        if not pid:
                            raise Exception("Failed to capture benchmark PID")


                        # trigger benchmark
                        self.client.run(f"touch {comm_file}", as_root=does_device_need_root)


                        test_start = time.time()
                        last_check = test_start

                        while True:


                            now = time.time()

                            if now - last_check > 5:
                                # check if benchmark finished
                                code, out, _ = self.client.run(
                                    f"test -f {comm_file} && cat {comm_file}", as_root=does_device_need_root
                                )

                                if code == 0 and "done" in out:
                                    break

                                code, out, err = self.client.run(
                                    f"kill -0 {pid}", as_root=does_device_need_root
                                )
                                alive = (code == 0)

                                if not alive:
                                    raise Exception("Benchmark has died unexpectedly")
                                
                                if now - test_start > test["max_runtime_sec"]:
                                    code, out, _ = self.client.run(
                                        f(f"kill {pid}"), as_root=does_device_need_root
                                    )
                                    raise Exception("Benchmark timeout")

                                last_check = time.time()


                        print("Run completed")

                        self.client.fetch_directory(remote_log_dir, run_dir)

                        break

                    except Exception as e:

                        print("Failure:", e)

                        attempt += 1

                        time.sleep(self.cfg["retry"]["cooldown_sec"])