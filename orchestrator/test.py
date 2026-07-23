import threading

from wattsup import *
from logger import *
from ssh_client import *
from pi_device_manager import *

max_runtime_sec = 6 # in seconds

meter = WattsUpMeter(
    port="/dev/ttyUSB0"
)
logger = RunLogger("./")
run_dir = Path("./")

client = DUTClient(
    '100.65.92.213',
    'dev',
    root_access=False,
    key_filename="/home/erippey3-laptop/.ssh/id_rsa"
)
client.connect()


# root directory for executable
remote_bin_dir = "/home/dev/tmp"
exe = "ls"
args = "-a -l"
# root directory for log files
remote_log_dir = "/home/dev/tmp"




input("Press Enter to begin power logging...")

cmd = (
    f"(cd {remote_bin_dir} && "
    f"setsid nohup {exe} {args} "
    f"&> {remote_log_dir}/stdout.txt 2>&1 "
    f"< /dev/null & "
    f"echo $! > {remote_log_dir}/run.pid)"
)

code, text, err = client.run(cmd, async_run=True)
                        
                        
code, pid, _ = client.run(f"cat {remote_log_dir}/run.pid")
pid = pid.strip()


f, writer = logger.open_power_log(
    run_dir,
    "run_power.csv",
)


test_start = time.time()
last_check = test_start

while True:

    sample = meter.read_sample()

    writer.writerow([
        sample.timestamp,
        sample.watts,
        sample.volts,
        sample.amps
    ])

    now = time.time()


    if now - last_check > 2:

        code, out, err = client.run(
            f"kill -0 {pid}")
        alive = (code == 0)

        if not alive:
            break


                                
        if now - test_start > max_runtime_sec:
            code, out, _ = client.run(
                f"kill {pid}")
            print("benchmark timeout")
            break

        last_check = time.time()


client.fetch_directory(remote_log_dir, "./")

f.close()
client.close()

print("Power logging stopped.")