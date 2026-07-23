from orchestrator.config_loader import load_config
from orchestrator.runner import BenchmarkRunner
from orchestrator.pi_device_manager import CM5DeviceManager
from orchestrator.nano_device_manager import NanoDeviceManager
from orchestrator.opi_device_manager import OPI5DeviceManager
import sys

def main():


    cf_file = ""
    one_off = False

    for arg in sys.argv[1:]:
        if (".json" in arg):
            print(f'Running config {arg}')
            cf_file = arg
        if (arg == "-o" or arg == "--one-off"):
            print("Running one off test")
            one_off = True


    try: 
        config = load_config(cf_file)

        manager = CM5DeviceManager(config)
        #manager = NanoDeviceManager(config)
        #manager = OPI5DeviceManager(config)

        runner = BenchmarkRunner(config, manager, one_off=one_off)

        runner.run()

        runner.close()
    except Exception as e:
        print(f"exception: {e}")



if __name__ == "__main__":
    main()