from orchestrator.config_loader import load_config
from orchestrator.runner import BenchmarkRunner
from orchestrator.pi_device_manager import CM5DeviceManager
from orchestrator.nano_device_manager import NanoDeviceManager
import sys

def main():


    for cf_file in sys.argv[1:]:
        try: 
            config = load_config(cf_file)

            #manager = CM5DeviceManager(config)
            manager = NanoDeviceManager(config)

            runner = BenchmarkRunner(config, manager)

            runner.run()

            runner.close()
        except Exception as e:
            print(f"exception: {e}")
            continue



if __name__ == "__main__":
    main()