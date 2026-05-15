from orchestrator.config_loader import load_config
from orchestrator.runner import BenchmarkRunner
from orchestrator.pi_device_manager import CM5DeviceManager
from orchestrator.nano_device_manager import NanoDeviceManager

def main():

    config = load_config("config.json")

    manager = CM5DeviceManager(config)
    # manager = NanoDeviceManager(config)

    runner = BenchmarkRunner(config, manager)

    runner.run()


if __name__ == "__main__":
    main()