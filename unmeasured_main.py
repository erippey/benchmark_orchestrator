from orchestrator.config_loader import load_config
from orchestrator.unmeasured_runner import BenchmarkRunnerUnmeasured
from orchestrator.pi_device_manager import CM5DeviceManager
from orchestrator.nano_device_manager import NanoDeviceManager
from orchestrator.opi_device_manager import OPI5DeviceManager

def main():

    config = load_config("config.json")

    manager = OPI5DeviceManager(config)
    # manager = CM5DeviceManager(config)
    # manager = NanoDeviceManager(config)

    runner = BenchmarkRunnerUnmeasured(config, manager)

    runner.run()


if __name__ == "__main__":
    main()