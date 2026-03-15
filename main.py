from orchestrator.config_loader import load_config
from orchestrator.runner import BenchmarkRunner
from orchestrator.device_manager import CM5DeviceManager


def main():

    config = load_config("config.json")

    manager = CM5DeviceManager(config)

    runner = BenchmarkRunner(config, manager)

    runner.run()


if __name__ == "__main__":
    main()