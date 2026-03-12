from orchestrator.config_loader import load_config
from orchestrator.runner import BenchmarkRunner


def main():

    config = load_config("config.json")

    runner = BenchmarkRunner(config)

    runner.run()


if __name__ == "__main__":
    main()