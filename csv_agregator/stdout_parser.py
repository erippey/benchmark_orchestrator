import csv
import re


class StdoutParser:
    def __init__(self, path):
        self.path = path
        self.rows = []

    def parse(self):
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        current = None

        for line in lines:
            line = line.strip()

            # Detect new event
            m = re.match(r"Event\s+\d+:\s+(.+)", line)
            if m:
                if current:
                    self.rows.append(current)

                current = {
                    "event": m.group(1).strip(),
                    "runs": None,
                    "avg_time": None,
                    "min_time": None,
                    "max_time": None,
                    "std_dev": None,
                }
                continue

            if not current:
                continue

            if line.startswith("Runs"):
                current["runs"] = int(line.split(":")[1].strip())

            elif line.startswith("Avg time"):
                current["avg_time"] = float(line.split(":")[1].split()[0])

            elif line.startswith("Min time"):
                val = line.split(":")[1].split()[0]
                current["min_time"] = float(val) if val not in ["inf", "-inf"] else val

            elif line.startswith("Max time"):
                val = line.split(":")[1].split()[0]
                current["max_time"] = float(val) if val not in ["inf", "-inf"] else val

            elif line.startswith("Std Dev"):
                current["std_dev"] = float(line.split(":")[1].split()[0])

        if current:
            self.rows.append(current)

    def write_csv(self, output):
        with open(output, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "event",
                    "runs",
                    "avg_time",
                    "min_time",
                    "max_time",
                    "std_dev",
                ],
            )

            writer.writeheader()
            writer.writerows(self.rows)


if __name__ == "__main__":
    parser = StdoutParser("stdout_dump.txt")
    parser.parse()
    parser.write_csv("benchmarks.csv")