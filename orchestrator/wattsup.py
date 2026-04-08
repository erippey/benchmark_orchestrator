import serial
from dataclasses import dataclass
from datetime import datetime

@dataclass
class PowerSample:
    timestamp: str
    watts: float
    volts: float
    amps: float
    raw: str

class WattsUpMeter:
    def __init__(self, port="COM3", baud=115200, timeout=1):
        self.ser = serial.Serial(port, baudrate=baud, timeout=timeout)

    def start_stream(self, interval_sec=1):
        if int(interval_sec) != interval_sec:
            raise ValueError("WattsUp command interval should be integer seconds")
        cmd = f"#L,W,3,E,,{int(interval_sec)};"
        self.ser.write(cmd.encode("ascii"))

    def read_sample(self) -> PowerSample:
        while True:
            line = self.ser.readline().decode(errors="ignore").strip()
            if not line.startswith("#d"):
                continue

            fields = line.split(",")
            if len(fields) < 6:
                continue

            try:
                watts=float(float(fields[3]) / 10.0)
            except:
                watts=float(0)
            try:
                volts=float(float(fields[4]) / 10.0)
            except:
                volts=float(0)
            try:
                amps=float(float(fields[5]) / 1000.0)
            except:
                amps=float(0)

            return PowerSample(
                timestamp=datetime.now().isoformat(),
                watts=watts,
                volts=volts,
                amps=amps,
                raw=line,
            )