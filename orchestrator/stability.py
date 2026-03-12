from collections import deque

class StabilityDetector:
    def __init__(self, window_size=30, required_consecutive=3,
    max_range_watts=0.2, max_step_delta_watts=0.1):
        self.window = deque(maxlen=window_size)
        self.required_consecutive = required_consecutive
        self.max_range_watts = max_range_watts
        self.max_step_delta_watts = max_step_delta_watts
        self.consecutive_good = 0

    
    def update(self, watts: float) -> bool:
        self.window.append(watts)

        if len(self.window) < self.window.maxlen:
            self.consecutive_good = 0
            return False

        vals = list(self.window)
        range_watts = max(vals) - min(vals)
        max_step = max(abs(vals[i] - vals[i-1]) for i in range(1, len(vals)))

        stable = (
            range_watts <= self.max_range_watts and
            max_step <= self.max_step_delta_watts
        )

        if stable:
            self.consecutive_good += 1
        else:
            self.consecutive_good = 0

        return self.consecutive_good >= self.required_consecutive
    
    def get_range_step(self):
        vals = list(self.window)

        range_watts = max(vals) - min(vals)
        max_step = max(abs(vals[i] - vals[i-1]) for i in range(1, len(vals)))

        return range_watts, max_step
    
    def clear(self):
        self.window.clear()
        self.consecutive_good = 0