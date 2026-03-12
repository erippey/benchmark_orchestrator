from orchestrator.stability import StabilityDetector
from orchestrator.wattsup import WattsUpMeter, PowerSample

stability = StabilityDetector(window_size=15)
wattsup = WattsUpMeter()

wattsup.start_stream()

def wait_for_idle():

    stability.clear()
    print("Ranges and steps ", end='', flush=True)
    

    while True: 

        for i in range(10):
            sample = wattsup.read_sample()
            
            if stability.update(sample.watts):
                print('')
                return
            
        rnge, step = stability.get_range_step()

        print(f'[{rnge}, {step}], ', end='', flush=True)

wait_for_idle()
                