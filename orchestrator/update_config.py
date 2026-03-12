

def update_config(text, params):
    managed_keys = [
        "arm_freq",
        "arm_freq_min",
        "core_freq",
        "core_freq_min",
        "v3d_freq",
        "v3d_freq_min",
        "gpu_freq",
        "gpu_freq_min",
        "over_voltage",
        "over_voltage_min",
    ]

    lines = text.splitlines()
    output = []
    seen_managed = set()

    for line in lines:
        stripped = line.strip()
        handled = False

        for key in managed_keys:
            active_prefix = f"{key}="
            commented_prefix = f"#{key}="

            if stripped.startswith(active_prefix) or stripped.startswith(commented_prefix):
                if key in seen_managed:
                    handled = True
                    break

                seen_managed.add(key)

                if key in params:
                    output.append(f"{key}={params[key]}")
                else:
                    output.append(f"#{key}=")
                handled = True
                break

        if not handled:
            output.append(line)

    for key in managed_keys:
        if key in params and key not in seen_managed:
            output.append(f"{key}={params[key]}")
        elif key not in params and key not in seen_managed:
            # optional: append commented default marker for visibility
            # output.append(f"#{key}=")
            pass

    return "\n".join(output) + "\n"