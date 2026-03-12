

def update_config(text, params):
    lines = text.splitlines()
    found = set()

    for i,line in enumerate(lines):

        for key,val in params.items():

            if line.startswith(key + "="):
                lines[i] = f"{key}={val}"
                found.add(key)

    for key,val in params.items():
        if key not in found:
            lines.append(f"{key}={val}")

    return "\n".join(lines) + "\n"