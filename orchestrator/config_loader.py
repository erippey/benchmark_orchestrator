import json


def load_config(path):

    with open(path) as f:
        return json.load(f)


def expand_tests(cfg, one_off=False):

    expanded = []

    for test in cfg["tests"]:

        sweep = test["sweep"]
        var = sweep["var"]
        values = sweep["values"]

        if one_off:
            if not values:
                raise ValueError(f"One-off run requested, but test '{test['name']}' has an empty sweep")

            values = values[:1]

        for val in values:

            run_cfg = dict(test["fixed_config"])
            run_cfg[var] = val

            expanded.append({
                "name": test["name"],
                "executable": test["executable"],
                "governor": test["governor"],
                "config": run_cfg,
                "independant_var": var,
                "tag_fields": test["tag_fields"],
                "iterations": 1 if one_off else test["total_iterations"],
                "max_runtime_sec": test["max_runtime_sec"],
                "one_off": one_off
            })

    return expanded


def config_tag(config, fields):

    parts = []

    for f in fields:
        parts.append(f"{f}{config[f]}")

    return "_".join(parts)
