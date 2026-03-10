from itertools import product


def make_param_grid(grid_spec: dict) -> list[dict]:
    keys = list(grid_spec.keys())
    values = [grid_spec[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*values)]
