from copy import deepcopy
from pathlib import Path


def update_config(config, optimizer_config, aconfig):
    o = deepcopy(optimizer_config)
    c = deepcopy(config)
    for key in aconfig:
        if key in o:
            o[key] = aconfig[key]
        elif key in c:
            c[key] = aconfig[key]
        else:
            raise ValueError(f"Key {key} not recognized.")
    c.update(
        {
            "optimizer_config": o,
            "checkpoint_path": str(Path("checkpoints") / c["project"] / c["env_name"]),
        }
    )
    return c
