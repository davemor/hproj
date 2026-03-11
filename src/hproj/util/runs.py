from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import yaml



def generate_run_id(experiment_name: str, config_path: Path) -> str:
    """
    Generate a run ID in the format:
    <experiment>-<timestamp>-<config-hash>

    Note that by default the experiment name is the name of the config file without extension
    """

    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # convet config to a sorted string representation
    config_str = json.dumps(data, sort_keys=True, separators=(",", ":"))

    # hash the config string to get a unique identifier, limiting to 8 chars
    config_hash = hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:8]

    # normalize experiment name
    name = re.sub(r"[^a-z0-9]+", "-", experiment_name.lower()).strip("-")

    # get the timestamp
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")

    # combine to form the run ID
    run_id = f"{name}-{timestamp}-{config_hash}"
    return run_id
