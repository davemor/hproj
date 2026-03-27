import csv
from datetime import timedelta
from pathlib import Path


def append_stage_time(run_root: Path, stage: str, elapsed_seconds: float):
    """Append a row to times.csv in the run directory."""
    delta_time = timedelta(seconds=elapsed_seconds)
    times_path = run_root / "times.csv"
    write_header = not times_path.exists()
    with open(times_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["stage", "time", "delta_time"])
        writer.writerow([stage, f"{elapsed_seconds:.2f}", str(delta_time)])
