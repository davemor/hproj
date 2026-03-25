import click

from hproj.data.paths import Paths
from hproj.util.config import Config
from hproj.util.logging import setup_logging


@click.command()
@click.option("--run-id", "-r", type=str, required=True, help="The id of a run to continue.")
def estimate_curve(run_id: str):
    paths = Paths.from_env()
    run_paths = paths.run(run_id)
    cfg = Config.from_yaml(run_paths.config())

    logger = setup_logging(run_paths.log_file())
    logger.info("Running the threshold identification phase of the experiment.")
    logger.info(f"Run id is {run_id}.")

    