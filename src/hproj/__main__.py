import click
from dotenv import load_dotenv

from hproj.cli import (
    calibrate_classifier,
    calibrate_projector,
    clustering,
    estimate_curve,
    example,
    generalise,
    intrinsic_dims,
    report,
    thresholds,
)

load_dotenv()


@click.group(help="CLI tool for projecting and characterising histology embeddings.")
def cli():
    pass


cli.add_command(example)
cli.add_command(calibrate_projector)
cli.add_command(calibrate_classifier)
cli.add_command(clustering)
cli.add_command(estimate_curve)
cli.add_command(generalise)
cli.add_command(thresholds)
cli.add_command(intrinsic_dims)
cli.add_command(report)


def main():
    cli(prog_name="hproj")


if __name__ == "__main__":
    main()
