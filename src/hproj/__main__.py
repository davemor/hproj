import click
from dotenv import load_dotenv

from hproj.cli import calibrate_projector, calibrate_classifier, example

load_dotenv()


@click.group(help="CLI tool for projecting and characterising histology embeddings.")
def cli():
    pass


cli.add_command(example)
cli.add_command(calibrate_projector)
cli.add_command(calibrate_classifier)


def main():
    cli(prog_name="hproj")


if __name__ == "__main__":
    main()
