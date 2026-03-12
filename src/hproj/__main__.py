import click
from dotenv import load_dotenv

from hproj.cli import calibrate, example

load_dotenv()


@click.group(help="CLI tool for projecting and characterising histology embeddings.")
def cli():
    pass


cli.add_command(example)
cli.add_command(calibrate)


def main():
    cli(prog_name="hproj")


if __name__ == "__main__":
    main()
