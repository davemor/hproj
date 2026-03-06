import click

from dotenv import load_dotenv

from hproj.cli import example 

load_dotenv()

@click.group(help="CLI tool for projecting and characterising histology embeddings.")
def cli():
    pass


cli.add_command(example)


def main():
    cli(prog_name="hproj")


if __name__ == "__main__":
    main()
