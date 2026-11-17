"""Parse launcher commands and pass remaining options to the selected runner."""

import argparse


def parse_command(description, commands, argv=None, *, extra_help=None):
    entries = "\n".join(f"  {name:<20} {help_text}" for name, help_text in commands.items())
    epilog = f"Commands\n{entries}"
    if extra_help:
        epilog += f"\n\n{extra_help}"
    parser = argparse.ArgumentParser(
        description=description, epilog=epilog, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("command", choices=commands, metavar="COMMAND")
    parser.add_argument("options", nargs=argparse.REMAINDER, metavar="OPTIONS")
    return parser.parse_args(argv)
