"""Launch PAVE component ablations, design choices, and hyperparameter sweeps."""

from arguments import parse_command
from utils import run_module


COMMANDS = {
    "ablation": ("evaluation.ablation", "Evaluate the component ablations on RULER"),
    "design-choice": ("evaluation.design_choice", "Compare summary and retrieval alternatives"),
    "hyper": ("evaluation.hyper", "Sweep salience temperature and sparsity decay"),
}


def main(argv=None):
    choices = {name: description for name, (_, description) in COMMANDS.items()}
    args = parse_command(__doc__, choices, argv, extra_help="Use COMMAND --help for its options.")
    return run_module(COMMANDS[args.command][0], args.options)


if __name__ == "__main__":
    raise SystemExit(main())
