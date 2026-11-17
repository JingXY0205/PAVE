"""Launch PAVE data preparation, training, evaluation, and analysis."""

from arguments import parse_command
from utils import ROOT, run_module


COMMANDS = {
    "prepare-longalpaca": ("dataset.longalpaca.prepare", "Prepare the shared LongAlpaca-12k splits"),
    "prepare-ruler": ("dataset.ruler.prepare", "Generate RULER prompts for either backbone"),
    "prepare-pg19": ("dataset.ppl.rawtext_to_bin", "Tokenize the PG-19 input"),
    "prepare-arxiv": ("dataset.ppl.prepare_proof_pile_arxiv", "Prepare Proof-Pile-ArXiv token streams"),
    "train": ("training.train", "Train PAVE"),
    "ruler": ("evaluation.ruler", "Evaluate PAVE on RULER"),
    "l-eval": ("evaluation.l_eval", "Generate PAVE predictions for L-Eval"),
    "ppl": ("evaluation.ppl", "Evaluate PAVE perplexity"),
    "ablation": ("evaluation.ablation", "Evaluate the component ablations on RULER"),
    "design-choice": ("evaluation.design_choice", "Compare summary and retrieval alternatives"),
    "hyper": ("evaluation.hyper", "Sweep salience temperature and sparsity decay"),
}


def run_baseline(argv):
    methods = {
        path.parent.name: "Run a baseline"
        for path in sorted((ROOT / "baselines").glob("*/run.py"))
    }
    args = parse_command(
        "Launch a baseline runner.", methods, argv,
        extra_help="Use METHOD download --help, METHOD evaluate --help, or METHOD train --help for FT methods.",
    )
    action = "run"
    options = args.options
    if options[:1] == ["train"] and (ROOT / "baselines" / args.command / "train.py").is_file():
        action = "train"
        options = options[1:]
    return run_module(f"baselines.{args.command}.{action}", options)


def main(argv=None):
    choices = {name: description for name, (_, description) in COMMANDS.items()}
    choices["baseline"] = "Download, evaluate, or train a baseline"
    args = parse_command(
        __doc__, choices, argv,
        extra_help="Use COMMAND --help for its options.",
    )
    if args.command == "baseline":
        return run_baseline(args.options)
    return run_module(COMMANDS[args.command][0], args.options)


if __name__ == "__main__":
    raise SystemExit(main())
