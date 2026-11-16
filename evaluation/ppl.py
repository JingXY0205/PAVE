"""Score default PAVE perplexity on prepared PG-19 and ArXiv token windows."""
from collections import defaultdict
import json

from PAVE.inference import load
from .checks import verify_backbone, verify_sha256, write_json
from .hyper import evaluate_window, ppl_arguments, summarize_windows
from .ruler import FULL_PROTOCOL


def main():
    parser = ppl_arguments(__doc__, input_help='Prepared PG-19 or ArXiv token windows (JSONL)')
    parser.add_argument('--length', type=int, action='append', help='Score a prefix of this length. May be repeated.')
    args = parser.parse_args()
    verify_backbone(args.model_path, FULL_PROTOCOL['model_weight_sha256'])
    verify_sha256(args.checkpoint, FULL_PROTOCOL['checkpoint_sha256'])
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import torch

    torch.manual_seed(42)
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    model, core, _, _ = load(args.model_path, args.checkpoint, decode=False)
    grouped = defaultdict(list)
    with (args.inputs.open(encoding='utf-8') as source,
          (args.output_dir / 'windows.jsonl').open('w', encoding='utf-8') as output):
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            size = len(row['input_ids'])
            if row['dataset'] not in ('pg19', 'arxiv'):
                raise ValueError(f"Unsupported PPL dataset {row['dataset']}")
            for length in args.length or [size]:
                if not 1 <= row['target_start'] < length <= size:
                    raise ValueError(f"Invalid score range for window {row['id']} at length {length}")
                result = evaluate_window(model, core, row, length)
                grouped[row['dataset'], length].append(result)
                output.write(json.dumps(dict(dataset=row['dataset'], length=length, id=row['id'], **result)) + '\n')
    if not grouped:
        raise ValueError('No PPL windows supplied')
    write_json(args.output_dir / 'summary.json', [
        dict(dataset=dataset, length=length, **summarize_windows(windows))
        for (dataset, length), windows in sorted(grouped.items())
    ])


if __name__ == '__main__':
    main()
