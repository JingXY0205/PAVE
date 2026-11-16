"""Generate PAVE predictions for the four closed-ended L-Eval tasks."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from PAVE.inference import generate_tokens, load
from .checks import verify_backbone, verify_sha256
from .ruler import FULL_PROTOCOL


TASKS = ('tpo', 'quality', 'coursera', 'sci_fi')
EXAM = ('Now you are given a very long document. Please follow the instruction based on this document. '
        'For multiple-choice questions, there is only one correct option. Please only provide the letter '
        'corresponding to the answer, such as A or B. For other questions, directly provide a concise and accurate answer.')
HEADERS = {
    'coursera': ('Now you are given a very long document. Please follow the instruction based on this document. '
                 'For multiple-choice questions, there could be a single correct option or multiple correct options. '
                 'Please only provide the letter corresponding to the answer (such as A or AB).'),
    'sci_fi': ("Now you are given a science-fiction document. I will ask questions whose answers should be "
               "'True' or 'False'. Answer based on evidence in the document rather than background knowledge."),
}


def prompt_ids(tokenizer, task, document, question, budget):
    prefix = HEADERS.get(task, EXAM) + '\n\nDocument:\n'
    suffix = f'\n\nQuestion:\n{question}\n\nGive only the answer, without additional explanation.\nAnswer:'

    def encode(text):
        return tokenizer(prefix + text + suffix, add_special_tokens=True, truncation=False)['input_ids']

    original = encode(document)
    if len(original) <= budget:
        return original
    best = encode('')
    if len(best) > budget:
        raise ValueError('Question and instruction exceed the configured L-Eval input length')
    doc_ids = tokenizer(document, add_special_tokens=False, truncation=False)['input_ids']
    low, high = 1, len(doc_ids)
    while low <= high:
        middle = (low + high) // 2
        text = tokenizer.decode(doc_ids[:middle], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        candidate = encode(text)
        if len(candidate) <= budget:
            best = candidate
            low = middle + 1
        else:
            high = middle - 1
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-length', type=int, default=32768)
    parser.add_argument('--max-new-tokens', type=int, default=512)
    parser.add_argument('--scorer', type=Path, help='Path to the official L-Eval auto_eval.py')
    args = parser.parse_args()
    if not 1 <= args.max_new_tokens <= 512 or args.max_length <= args.max_new_tokens:
        parser.error('Require a positive prompt budget and a generation limit from 1 to 512')
    for task in TASKS:
        if not (args.data_dir / f'{task}.jsonl').is_file():
            parser.error(f'Missing L-Eval input {task}.jsonl in {args.data_dir}')
    if args.scorer is not None and not args.scorer.is_file():
        parser.error(f'Missing official scorer {args.scorer}')

    import torch
    from transformers import AutoTokenizer

    verify_backbone(args.model_path, FULL_PROTOCOL['model_weight_sha256'])
    verify_sha256(args.checkpoint, FULL_PROTOCOL['checkpoint_sha256'])
    verify_sha256(args.model_path / 'tokenizer.json', FULL_PROTOCOL['tokenizer_sha256'])
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model, core, _, reset = load(args.model_path, args.checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=False)

    def answer(ids):
        generated = generate_tokens(model, core, reset, ids, args.max_new_tokens, tokenizer.eos_token_id)
        return tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()

    budget = args.max_length - args.max_new_tokens
    for task in TASKS:
        destination = args.output_dir / f'{task}.pred.jsonl'
        with (torch.inference_mode(),
              (args.data_dir / f'{task}.jsonl').open(encoding='utf-8') as source,
              destination.open('w', encoding='utf-8') as output):
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                for question, gold in zip(row['instructions'], row['outputs'], strict=True):
                    prediction = answer(prompt_ids(tokenizer, task, row['input'], question, budget))
                    if task == 'sci_fi':
                        fact_question = question.replace('based on the world described in the document.',
                            'based on real-world knowledge and facts up to your training cutoff.')
                        fact_prompt = ('Answer the following question using real-world knowledge.\n\n'
                                       f'Question:\n{fact_question}\n\nAnswer:')
                        fact = answer(tokenizer(fact_prompt, add_special_tokens=True, truncation=False)['input_ids'])
                        prediction += f' [fact: {fact}]'
                    output.write(json.dumps(dict(query=question, gt=gold, evaluation=row.get('evaluation', 'exam_eval'),
                                                 PAVE_pred=prediction), ensure_ascii=False) + '\n')
        if args.scorer:
            subprocess.run([sys.executable, str(args.scorer.resolve()), '--pred_file',
                            str(destination.resolve())], cwd=args.scorer.resolve().parent, check=True)


if __name__ == '__main__':
    main()
