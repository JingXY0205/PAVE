"""Generate and validate RULER prompts for Llama-2 and Llama-3 Base."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory

from evaluation.ruler import FULL_PROTOCOL


BOS_IDS = {'llama2': 1, 'llama3': 128000}


def generator_command(scripts, task, options, definition, staging, tokenizer, backbone, length, count, seed):
    family = options['task']
    template = definition['template'] + definition.get('answer_prefix', '')
    command = [sys.executable, str(scripts / 'data' / 'synthetic' / f'{family}.py'),
               '--save_dir', str(staging), '--save_name', task, '--subset', 'validation',
               '--tokenizer_path', str(tokenizer), '--tokenizer_type', 'hf',
               '--max_seq_length', str(length - 1 if backbone == 'llama2' else length),
               '--tokens_to_generate', str(definition['tokens_to_generate']), '--num_samples', str(count),
               '--random_seed', str(seed), '--template', template]
    if backbone == 'llama3':
        # Reserve one token for the explicit BOS added after generation.
        command.extend(('--model_template_token', '1'))
    if family == 'qa':
        command.extend(('--pre_samples', '0'))
    for name, value in options['args'].items():
        command.extend((f'--{name}', str(value)))
    return command


def validate_record(row, tokenizer, backbone, length, cap):
    prompt = row['input']
    prefix = row.get('answer_prefix', '')
    outputs = row['outputs']
    if (not isinstance(prompt, str) or not prompt.strip() or not isinstance(prefix, str)
            or not isinstance(outputs, list) or not outputs
            or not all(isinstance(item, str) and item.strip() for item in outputs)):
        raise ValueError('Invalid RULER prompt or answers')
    if backbone == 'llama3':
        if tokenizer.bos_token not in prompt + prefix:
            prompt = tokenizer.bos_token + prompt
        elif not prompt.startswith(tokenizer.bos_token):
            raise ValueError('An explicit BOS may only appear at the beginning of a Llama-3 prompt')
    ids = tokenizer.encode(prompt + prefix, add_special_tokens=backbone == 'llama2')
    bos = BOS_IDS[backbone]
    if not ids or ids[0] != bos or ids.count(bos) != 1:
        raise ValueError('Expected one BOS at the start of the prompt')
    if len(ids) + cap > length:
        raise ValueError(f'Prompt has {len(ids)} tokens, plus {cap} output tokens, for a {length}-token budget')
    return dict(row, input=prompt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backbone', choices=tuple(BOS_IDS), required=True)
    parser.add_argument('--ruler-root', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--length', action='append', type=int,
                        choices=(4096, 8192, 16384, 32768, 65536, 131072))
    args = parser.parse_args()
    root = args.ruler_root.resolve()
    tokenizer_path = args.tokenizer.resolve()
    if not tokenizer_path.is_dir():
        parser.error('Supply a local Hugging Face tokenizer directory')
    config = FULL_PROTOCOL
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != config['ruler_commit']:
        parser.error(f"Expected RULER revision {config['ruler_commit']}, got {revision}")

    import yaml
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    if tokenizer.bos_token_id != BOS_IDS[args.backbone] or not tokenizer.bos_token:
        parser.error(f'Tokenizer BOS does not match {args.backbone}')
    scripts = root / 'scripts'
    options = yaml.safe_load((scripts / 'synthetic.yaml').read_text(encoding='utf-8'))
    definitions = runpy.run_path(str(scripts / 'data' / 'synthetic' / 'constants.py'))['TASKS']
    lengths = list(dict.fromkeys(args.length or config['lengths']))
    output = args.output_dir.resolve() / args.backbone
    count = config['samples_per_task']
    for length in lengths:
        for task in config['tasks']:
            if (output / str(length) / task / 'validation.jsonl').exists():
                parser.error(f'Output already exists for {args.backbone}/{length}/{task}')
    output.mkdir(parents=True, exist_ok=True)
    for length in lengths:
        for task in config['tasks']:
            task_options = options[task]
            definition = definitions[task_options['task']]
            cap = config['generation_caps'][task]
            if definition['tokens_to_generate'] != cap:
                raise ValueError(f'Upstream generation cap differs for {task}')
            with TemporaryDirectory(prefix='ruler-', dir=output) as directory:
                staging = Path(directory)
                command = generator_command(scripts, task, task_options, definition, staging,
                                            tokenizer_path, args.backbone, length, count, config['seed'])
                subprocess.run(command, cwd=scripts, check=True)
                source = staging / task / 'validation.jsonl'
                validated = staging / 'validation.jsonl'
                with (source.open(encoding='utf-8') as records,
                      validated.open('w', encoding='utf-8', newline='\n') as destination):
                    seen = 0
                    for line in records:
                        if line.strip():
                            seen += 1
                            try:
                                row = validate_record(json.loads(line), tokenizer, args.backbone, length, cap)
                            except (ValueError, KeyError) as exc:
                                raise ValueError(f'{args.backbone}/{length}/{task} row {seen}: {exc}') from exc
                            destination.write(json.dumps(row, ensure_ascii=False) + '\n')
                if seen != count:
                    raise ValueError(f'{args.backbone}/{length}/{task} has {seen} records, expected {count}')
                final = output / str(length) / task / 'validation.jsonl'
                final.parent.mkdir(parents=True, exist_ok=True)
                validated.replace(final)
            print(f'{args.backbone}/{length}/{task}: {count} examples', flush=True)


if __name__ == '__main__':
    main()
