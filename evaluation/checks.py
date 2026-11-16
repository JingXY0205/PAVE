"""Artifact integrity checks shared by the release entry points."""
import hashlib
import json


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def verify_sha256(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f'SHA-256 mismatch for {path}: expected {expected}, got {actual}')


def verify_backbone(model_path, shards):
    config = read_json(model_path / 'config.json')
    if config.get('model_type') != 'llama' or config.get('num_hidden_layers') != 32:
        raise ValueError('Expected a 32-layer Llama-2-7B Base checkpoint')
    if config.get('max_position_embeddings') != 4096 or config.get('rope_scaling') is not None:
        raise ValueError('Expected the original 4K Llama-2 configuration')
    for name, digest in shards.items():
        verify_sha256(model_path / name, digest)
