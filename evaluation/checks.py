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


def verify_llama3_backbone(model_path, *, check_weights=True):
    config = read_json(model_path / 'config.json')
    expected = dict(
        model_type='llama', hidden_size=4096, num_hidden_layers=32, num_attention_heads=32,
        num_key_value_heads=8, vocab_size=128256, max_position_embeddings=8192,
        rope_theta=500000, bos_token_id=128000, eos_token_id=128001, rope_scaling=None,
    )
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Expected the original Meta-Llama-3-8B Base configuration')
    if check_weights:
        shards = (
            'f2c144103072514542e327fa8080bd375cb300f2d453fba9ca3aea81d0d4cf33',
            'd9eee5f23d94405d90b7e9ff88b9443fee42f8528a658f54214c2aba7530d80c',
            '4b8fbc5e113f69768dd8de84661ea20af8a32b734a9976144b4236c447b40ccc',
            '5dc34e6bdf2da9e35f0d93b5c333c870f3677dc43dc3a91ea3a8ad28a1fe1acb',
        )
        for index, digest in enumerate(shards, 1):
            verify_sha256(model_path / f'model-{index:05}-of-00004.safetensors', digest)
