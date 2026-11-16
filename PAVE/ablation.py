"""Run the PAVE routing, summary, and Remote retention ablations."""
from .checks import write_json
from .full import FULL_PROTOCOL, main

VARIANTS = ('fixed_regions', 'max_pool', 'fixed_retention', 'full')
PROTOCOLS = {
    32768: dict(FULL_PROTOCOL, lengths=[32768], variants=VARIANTS),
    65536: dict(FULL_PROTOCOL, lengths=[65536], variants=VARIANTS),
}


def install(model, core, config, mode, output):
    import torch

    if mode == 'fixed_regions':
        core.ABLATION_MODE = 'fixed_router'
        core.FIXED_ROUTER_RATIOS = (1 / 3, 1 / 3, 1 / 3)
    elif mode == 'max_pool':
        def max_pool(module, blocks):
            representative = blocks.amax(dim=-2)
            weights = blocks.new_zeros(blocks.shape[:-1])
            return representative, weights, blocks.new_zeros(())

        import types
        for layer in model.model.layers:
            module = layer.self_attn.block_representative
            module.forward = types.MethodType(max_pool, module)
    elif mode == 'fixed_retention':
        config.sparsity_decay_strength = 0.0
    else:
        assert mode == 'full'

    expected = dict(mode=mode, inference_only=True, fixed_regions=[1 / 3] * 3 if mode == 'fixed_regions' else None,
                    pooling='elementwise_max_over_tokens' if mode == 'max_pool' else 'learned_salience',
                    initial_remote_keep_ratio=config.initial_remote_keep_ratio,
                    decay_strength=config.sparsity_decay_strength)
    assert config.initial_remote_keep_ratio == 0.5
    if mode == 'fixed_retention':
        assert all(layer.self_attn._pave_config.sparsity_decay_strength == 0 for layer in model.model.layers)
    if mode == 'max_pool':
        example = torch.tensor([[[[[3., -1.], [1., 2.]]]]], device=next(model.parameters()).device)
        actual = model.model.layers[0].self_attn.block_representative(example)[0]
        assert torch.equal(actual, torch.tensor([[[[3., 2.]]]], device=actual.device))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'effective_ablation.json', expected)

    original_proposal = core.make_proposal

    def proposed(*args, **kwargs):
        proposal = original_proposal(*args, **kwargs)
        if proposal.remote and mode == 'fixed_retention':
            assert abs(proposal.ratio - 0.5) < 1e-12
        if mode == 'fixed_regions':
            target = proposal.ratios.new_tensor([1 / 3] * 3)
            assert torch.allclose(proposal.ratios, target)
        return proposal

    core.make_proposal = proposed


if __name__ == '__main__':
    main(PROTOCOLS, install)
