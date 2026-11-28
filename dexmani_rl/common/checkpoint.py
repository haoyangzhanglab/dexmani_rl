import torch


def save(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def capture_rng(seed_rng, device):
    return {'seed_rng': seed_rng.bit_generator.state, 'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, seed_rng, device):
    torch.set_rng_state(state['torch_rng'])
    if device.type == 'cuda' and state['cuda_rng'] is not None:
        torch.cuda.set_rng_state(state['cuda_rng'], device)
    seed_rng.bit_generator.state = state['seed_rng']
