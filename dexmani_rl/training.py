"""Shared command-line inputs and validation bookkeeping for the training scripts."""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf


def training_arguments(algorithm):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--resume')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--weights', choices=['ema', 'raw'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.resume:
        if args.config or args.weights:
            parser.error('--resume uses the saved config and weight selection')
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (not isinstance(saved, dict) or saved.get('algorithm') != algorithm
                or 'model' not in saved):
            parser.error(f'--resume requires a {algorithm} training snapshot (resume.pt)')
        return args, saved['config'], saved['base_checkpoint'], saved['weights'], saved

    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).resolve().parents[1] / 'configs' / f'{algorithm}.yaml'
        if not config_path.is_file():
            parser.error('Default configuration is unavailable. Pass --config with a YAML file.')
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return args, cfg, str(Path(args.checkpoint).resolve()), args.weights, None


def selection_deadline(training_deadline, deadline, now, validation_count, test_count):
    # After the training window, share the remaining budget by evaluation episode count.
    start = min(deadline, max(training_deadline, now))
    return start + (deadline - start) * validation_count / (validation_count + test_count)


def record_validation(results, expected, phase, log, **progress):
    complete = len(results) == expected and expected > 0
    score = sum(row['success'] for row in results) / expected if complete else None
    record = dict(progress, validation=results, success_rate=score,
                  validation_complete=complete, validation_phase=phase)
    log.write(json.dumps(record) + '\n')
    return score


def actor_snapshot(actor):
    return {name: value.detach().cpu().clone() for name, value in actor.state_dict().items()}
