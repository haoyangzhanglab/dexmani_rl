import argparse
import json
from pathlib import Path
from pickle import UnpicklingError

import torch
from omegaconf import OmegaConf

RECIPE_DIR = Path(__file__).resolve().parents[2] / 'recipes'


def training_parser():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--resume')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--weights', choices=['ema', 'raw'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    return parser


def training_inputs(parser, args, *, method, default_config):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.resume:
        if args.config or args.weights:
            parser.error('--resume uses the saved config, weight selection and data source')
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (not isinstance(saved, dict) or saved.get('algorithm') != method
                or 'model' not in saved):
            parser.error(f'--resume requires a {method} training snapshot (resume.pt)')
        return saved['config'], saved['base_checkpoint'], saved['weights'], saved

    cfg = OmegaConf.to_container(OmegaConf.load(args.config or default_config), resolve=True)
    return cfg, str(Path(args.checkpoint).resolve()), args.weights, None


def evaluation_inputs(method):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--rl-checkpoint')
    parser.add_argument('--weights', choices=['ema', 'raw'])
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, help='default: full test pool (100)')
    args = parser.parse_args()
    if args.episodes is not None and args.episodes <= 0:
        parser.error('--episodes must be positive')
    if args.output.exists():
        raise FileExistsError(args.output)
    saved = None
    if args.rl_checkpoint:
        if args.weights:
            parser.error('--rl-checkpoint uses the saved base policy and weight selection')
        actor_error = '--rl-checkpoint requires a saved RL actor'
        try:
            saved = torch.load(args.rl_checkpoint, map_location='cpu', weights_only=True)
        except UnpicklingError as error:
            raise ValueError(actor_error) from error
        if not isinstance(saved, dict) or 'algorithm' not in saved or 'actor' not in saved:
            raise ValueError(actor_error)
        if saved['algorithm'] != method:
            parser.error(f'Expected a {method} actor checkpoint')
        checkpoint, weights = saved['base_checkpoint'], saved['weights']
    else:
        checkpoint, weights = str(Path(args.checkpoint).resolve()), args.weights
    return args, checkpoint, weights, saved


def evaluation_seeds(saved, policy_cfg, episodes):
    if saved is None:
        from dexmani_rl.envs.dexmani_sim import seed_split
        seeds, _, _ = seed_split(policy_cfg['env_runner']['task_name'])
    else:
        seeds = saved['test_seeds']
    if episodes is not None and episodes > len(seeds):
        raise ValueError(f'--episodes must not exceed the test seed count ({len(seeds)})')
    return seeds[:episodes]


def write_evaluation(args, checkpoint, weights, saved, report):
    result = dict(algorithm=saved['algorithm'] if saved is not None else 'bc',
                  checkpoint=checkpoint, weights=weights,
                  rl_checkpoint=str(Path(args.rl_checkpoint).resolve()) if args.rl_checkpoint else None,
                  **report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2)
    with args.output.open('x') as file:
        file.write(serialized)
    summary = report['comparison'] if saved is not None else {'episodes': len(report['baseline'])}
    print(json.dumps(summary))
