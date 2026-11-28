"""Shared command-line inputs and validation bookkeeping for the training scripts."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from .base_policy import load_policy
from .checkpoint import capture_rng, save
from .env import seed_split


def training_arguments(algorithm):
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--checkpoint')
    source.add_argument('--resume')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--weights', choices=['ema', 'raw'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    if algorithm == 'bora':
        parser.add_argument('--through', choices=['offline', 'round1', 'round2'], default='round2',
                            help='Last stage to run before saving the training snapshot')
        parser.add_argument('--demonstrations', type=Path, help='Directory containing episode_*.h5')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.resume:
        if args.config or args.weights or getattr(args, 'demonstrations', None):
            parser.error('--resume uses the saved config, weight selection and data source')
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if (not isinstance(saved, dict) or saved.get('algorithm') != algorithm
                or 'model' not in saved):
            parser.error(f'--resume requires a {algorithm} training snapshot (resume.pt)')
        return args, saved['config'], saved['base_checkpoint'], saved['weights'], saved

    config_path = args.config
    if config_path is None:
        config_path = Path(__file__).resolve().parents[2] / 'configs' / f'{algorithm}.yaml'
        if not config_path.is_file():
            parser.error('Default configuration is unavailable. Pass --config with a YAML file.')
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    return args, cfg, str(Path(args.checkpoint).resolve()), args.weights, None


class OnlineRun:
    def __init__(self, algorithm, args, cfg, checkpoint, weights, saved):
        started = time.monotonic()
        torch.manual_seed(cfg['seed'])
        self.rng = np.random.default_rng(cfg['seed'])
        self.policy, self.policy_cfg, weights = load_policy(checkpoint, args.device, weights)
        if saved is None:
            test, validation, excluded = seed_split(self.policy_cfg['env_runner']['task_name'])
        else:
            test, validation, excluded = saved['test_seeds'], saved['validation_seeds'], saved['excluded_seeds']
        self.test, self.validation, self.excluded = test, validation, set(excluded)
        self.metadata = {'algorithm': algorithm, 'config': cfg, 'base_checkpoint': checkpoint, 'weights': weights,
                         'torch': str(torch.__version__), 'test_seeds': test, 'validation_seeds': validation,
                         'excluded_seeds': sorted(excluded)}
        self.output = args.output
        elapsed = saved['elapsed'] if saved is not None else 0.
        self.time_origin = started - elapsed
        self.deadline = max(started, self.time_origin + cfg['max_hours'] * 3600)
        self.training_deadline = max(started, self.time_origin + cfg['train_hours'] * 3600)
        self.best_score = saved['best_score'] if saved is not None else -1.
        self.best_actor = saved['best_actor'] if saved is not None else None

    def open_output(self):
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / 'config.json').write_text(json.dumps(dict(self.metadata, policy_config=self.policy_cfg), indent=2))

    def snapshot_fields(self, device):
        return dict(self.metadata, elapsed=time.monotonic() - self.time_origin,
                    best_score=self.best_score, best_actor=self.best_actor, **capture_rng(self.rng, device))

    def selection_deadline(self, test_policy_count):
        # Divide the remaining budget in proportion to the number of episodes.
        start = min(self.deadline, max(self.training_deadline, time.monotonic()))
        validation_count = len(self.validation)
        episode_count = validation_count + test_policy_count * len(self.test)
        return start + (self.deadline - start) * validation_count / episode_count

    def record_validation(self, results, actor, phase, log, progress, **actor_fields):
        complete = len(results) == len(self.validation) and len(results) > 0
        score = sum(row['success'] for row in results) / len(results) if complete else None
        record = dict(progress, validation=results, success_rate=score,
                      validation_complete=complete, validation_phase=phase)
        log.write(json.dumps(record) + '\n')
        if score is not None and score > self.best_score:
            self.best_score = score
            self.best_actor = {key: tensor.detach().to("cpu", copy=True) for key, tensor in actor.state_dict().items()}
            save(self.output / 'best.pt', dict(self.metadata, actor=self.best_actor, **actor_fields))
        return score

    def selected_actor(self, latest, **actor_fields):
        if self.best_actor is None:
            return 'last.pt', latest
        path = self.output / 'best.pt'
        if not path.exists():
            save(path, dict(self.metadata, actor=self.best_actor, **actor_fields))
        actor = copy.deepcopy(latest)
        actor.load_state_dict(self.best_actor)
        return 'best.pt', actor
