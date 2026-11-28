import tempfile
import unittest
from pathlib import Path

import torch

from dexmani_rl.policies.flowmatch.loader import load_flowmatch_policy
from dexmani_rl.policies.flowmatch.model.flowmatch import FlowMatch


class FlowMatchLoaderTests(unittest.TestCase):
    def test_load_raw_and_ema_with_fixed_noise(self):
        agent = dict(horizon=4, n_action_steps=2, n_layers=1, hidden_dim=24, n_head=2,
                     timestep_embed_dim=16, pc_encoder_config=dict(channels=24, num_heads=2),
                     rgb_backbone_config=dict(architecture=dict(model_type='dinov2', hidden_size=24,
                         num_hidden_layers=1, num_attention_heads=2, image_size=16, patch_size=8)))
        model = FlowMatch(**agent).eval()
        model.configure_normalizer({'action': 19}, {'action': (19,)})
        cfg = dict(agent=dict(agent, _target_='external.FlowMatch'), action_key='action', dataset={},
                   normalization_dims={'action': 19}, normalization={'action': 'limits'},
                   eval={'use_ema': True, 'inference_steps': 2})
        raw = model.state_dict()
        ema = dict(raw)
        ema['normalizer.fields.action.offset'] = torch.ones(19)
        context, noise = torch.randn(1, 6, model.obs_encoder.out_dim), torch.randn(1, 4, 19)
        expected = model.decode_condition(context, noise, 2)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bc.pt'
            torch.save(dict(effective_config=cfg, model_state=raw, ema_model_state=ema), path)
            for selection, offset in [('raw', 0.), (None, 1.)]:
                loaded, config, weights = load_flowmatch_policy(path, 'cpu', selection)
                self.assertEqual(config, cfg)
                self.assertEqual(weights, selection or 'ema')
                self.assertFalse(any(p.requires_grad for p in loaded.parameters()))
                torch.testing.assert_close(loaded.normalizer['action'].offset, torch.full((19,), offset))
                torch.testing.assert_close(loaded.decode_condition(context, noise, 2), expected)
