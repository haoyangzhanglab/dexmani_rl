from types import SimpleNamespace
import numpy as np
import pytest
import torch
from dexmani_rl.env import ChunkEnv
from dexmani_rl.policy.preprocess import prepare_observations
from dexmani_rl.policy.vision import ImageProcessor
from dexmani_rl.policy.fusion import project_to_image


class Sim:
    dt = 1/16
    def __init__(self, terminated, truncated):
        self.terminated, self.truncated, self.count = terminated, truncated, 0
    def reset(self, seed):
        self.count = 0
        return {'joint_state': np.zeros(19)}, {}
    def step(self, action):
        self.count += 1
        end = self.count == 3
        return ({'joint_state': np.full(19, self.count)}, 999., self.terminated and end,
                self.truncated and end, {'success': self.terminated and end})


@pytest.mark.parametrize('terminated,truncated', [(True, False), (False, True), (True, True)])
def test_chunk_stops_and_preserves_terminal_history(terminated, truncated):
    policy = SimpleNamespace(n_obs_steps=2, normalization_dims={'joint_state': 19},
                             parameters=lambda: iter([torch.empty(0)]))
    env = ChunkEnv(Sim(terminated, truncated), policy, {'control_dt_s': 1/16})
    initial = env.reset(1)['joint_state']
    assert initial.dtype == torch.float32
    torch.testing.assert_close(initial, torch.zeros(1, 2, 19))
    obs, reward, done, timeout, info = env.step(np.zeros((8, 19)))
    assert done == terminated and timeout == truncated
    assert info['steps'] == 3 and env.env.count == 3
    assert reward == float(terminated)
    torch.testing.assert_close(obs['joint_state'][0, :, 0], torch.tensor([2., 3.]))


def test_observation_preprocessing_preserves_rgb_and_point_coordinates():
    rgb = np.arange(2 * 3 * 4 * 3, dtype=np.uint8).reshape(2, 3, 4, 3)
    points = np.arange(2 * 4 * 6, dtype=np.float64).reshape(2, 4, 6)
    result = prepare_observations({'rgb': rgb, 'point_cloud': points},
        normalization_dims={'action': 19, 'rgb': 3, 'point_cloud': 3}, device='cpu')
    assert set(result) == {'rgb', 'point_cloud'}
    torch.testing.assert_close(result['rgb'], torch.from_numpy(rgb).permute(0, 3, 1, 2).unsqueeze(0))
    torch.testing.assert_close(result['point_cloud'], torch.from_numpy(points[..., :3]).float().unsqueeze(0))


@pytest.mark.parametrize('key,value,width,message', [
    ('joint_state', np.full((2, 19), np.nan), 19, 'Nonfinite observation'),
    ('camera_intrinsic', np.zeros((2, 3, 3)), 9, 'frame shape'),
    ('rgb', np.zeros((2, 3, 4, 3)), 3, 'dtype uint8'),
])
def test_observation_boundary_errors(key, value, width, message):
    with pytest.raises(ValueError, match=message):
        prepare_observations({key: value}, normalization_dims={key: width}, device='cpu')


@pytest.mark.parametrize('image_size', [None, (8, 12)])
def test_image_resize_preserves_calibrated_projection(image_size):
    raw = dict(rgb=np.zeros((1, 4, 6, 3), dtype=np.uint8),
               point_cloud=np.array([[[1., 2., 1.]]]),
               camera_intrinsic=np.eye(3).reshape(1, 9),
               camera_extrinsic=np.eye(3, 4).reshape(1, 12))
    dims = dict(rgb=3, point_cloud=3, camera_intrinsic=9, camera_extrinsic=12)
    obs = prepare_observations(raw, normalization_dims=dims, device='cpu')
    obs = {key: value.flatten(0, 1) for key, value in obs.items()}
    assert obs['rgb'].shape[-2:] == (4, 6)
    rgb, transform = ImageProcessor(image_size=image_size).process_images(obs['rgb'])
    projection = project_to_image(obs['point_cloud'], obs['camera_intrinsic'].unflatten(-1, (3, 3)),
        obs['camera_extrinsic'].unflatten(-1, (3, 4)), transform, rgb.shape[-2:])
    assert projection.valid.all()
    torch.testing.assert_close(projection.grid_xy, torch.tensor([[[-.5, .25]]]))
