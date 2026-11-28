import subprocess
import sys
import unittest


class MethodIsolationTests(unittest.TestCase):
    def test_each_method_can_be_removed(self):
        script = """
import sys
import unittest
from importlib import import_module
from pathlib import Path
from omegaconf import OmegaConf

methods = ('dsrl', 'rfs', 'robofft', 'bora')
prefix = 'dexmani_rl.methods.'
blocked = sys.argv[1]
sys.modules[prefix + blocked] = None
for shared in ('cli', 'returns', 'flowmatch_online', 'torch_checkpoint'):
    import_module(prefix + '_shared.' + shared)
assert not any(sys.modules.get(prefix + method) for method in methods)
for method in methods:
    if method == blocked:
        continue
    train = import_module(prefix + method + '.train')
    import_module(prefix + method + '.algorithm')
    import_module(prefix + method + '.evaluate')
    args = train.training_parser().parse_args(['--checkpoint', 'bc.pt', '--output', 'unused'])
    cfg = train.training_inputs(train.training_parser(), args, method=method,
                                default_config=train.DEFAULT_CONFIG)[0]
    assert cfg == OmegaConf.to_container(OmegaConf.load(Path('configs/methods') / (method + '.yaml')))
from tests.test_method_updates import MethodUpdateTests
suite = unittest.TestSuite(MethodUpdateTests('test_' + name) for name in methods if name != blocked)
assert unittest.TextTestRunner(verbosity=0).run(suite).wasSuccessful()
assert not {'dexmani_sim', 'transformers', 'torchvision', 'timm'}.intersection(sys.modules)
"""
        for method in ('dsrl', 'rfs', 'robofft', 'bora'):
            with self.subTest(blocked=method):
                subprocess.run([sys.executable, '-c', script, method], check=True)
