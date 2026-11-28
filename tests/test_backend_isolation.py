import subprocess
import sys
import unittest


class BackendIsolationTests(unittest.TestCase):
    def test_independent_backend_inference_without_flowmatch(self):
        code = """
import sys
sys.modules["dexmani_rl.policies.flowmatch"] = None
sys.modules["dexmani_rl.policy"] = None

from tests.fixtures.independent_backend import IndependentTestPolicy
from dexmani_rl.core import ActionSegment, ActionSpec, PolicyAdapter, PolicyOutput

policy: PolicyAdapter = IndependentTestPolicy()
assert policy.action_spec == ActionSpec((ActionSegment("hand_latent", 2, "latent", "normalized"),))
assert policy.predict(0.25) == PolicyOutput(action=[0.25, -0.25])

for backend in ("dexmani_rl.policies.flowmatch", "dexmani_rl.policy"):
    assert sys.modules[backend] is None
    assert not any(name.startswith(backend + ".") for name in sys.modules)
assert not {"torch", "torchvision", "transformers", "timm", "dexmani_sim"}.intersection(sys.modules)
"""
        subprocess.run([sys.executable, "-E", "-S", "-c", code], check=True)
