import subprocess
import sys
import unittest


def check_backend_isolation() -> None:
    sys.modules["dexmani_rl.policies.flowmatch"] = None  # type: ignore[assignment]
    sys.modules["dexmani_rl.policy"] = None  # type: ignore[assignment]
    from dexmani_rl.core import ActionSegment, ActionSpec, PolicyAdapter, PolicyOutput
    from tests.fixtures.independent_backend import IndependentTestPolicy

    policy: PolicyAdapter = IndependentTestPolicy()
    assert policy.action_spec == ActionSpec((ActionSegment("hand_latent", 2, "latent", "normalized"),))
    assert policy.predict(0.25) == PolicyOutput(action=[0.25, -0.25])
    for name in ("dexmani_rl.policies.flowmatch", "dexmani_rl.policy"):
        assert sys.modules[name] is None
        assert not any(module.startswith(name + ".") for module in sys.modules)
    assert not {"torch", "torchvision", "transformers", "timm", "dexmani_sim"}.intersection(sys.modules)


class BackendIsolationTests(unittest.TestCase):
    def test_independent_backend_inference_without_flowmatch(self):
        subprocess.run([sys.executable, "-E", "-S", "-m", "tests.test_backend_isolation"], check=True)


if __name__ == "__main__":
    check_backend_isolation()
