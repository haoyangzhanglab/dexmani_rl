from dexmani_rl.core import ActionSegment, ActionSpec, PolicyOutput


class IndependentTestPolicy:
    action_spec = ActionSpec((ActionSegment("hand_latent", 2, "latent", "normalized"),))

    def predict(self, observation: float) -> PolicyOutput:
        return PolicyOutput(action=[observation, -observation])
