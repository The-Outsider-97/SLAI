import unittest
from evaluators.reward_function import RewardFunction

class DummyRollbackHandler:
    def __init__(self):
        self.triggered = False

    def rollback_model(self):
        self.triggered = True

class DummyHyperParamTuner:
    def __init__(self):
        self.triggered = False

    def run_tuning_pipeline(self):
        self.triggered = True

class TestRewardFunction(unittest.TestCase):
    def setUp(self):
        self.rollback_handler = DummyRollbackHandler()
        self.tuner = DummyHyperParamTuner()

        self.reward_func = RewardFunction(
            rollback_handler=self.rollback_handler,
            hyperparam_tuner=self.tuner,
            safety_thresholds={
                "negative_reward_limit": -50,
                "alignment_violation_limit": 2
            }
        )

    def test_no_violation(self):
        state = {}
        action = "recommend"
        outcome = {"reward": 80, "harm": False}
        reward = self.reward_func.compute_reward(state, action, outcome)
        self.assertGreater(reward, 0)
        self.assertFalse(self.rollback_handler.triggered)

    def test_rollback_trigger(self):
        state = {}
        action = "recommend"
        outcome = {"reward": -100, "harm": True}
        reward = self.reward_func.compute_reward(state, action, outcome)
        self.assertTrue(self.rollback_handler.triggered)

    def test_retrain_trigger(self):
        state = {}
        action = "recommend"
        outcome = {"reward": 10, "harm": True, "bias_detected": True, "discrimination_detected": True}
        self.reward_func.compute_reward(state, action, outcome)
        self.reward_func.compute_reward(state, action, outcome)
        self.assertTrue(self.tuner.triggered)

if __name__ == "__main__":
    unittest.main()
