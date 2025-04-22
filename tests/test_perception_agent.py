import numpy as np
from src.agents.perception_agent import PerceptionAgent

class DummyMemory:
    def get(self, key): return None
    def set(self, key, value): pass

class DummyFactory:
    def create(self, name): return None

def test_forward_and_backward_pass():
    config = {
        'modalities': ['vision', 'text', 'audio'],
        'embed_dim': 512,
        'projection_dim': 256
    }
    agent = PerceptionAgent(config, shared_memory=DummyMemory(), agent_factory=DummyFactory())

    batch = {
        'vision': np.random.randn(2, 3, 224, 224),
        'text': np.random.randint(0, 50257, (2, 77)),
        'audio': np.random.randn(2, 1, 16000)
    }

    output = agent.forward(batch)
    assert output.shape == (2, config['projection_dim']), f"Unexpected output shape: {output.shape}"

    # Dummy gradient
    dout = np.ones_like(output)
    agent.backward(dout)

    # Check at least one gradient is non-zero
    nonzero_grads = sum([np.any(p.grad != 0) for p in agent.params])
    assert nonzero_grads > 0, "No gradients were updated during backward pass."

    print("✅ Forward and backward test passed.")

if __name__ == "__main__":
    test_forward_and_backward_pass()
