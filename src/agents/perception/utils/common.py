import numpy as np
import math

class Parameter:
    def __init__(self, data):
        self.data = data
        self.grad = np.zeros_like(data)

class TensorOps:
    @staticmethod
    def layer_norm(x, eps=1e-5):
        mean = x.mean(axis=-1, keepdims=True)
        std = x.std(axis=-1, keepdims=True)
        return (x - mean) / (std + eps)

    @staticmethod
    def gelu(x):
        return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

    @staticmethod
    def he_init(shape, fan_in):
        return np.random.randn(*shape) * math.sqrt(2.0 / fan_in)

    @staticmethod
    def xavier_init(shape, fan_in=None):
        if fan_in is None:
            fan_in = shape[0]
        limit = math.sqrt(6 / fan_in)
        return np.random.uniform(-limit, limit, size=shape).astype(np.float32)

