import numpy as np
import math

from src.agents.perception import TensorOps, Parameter

class FeedForward:
    """Position-wise Feed-Forward Network (FFN) from original transformer"""
    def __init__(self, embed_dim=512, ff_dim=2048):
        self.w1 = Parameter(TensorOps.he_init((embed_dim, ff_dim), embed_dim))
        self.b1 = Parameter(np.zeros(ff_dim))
        self.w2 = Parameter(TensorOps.he_init((ff_dim, embed_dim), ff_dim))
        self.b2 = Parameter(np.zeros(embed_dim))
        
    def forward(self, x):
        self._x = x
        x = np.matmul(x, self.w1.data) + self.b1.data
        x = TensorOps.gelu(x)
        self._act = x
        x = np.matmul(x, self.w2.data) + self.b2.data
        return x
        
    def backward(self, dout):
        """Backprop through FFN"""
        # Output projection
        d_w2 = np.matmul(self._act.T, dout)
        d_act = np.matmul(dout, self.w2.data.T)
        self.w2.grad += d_w2
        self.b2.grad += dout.sum(axis=0)
        
        # GELU derivative
        d_gelu = d_act * (
            0.5 * (1 + np.tanh(math.sqrt(2/math.pi) * 
                (self._act + 0.044715 * self._act**3)) 
            + 0.5 * self._act * 
            (1 - np.tanh(math.sqrt(2/math.pi) * 
                (self._act + 0.044715 * self._act**3))**2 
            * math.sqrt(2/math.pi) * 
            (1 + 3*0.044715*self._act**2)
        )))
            
        # Input projection
        d_w1 = np.matmul(self._x.T, d_gelu)
        d_x = np.matmul(d_gelu, self.w1.data.T)
        self.w1.grad += d_w1
        self.b1.grad += d_gelu.sum(axis=0)
        
        return d_x
