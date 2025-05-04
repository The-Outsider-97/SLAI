import torch
import math

from src.agents.perception.utils.common import TensorOps, Parameter

class FeedForward:
    """Enhanced Position-wise Feed-Forward Network with configurable components"""
    def __init__(
        self,
        embed_dim=512,
        ff_dim=2048,
        activation='gelu',
        dropout_rate=0.1,
        use_bias=True,
        initializer='he',
        device='cpu',
        **kwargs
    ):
        self.embed_dim = embed_dim
        self.ff_dim = ff_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_bias = use_bias
        self.training = True
        self.device = device

        # Initialize parameters with configurable scheme
        init_fn = {
            'he': TensorOps.he_init,
            'xavier': TensorOps.xavier_uniform,
            'lecun': TensorOps.lecun_normal
        }.get(initializer, TensorOps.he_init)

        # First linear transformation
        self.w1 = Parameter(init_fn((embed_dim, ff_dim), embed_dim, device=self.device))
        self.b1 = Parameter(torch.zeros(ff_dim, device=self.device) if use_bias else None)

        # Second linear transformation
        self.w2 = Parameter(init_fn((ff_dim, embed_dim), ff_dim, device=self.device))
        self.b2 = Parameter(torch.zeros(embed_dim, device=self.device) if use_bias else None)

        # Intermediate values cache
        self._cache = {}
        self._activation_derivatives = {
            'gelu': self._gelu_derivative,
            'relu': self._relu_derivative,
            'swish': self._swish_derivative
        }

    def forward(self, x):
        """Forward pass with activation and dropout"""
        x = x.to(self.device)
        self._cache['x'] = x

        # First projection
        x = torch.matmul(x, self.w1.data)
        if self.use_bias:
            x += self.b1.data
        self._cache['pre_act'] = x

        # Activation function
        x = self._apply_activation(x)
        self._cache['act'] = x

        # Apply dropout
        if self.training and self.dropout_rate > 0:
            self._cache['dropout_mask'] = (torch.rand(*x.shape, device=self.device) > self.dropout_rate).to(torch.float32)
            x *= self._cache['dropout_mask']

        # Second projection
        x = torch.matmul(x, self.w2.data)
        if self.use_bias:
            x += self.b2.data

        return x

    def backward(self, dout):
        """Backpropagation with activation derivatives and dropout"""
        dout = dout.to(self.device)

        # Output projection gradients
        d_w2 = torch.matmul(self._cache['act'].T, dout)
        d_act = torch.matmul(dout, self.w2.data.T)
        if self.w2.grad is None: self.w2.grad = torch.zeros_like(self.w2.data)
        self.w2.grad += d_w2
        if self.use_bias:
            if self.b2.grad is None: self.b2.grad = torch.zeros_like(self.b2.data)
            self.b2.grad += dout.sum(dim=0)

        # Apply dropout mask gradient
        if 'dropout_mask' in self._cache:
            d_act *= self._cache['dropout_mask']
 
        # Activation derivative
        d_act = self._activation_derivatives[self.activation](d_act)

        # First projection gradients
        d_w1 = torch.matmul(self._cache['x'].T, d_act)
        d_x = torch.matmul(d_act, self.w1.data.T) 
        if self.w1.grad is None: self.w1.grad = torch.zeros_like(self.w1.data)
        self.w1.grad += d_w1
        if self.use_bias:
            if self.b1.grad is None: self.b1.grad = torch.zeros_like(self.b1.data)
            self.b1.grad += d_act.sum(dim=0)

        return d_x

    def _apply_activation(self, x):
        """Apply configured activation function"""
        if self.activation == 'gelu':

            act = TensorOps.gelu(x) # Or directly: act = torch.nn.functional.gelu(x)
           
        elif self.activation == 'relu':
            act = torch.relu(x)
        elif self.activation == 'swish':

            act = x * TensorOps.sigmoid(x) # Or directly: act = torch.nn.functional.silu(x)
           
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

        return act

    def _gelu_derivative(self, d_act):
        """GELU derivative implementation using PyTorch"""
        x = self._cache['pre_act']
        # Using approximation formula provided. PyTorch autograd handles this if using nn.GELU
        tanh_term = torch.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3))
        derivative = 0.5 * (1 + tanh_term) + \
            0.5 * x * (1 - tanh_term**2) * \
            math.sqrt(2/math.pi) * (1 + 3*0.044715*x**2)
        return d_act * derivative

    def _relu_derivative(self, d_act):
        """ReLU derivative implementation using PyTorch"""
        return d_act * (self._cache['pre_act'] > 0).to(d_act.dtype)

    def _swish_derivative(self, d_act):
        """Swish derivative implementation using PyTorch"""
        x = self._cache['pre_act']
        sigmoid = TensorOps.sigmoid(x) # Or directly: sigmoid = torch.sigmoid(x)
       
        return d_act * (sigmoid + x * sigmoid * (1 - sigmoid))

    def parameters(self):
        params = [self.w1, self.w2]
        if self.use_bias:
            params += [self.b1, self.b2]

        return params # Or potentially [p.data for p in params] if needed

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def load_pretrained(self, weights, prefix=''):
        """Handle loading weights (assuming weights are torch tensors)"""
        self.w1.data = weights.get(f'{prefix}intermediate.dense.weight', self.w1.data).to(self.device)
        if self.use_bias:
          self.b1.data = weights.get(f'{prefix}intermediate.dense.bias', self.b1.data).to(self.device)
        self.w2.data = weights.get(f'{prefix}output.dense.weight', self.w2.data).to(self.device)
        if self.use_bias:
          self.b2.data = weights.get(f'{prefix}output.dense.bias', self.b2.data).to(self.device)
