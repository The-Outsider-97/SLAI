import torch
import math
import yaml

from src.agents.perception.utils.common import TensorOps, Parameter
from logs.logger import get_logger

logger = get_logger("Feedforward")

CONFIG_PATH = "src/agents/perception/configs/perception_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def get_merged_config(user_config=None):
    base_config = load_config()
    if user_config:
        base_config.update(user_config)
    return base_config

class FeedForward(torch.nn.Module):
    """Enhanced Position-wise Feed-Forward Network with configurable components"""
    def __init__(self, config, device='cpu'):
        super().__init__()
        cfg = config['feedforward']
        transformer_cfg = config['transformer']
        self.embed_dim = transformer_cfg['embed_dim']
        self.ff_dim = transformer_cfg['ff_dim']
        self.activation = cfg['activation']
        self.dropout_rate = cfg['dropout_rate']
        self.use_bias = cfg['use_bias']
        self.initializer = cfg['initializer']
        self.training = True
        self.device = device

        # Initialize parameters with configurable scheme
        init_fn = {
            'he': TensorOps.he_init,
            'xavier': TensorOps.xavier_uniform,
            'lecun': TensorOps.lecun_normal
        }.get(self.initializer, TensorOps.he_init)

        # First linear transformation
        self.w1 = Parameter(init_fn((self.embed_dim, self.ff_dim), self.embed_dim, device=self.device))
        self.b1 = Parameter(torch.zeros(self.ff_dim, device=self.device) if self.use_bias else None)

        # Second linear transformation
        self.w2 = Parameter(init_fn((self.ff_dim, self.embed_dim), self.ff_dim, device=self.device))
        self.b2 = Parameter(torch.zeros(self.embed_dim, device=self.device) if self.use_bias else None)

        # Intermediate values cache
        self._cache = {}
        self._activation_derivatives = {
            'gelu': self._gelu_derivative,
            'relu': self._relu_derivative,
            'swish': self._swish_derivative
        }

        logger.info(f"Feedforward is successfully initialized with:\n- {torch.nn.Module}")

    def forward(self, x):
        x = x.to(self.device)
        self._cache['x'] = x

        # First projection
        h = torch.matmul(x, self.w1.data)
        if self.use_bias and self.b1 is not None:
            h += self.b1.data
        self._cache['pre_act'] = h.clone()

        # Activation function
        h_act = self._apply_activation(h)
        self._cache['act'] = h_act.clone()

        # Apply dropout
        if self.training and self.dropout_rate > 0:
            self._cache['dropout_mask'] = (torch.rand(*h_act.shape, device=self.device) > self.dropout_rate).to(torch.float32)
            h_drop = h_act * self._cache['dropout_mask']
        else:
            h_drop = h_act
        self._cache['dropped_act'] = h_drop.clone()

        # Second projection
        output = torch.matmul(h_drop, self.w2.data)
        if self.use_bias and self.b2 is not None:
            output += self.b2.data

        return output

    def backward(self, dout):
        dout = dout.to(self.device) # dout is dL/doutput

        # Handle reshaping based on input's original dimension to forward
        original_x_shape = self._cache['x'].shape
        if len(original_x_shape) == 2: # Input was [batch, embed_dim]
            is_2d_input = True

            dout_3d = dout.unsqueeze(1) if dout.dim() == 2 else dout
        else: # Input was [batch, seq_len, embed_dim]
            is_2d_input = False
            dout_3d = dout

        effective_batch_dim = dout_3d.shape[0] * dout_3d.shape[1] # B*S

        h_drop_reshaped = self._cache['dropped_act'].reshape(effective_batch_dim, self.ff_dim)
        dout_reshaped_for_w2 = dout_3d.reshape(effective_batch_dim, self.embed_dim)

        if self.w2.grad is None: self.w2.grad = torch.zeros_like(self.w2.data)
        self.w2.grad += torch.matmul(h_drop_reshaped.T, dout_reshaped_for_w2)

        if self.use_bias and self.b2 is not None:
            if self.b2.grad is None: self.b2.grad = torch.zeros_like(self.b2.data)
            self.b2.grad += dout_reshaped_for_w2.sum(dim=0)

        d_h_drop = torch.matmul(dout_reshaped_for_w2, self.w2.data.T) # Shape: [B*S, ff_dim]

        d_h_act = d_h_drop # Initialize
        if self.training and self.dropout_rate > 0 and 'dropout_mask' in self._cache:
            # dropout_mask was [B,S,F] or [B,F], needs to be [B*S, F]
            dropout_mask_reshaped = self._cache['dropout_mask'].reshape(effective_batch_dim, self.ff_dim)
            d_h_act = d_h_drop * dropout_mask_reshaped

        d_h = self._activation_derivatives[self.activation](d_h_act) # Shape: [B*S, ff_dim]
        
        # Gradient for self.w1 and self.b1
        x_reshaped = self._cache['x'].reshape(effective_batch_dim, self.embed_dim) # Original input to FF

        if self.w1.grad is None: self.w1.grad = torch.zeros_like(self.w1.data)
        self.w1.grad += torch.matmul(x_reshaped.T, d_h)
        
        if self.use_bias and self.b1 is not None:
            if self.b1.grad is None: self.b1.grad = torch.zeros_like(self.b1.data)
            self.b1.grad += d_h.sum(dim=0)

        d_x = torch.matmul(d_h, self.w1.data.T) # Shape [B*S, embed_dim]

        return d_x.reshape(original_x_shape)

    def _apply_activation(self, x):
        if self.activation == 'gelu':
            # Inline GELU to bypass broken TensorOps.gelu
            act = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * 
                            (x + 0.044715 * x ** 3)))
        elif self.activation == 'relu':
            act = torch.relu(x)
        elif self.activation == 'swish':
            act = x * TensorOps.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")
        return act

    def _gelu_derivative(self, d_act_input): # d_act_input is [B*S, F]
        pre_act_reshaped = self._cache['pre_act'].reshape(d_act_input.shape[0], self.ff_dim)
        x_flat = pre_act_reshaped
        tanh_term = torch.tanh(math.sqrt(2/math.pi) * (x_flat + 0.044715 * x_flat**3))
        derivative_factor = 0.5 * (1 + tanh_term) + \
            0.5 * x_flat * (1 - tanh_term**2) * \
            math.sqrt(2/math.pi) * (1 + 3*0.044715*x_flat**2)
        return d_act_input * derivative_factor

    def _relu_derivative(self, d_act_input): # d_act_input is [B*S, F]
        pre_act_reshaped = self._cache['pre_act'].reshape(d_act_input.shape[0], self.ff_dim)
        derivative_factor = (pre_act_reshaped > 0).to(d_act_input.dtype)
        return d_act_input * derivative_factor

    def _swish_derivative(self, d_act_input): # d_act_input is [B*S, F]
        pre_act_reshaped = self._cache['pre_act'].reshape(d_act_input.shape[0], self.ff_dim)
        x_flat = pre_act_reshaped
        sigmoid_x_flat = TensorOps.sigmoid(x_flat)
        derivative_factor = sigmoid_x_flat + x_flat * sigmoid_x_flat * (1 - sigmoid_x_flat)
        return d_act_input * derivative_factor

    def parameters(self):
        params = [self.w1, self.w2]
        if self.use_bias:
            if self.b1 is not None: params.append(self.b1)
            if self.b2 is not None: params.append(self.b2)
        return params

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def load_pretrained(self, weights, prefix=''):
        self.w1.data = weights.get(f'{prefix}intermediate.dense.weight', self.w1.data).to(self.device)
        if self.use_bias and self.b1 is not None:
          self.b1.data = weights.get(f'{prefix}intermediate.dense.bias', self.b1.data).to(self.device)
        self.w2.data = weights.get(f'{prefix}output.dense.weight', self.w2.data).to(self.device)
        if self.use_bias and self.b2 is not None:
          self.b2.data = weights.get(f'{prefix}output.dense.bias', self.b2.data).to(self.device)

if __name__ == "__main__":
    print("\n=== Running FeedForward ===\n")
    config = load_config()

    # Create a small feedforward model
    model = FeedForward(config)

    print("Initialized FeedForward module:")
    print(model)

    # Dummy input
    x = torch.randn(4, config['transformer']['embed_dim'])
    print(f"\nInput shape: {x.shape}")

    # Set to training mode
    model.train()
    print("\nMode: Training")
    y_train = model.forward(x)
    print("Forward output (train):\n", y_train)

    # Simulate dummy loss gradient
    dout = torch.ones_like(y_train)

    # Backward pass
    dx = model.backward(dout)
    print("Backward output (gradient w.r.t input):\n", dx)

    # Perform simple SGD step
    lr = 0.01
    for param in model.parameters():
        param.step(lr)
        param.zero_grad()

    # Switch to evaluation mode
    model.eval()
    print("\nMode: Evaluation")
    y_eval = model.forward(x)
    print("Forward output (eval):\n", y_eval)

    print("\n=== Successfully Ran FeedForward ===\n")
