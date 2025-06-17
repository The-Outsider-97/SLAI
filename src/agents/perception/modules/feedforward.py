import torch
import math
import yaml

from src.agents.perception.utils.config_loader import load_global_config, get_config_section
from src.agents.perception.utils.common import TensorOps, Parameter
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("FeedForward")
printer = PrettyPrinter

class FeedForward(torch.nn.Module):
    """Enhanced Position-wise Feed-Forward Network with configurable components"""
    def __init__(self):
        super().__init__()
        self.config = load_global_config()
        self.embed_dim = self.config.get('embed_dim')
        self.initializer = self.config.get('initializer', 'xavier_uniform')
        self.device = self.config.get('device')
        self.dropout_rate = self.config.get('dropout_rate')
        self.training = self.config.get('training')
        self.ff_dim = self.config.get('ff_dim')
        self.activation = self.config.get('activation')
        self.norm_type = self.config.get('norm_type')

        self.ff_config = get_config_section('feedforward')
        self.use_bias = self.ff_config.get('use_bias')
        self.use_residual = self.ff_config.get('use_residual', True)
        self.fusion_type = self.ff_config.get('fusion_type')
        self.context_dim = self.ff_config.get('context_dim', None)

        # Initialize normalization layer
        self._init_normalization()

        # Initialize fusion parameters if needed
        self._init_fusion()

        # Initialize parameters
        self._init_parameters()

        # Intermediate values cache
        self._cache = {}
        
        # Activation derivatives mapping
        self._activation_derivatives = {
            'gelu': self._gelu_derivative,
            'relu': self._relu_derivative,
            'swish': self._swish_derivative
        }
        
        logger.info(f"FeedForward initialized: "
                    f"embed_dim={self.embed_dim}, ff_dim={self.ff_dim}, "
                    f"activation={self.activation}, use_bias={self.use_bias}, "
                    f"residual={self.use_residual}, norm={self.norm_type}, "
                    f"fusion={self.fusion_type}")

    def _init_normalization(self):
        """Initialize normalization layer based on config"""
        if self.norm_type == 'layernorm':
            self.norm = torch.nn.LayerNorm(self.embed_dim)
        elif self.norm_type == 'instancenorm':
            self.norm = torch.nn.InstanceNorm1d(self.embed_dim)
        elif self.norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"Unsupported normalization type: {self.norm_type}")

    def _init_fusion(self):
        """Initialize parameters for multi-modal fusion"""
        if self.fusion_type is None:
            return
        if self.fusion_type == 'concat':    # Increase first linear layer input size
            self.fused_embed_dim = self.embed_dim + self.context_dim
        elif self.fusion_type == 'film':    # FiLM parameters (Feature-wise Linear Modulation)
            self.film_gamma = Parameter(torch.ones(1, self.embed_dim, device=self.device))
            self.film_beta = Parameter(torch.zeros(1, self.embed_dim, device=self.device))
        elif self.fusion_type == 'add':
            # No additional parameters needed
            pass
        else:
            raise ValueError(f"Unsupported fusion type: {self.fusion_type}")

    def _init_parameters(self):
        """Initialize weights and biases with configurable scheme"""
        # Weight initialization mapping
        init_fns = {
            'he': TensorOps.he_init,
            'xavier': TensorOps.xavier_uniform,
            'lecun': TensorOps.lecun_normal
        }
        init_fn = init_fns.get(self.initializer, TensorOps.xavier_uniform)

        # Determine input dimension for first linear layer
        input_dim = self.embed_dim
        if self.fusion_type == 'concat':
            input_dim = self.fused_embed_dim

        # First linear transformation
        self.w1 = Parameter(
            init_fn((input_dim, self.ff_dim), 
            input_dim, 
            device=self.device
        ))
        self.b1 = Parameter(
            torch.zeros(self.ff_dim, device=self.device) 
            if self.use_bias else None
        )

        # Second linear transformation
        self.w2 = Parameter(
            init_fn((self.ff_dim, self.embed_dim), 
            self.ff_dim, 
            device=self.device
        ))
        self.b2 = Parameter(
            torch.zeros(self.embed_dim, device=self.device) 
            if self.use_bias else None
        )


    def forward(self, x, context=None):
        """
        Forward pass with optional normalization, residual connection, and multi-modal fusion
        
        Args:
            x: Input tensor of shape (batch, seq_len, embed_dim)
            context: Context tensor for multi-modal fusion
        """
        x = x.to(self.device)
        residual = x  # Save for residual connection
        
        # Apply normalization if configured
        if self.norm is not None:
            if isinstance(self.norm, torch.nn.InstanceNorm1d):
                # InstanceNorm requires (batch, channels, seq)
                x = x.permute(0, 2, 1)
                x = self.norm(x)
                x = x.permute(0, 2, 1)
            else:
                x = self.norm(x)
        
        # Apply multi-modal fusion
        if context is not None:
            context = context.to(self.device)
            x = self._apply_fusion(x, context)
        
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
            mask = (torch.rand_like(h_act) > self.dropout_rate).float()
            scale = 1.0 / (1.0 - self.dropout_rate)
            h_drop = h_act * mask * scale
            self._cache['dropout_mask'] = mask
        else:
            h_drop = h_act
            
        self._cache['dropped_act'] = h_drop.clone()
        
        # Second projection
        output = torch.matmul(h_drop, self.w2.data)
        if self.use_bias and self.b2 is not None:
            output += self.b2.data
        
        # Apply residual connection
        if self.use_residual:
            output = residual + output
            
        return output

    def backward(self, dout):
        dout = dout.to(self.device)
        original_shape = self._cache['x'].shape
        
        # Handle different input dimensions
        if dout.dim() == 2:  # [batch, embed_dim]
            effective_batch_dim = dout.shape[0]
            dout_3d = dout.unsqueeze(1)
        else:  # [batch, seq_len, embed_dim]
            effective_batch_dim = dout.shape[0] * dout.shape[1]
            dout_3d = dout
        
        # Reshape tensors for matrix operations
        h_drop_reshaped = self._cache['dropped_act'].reshape(effective_batch_dim, -1)
        dout_reshaped = dout_3d.reshape(effective_batch_dim, -1)
        x_reshaped = self._cache['x'].reshape(effective_batch_dim, -1)
        
        # Gradient for w2 and b2
        if self.w2.grad is None:
            self.w2.grad = torch.zeros_like(self.w2.data)
        self.w2.grad += torch.matmul(h_drop_reshaped.T, dout_reshaped)
        
        if self.use_bias and self.b2 is not None:
            if self.b2.grad is None:
                self.b2.grad = torch.zeros_like(self.b2.data)
            self.b2.grad += dout_reshaped.sum(dim=0)
        
        # Gradient for activation output
        d_h_drop = torch.matmul(dout_reshaped, self.w2.data.T)
        
        # Apply dropout mask if in training
        if self.training and self.dropout_rate > 0 and 'dropout_mask' in self._cache:
            dropout_mask = self._cache['dropout_mask'].reshape(d_h_drop.shape)
            d_h_drop = d_h_drop * dropout_mask
        
        # Gradient through activation
        if self.activation not in self._activation_derivatives:
            raise ValueError(f"Unsupported activation for backward: {self.activation}")
        d_h = self._activation_derivatives[self.activation](d_h_drop)
        
        # Gradient for w1 and b1
        if self.w1.grad is None:
            self.w1.grad = torch.zeros_like(self.w1.data)
        self.w1.grad += torch.matmul(x_reshaped.T, d_h)
        
        if self.use_bias and self.b1 is not None:
            if self.b1.grad is None:
                self.b1.grad = torch.zeros_like(self.b1.data)
            self.b1.grad += d_h.sum(dim=0)
        
        # Gradient for input
        d_x = torch.matmul(d_h, self.w1.data.T)
        return d_x.reshape(original_shape)

    def _apply_fusion(self, x, context):
        """Apply multi-modal fusion"""
        if self.fusion_type == 'add':
            # Simple addition
            return x + context
        
        elif self.fusion_type == 'concat':
            # Concatenate along feature dimension
            return torch.cat([x, context], dim=-1)
        
        elif self.fusion_type == 'film':
            # Feature-wise Linear Modulation
            gamma = self.film_gamma.data
            beta = self.film_beta.data
            
            # Handle different context dimensions
            if context.dim() == 2:  # Global context (batch, features)
                context = context.unsqueeze(1)  # Add sequence dimension
            
            # Compute modulation parameters from context
            if context.size(-1) != self.embed_dim:
                # Project context to match embedding dimension
                projection = torch.nn.Linear(context.size(-1), self.embed_dim, device=self.device)
                context = projection(context)
                
            gamma = gamma * context
            beta = beta * context
            
            return x * gamma + beta
        
        return x

    def _apply_activation(self, x):
        """Apply activation function with numerical stability"""
        if self.activation == 'gelu':
            return 0.5 * x * (1 + torch.tanh(
                math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)
            ))
        elif self.activation == 'relu':
            return torch.relu(x)
        elif self.activation == 'swish':
            return x * torch.sigmoid(x)
        else:
            raise ValueError(f"Unsupported activation: {self.activation}")

    def _gelu_derivative(self, d_act_input):
        """Compute gradient for GELU activation"""
        pre_act = self._cache['pre_act'].reshape(d_act_input.shape)
        sqrt_2_over_pi = math.sqrt(2 / math.pi)
        tanh_in = sqrt_2_over_pi * (pre_act + 0.044715 * pre_act ** 3)
        tanh_term = torch.tanh(tanh_in)
        
        derivative = 0.5 * (1 + tanh_term) + \
            0.5 * pre_act * (1 - tanh_term ** 2) * \
            sqrt_2_over_pi * (1 + 3 * 0.044715 * pre_act ** 2)
            
        return d_act_input * derivative

    def _relu_derivative(self, d_act_input):
        """Compute gradient for ReLU activation"""
        pre_act = self._cache['pre_act'].reshape(d_act_input.shape)
        return d_act_input * (pre_act > 0).float()

    def _swish_derivative(self, d_act_input):
        """Compute gradient for Swish activation"""
        pre_act = self._cache['pre_act'].reshape(d_act_input.shape)
        sig = torch.sigmoid(pre_act)
        return d_act_input * (sig + pre_act * sig * (1 - sig))

    def parameters(self):
        """Return all learnable parameters"""
        params = [self.w1, self.w2]
        if self.use_bias:
            if self.b1 is not None: 
                params.append(self.b1)
            if self.b2 is not None: 
                params.append(self.b2)
        return params

    def train(self, mode=True):
        """Set training mode"""
        self.training = mode
        return self

    def eval(self):
        """Set evaluation mode"""
        return self.train(False)

    def load_pretrained(self, weights, prefix=''):
        """Load pretrained weights from dictionary"""
        weight_key = f'{prefix}intermediate.dense.weight'
        bias_key = f'{prefix}intermediate.dense.bias'
        
        if weight_key in weights:
            self.w1.data = weights[weight_key].to(self.device)
        if bias_key in weights and self.use_bias and self.b1 is not None:
            self.b1.data = weights[bias_key].to(self.device)
        
        weight_key = f'{prefix}output.dense.weight'
        bias_key = f'{prefix}output.dense.bias'
        
        if weight_key in weights:
            self.w2.data = weights[weight_key].to(self.device)
        if bias_key in weights and self.use_bias and self.b2 is not None:
            self.b2.data = weights[bias_key].to(self.device)

if __name__ == "__main__":
    print("\n=== Running FeedForward ===\n")
    model = FeedForward()

    print("Initialized FeedForward module:")
    print(model)

    x = torch.randn(4, 128, 512)
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
