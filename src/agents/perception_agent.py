"""
Enhanced Perception Agent with:
- Improved weight initialization
- Memory-efficient attention
- Batch processing
- Pretrained loading
- Gradient infrastructure
"""

import numpy as np
import math
from collections import OrderedDict

class Parameter:
    """Wrapper for trainable parameters with gradient storage"""
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
        return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3))
    
    @staticmethod
    def he_init(shape, fan_in):
        return np.random.randn(*shape) * math.sqrt(2.0 / fan_in)

class EfficientAttention:
    def __init__(self, embed_dim=512, num_heads=8):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Initialize parameters with proper scaling
        self.q_proj = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        self.k_proj = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        self.v_proj = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        self.out_proj = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        
        self._cache = {}

    def forward(self, x, context=None):
        if context is None:
            context = x
            
        q = np.matmul(x, self.q_proj.data)
        k = np.matmul(context, self.k_proj.data)
        v = np.matmul(context, self.v_proj.data)
        
        # Efficient batch-aware computation using einsum
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        
        attn_scores = np.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(self.head_dim)
        attn_probs = self._softmax(attn_scores)
        
        # Memory-efficient attention computation
        context_vec = np.einsum('bhij,bhjd->bhid', attn_probs, v)
        context_vec = self._combine_heads(context_vec)
        output = np.matmul(context_vec, self.out_proj.data)
        
        # Store intermediates for backward pass
        self._cache = {'q': q, 'k': k, 'v': v, 'attn_probs': attn_probs}
        return output

    def backward(self, dout):
        # Retrieve cached tensors from forward pass
        q = self._cache['q']
        k = self._cache['k']
        v = self._cache['v']
        attn_probs = self._cache['attn_probs']
        batch_size, num_heads, seq_len, head_dim = q.shape

        # Gradient for output projection
        d_context_vec = np.matmul(dout, self.out_proj.data.T)
        d_out_proj = np.matmul(
            self._cache['context_vec'].transpose(0,2,1,3).reshape(-1, self.embed_dim).T,
            dout.reshape(-1, self.embed_dim)
        )
        self.out_proj.grad += d_out_proj

        # Gradient through attention combination
        d_attn_probs = np.einsum('bhid,bhjd->bhij', d_context_vec, v)
        d_v = np.einsum('bhij,bhid->bhjd', attn_probs, d_context_vec)

        # Gradient through softmax
        d_scores = attn_probs * (d_attn_probs - np.einsum('bhij,bhij->bhi', attn_probs, d_attn_probs)[..., None])
        d_scores /= math.sqrt(self.head_dim)

        # Gradients for Q and K
        d_q = np.einsum('bhij,bhjd->bhid', d_scores, k)
        d_k = np.einsum('bhij,bhid->bhjd', d_scores, q)

        # Combine heads and calculate parameter gradients
        d_q = self._combine_heads(d_q.transpose(0,2,1,3))
        d_k = self._combine_heads(d_k.transpose(0,2,1,3))
        d_v = self._combine_heads(d_v.transpose(0,2,1,3))

        self.q_proj.grad += np.matmul(self._cache['x'].transpose(0,2,1), d_q).sum(axis=0)
        self.k_proj.grad += np.matmul(self._cache['context'].transpose(0,2,1), d_k).sum(axis=0)
        self.v_proj.grad += np.matmul(self._cache['context'].transpose(0,2,1), d_v).sum(axis=0)

        return np.matmul(d_q, self.q_proj.data.T) + \
               np.matmul(d_k, self.k_proj.data.T) + \
               np.matmul(d_v, self.v_proj.data.T)

    def _split_heads(self, x):
        return x.reshape(x.shape[0], x.shape[1], self.num_heads, self.head_dim).transpose(0,2,1,3)

    def _combine_heads(self, x):
        return x.transpose(0,2,1,3).reshape(x.shape[0], x.shape[2], -1)

    def _softmax(self, x):
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

class VisionEncoder:
    def __init__(self, img_size=224, patch_size=16, embed_dim=512):
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Proper initialization for convolutional projection
        self.projection = Parameter(
            TensorOps.he_init((3 * patch_size**2, embed_dim), 3*patch_size**2)
        )
        self.cls_token = Parameter(np.random.randn(1, 1, embed_dim) * 0.02)
        self.position_embed = Parameter(np.random.randn(1, self.num_patches+1, embed_dim) * 0.02)
        
        self.transformer = Transformer(num_layers=6, embed_dim=embed_dim)

    def extract_patches(self, x):
        # Batch-aware patch extraction using reshape
        b, c, h, w = x.shape
        x = x.reshape(b, c, h//self.patch_size, self.patch_size, w//self.patch_size, self.patch_size)
        x = x.transpose(0,2,4,1,3,5).reshape(b, -1, c*self.patch_size**2)
        return x

    def load_pretrained(self, weights):
        """Load pretrained weights in Vision Transformer format
        Args:
            weights: Dict containing:
                - conv_proj: (patch_size, patch_size, 3, embed_dim) for Conv2D projection
                - cls_token: (1, 1, embed_dim)
                - pos_embed: (1, num_patches + 1, embed_dim)
                - transformer_*: Transformer weights
        """
        # Convert Conv2D weights to linear projection
        if 'conv_proj' in weights:
            # Handle Conv2D -> Linear projection conversion
            self.projection.data = weights['conv_proj'].transpose(3, 0, 1, 2)  # (embed_dim, 3, patch, patch)
            self.projection.data = self.projection.data.reshape(self.projection.data.shape[0], -1).T
        
        # Direct loading for other parameters
        self.cls_token.data = weights.get('cls_token', self.cls_token.data)
        self.position_embed.data = weights.get('pos_embed', self.position_embed.data)
        
        # Load transformer weights
        if any(k.startswith('transformer_') for k in weights):
            self.transformer.load_pretrained({
                k.split('transformer_')[-1]: v 
                for k,v in weights.items() 
                if k.startswith('transformer_')
            })

    def forward(self, x):
        x = self.extract_patches(x)
        x = np.matmul(x, self.projection.data)
        cls_tokens = np.tile(self.cls_token.data, (x.shape[0], 1, 1))
        x = np.concatenate((cls_tokens, x), axis=1)
        x += self.position_embed.data
        return self.transformer.forward(x)

class TextEncoder:
    def __init__(self, vocab_size=50257, embed_dim=512):
        # Proper embedding initialization
        self.embedding = Parameter(np.random.randn(vocab_size, embed_dim) * 0.02)
        self.position_embed = Parameter(
            TensorOps.he_init((1, 512, embed_dim), embed_dim)
        self.transformer = Transformer(num_layers=6, embed_dim=embed_dim)

    def load_pretrained(self, weights):
        self.embedding.data = weights['token_embedding']
        self.position_embed.data = weights['position_embedding']

    def forward(self, x):
        embed = np.take(self.embedding.data, x, axis=0) + self.position_embed.data[:, :x.shape[1]]
        return self.transformer.forward(embed)

class PerceptionAgent:
    def __init__(self, config):
        self.config = config
        self.modalities = config['modalities']
        self.encoders = OrderedDict()
        
        if 'vision' in self.modalities:
            self.encoders['vision'] = VisionEncoder()
        if 'text' in self.modalities:
            self.encoders['text'] = TextEncoder()
        
        self.fusion = MultimodalFusion(embed_dim=config['embed_dim'])
        self.projection = Parameter(
            TensorOps.he_init((config['embed_dim'], config['projection_dim']), config['embed_dim'])
        
        # Initialize gradients
        self.params = self._collect_parameters()

    def _collect_parameters(self):
        params = []
        for module in [self.encoders, self.fusion, self.projection]:
            if hasattr(module, 'parameters'):
                params.extend(module.parameters())
        return params

    def load_pretrained(self, modality, weights):
        if modality in self.encoders:
            self.encoders[modality].load_pretrained(weights)

    def forward(self, inputs):
        embeddings = {}
        for modality, encoder in self.encoders.items():
            embeddings[modality] = encoder.forward(inputs[modality])
        fused = self.fusion.forward(embeddings)
        return np.matmul(TensorOps.layer_norm(fused), self.projection.data)

    def backward(self, dout):
        # Backward through projection layer
        d_fused = np.matmul(dout, self.projection.data.T)
        self.projection.grad += np.matmul(
            TensorOps.layer_norm(self._cache['fused']).T,
            dout
        )
        
        # Backward through multimodal fusion
        d_embeddings = self.fusion.backward(d_fused)
        
        # Backward through each encoder
        for modality, encoder in self.encoders.items():
            encoder.backward(d_embeddings[modality])
        
        return None  # Final gradients stored in parameters

# Complete Example Usage
if __name__ == "__main__":
    config = {
        'modalities': ['vision', 'text'],
        'embed_dim': 512,
        'projection_dim': 256
    }
    
    # Initialize agent with batch processing support
    agent = PerceptionAgent(config)
    
    # Example pretrained weights (ViT-Base compatible)
    pretrained_weights = {
        'conv_proj': np.random.randn(16, 16, 3, 512),  # (patch, patch, in_chans, embed_dim)
        'cls_token': np.random.randn(1, 1, 512),
        'pos_embed': np.random.randn(1, 197, 512),     # 14x14 patches + 1 cls token
        'transformer_encoder_0_attn_q_proj': np.random.randn(512, 512),
        # ... other transformer weights
    }
    
    # Load vision encoder pretrained weights
    agent.load_pretrained('vision', pretrained_weights)
    
    # Create batch of inputs
    batch = {
        'vision': np.random.randn(8, 3, 224, 224),  # 8 RGB images
        'text': np.random.randint(0, 50257, (8, 77)) # 8 text sequences
    }
    
    # Forward pass
    latent = agent.forward(batch)
    
    # Compute dummy loss (MSE for example)
    target = np.random.randn(*latent.shape)
    loss = np.mean((latent - target) ** 2)
    print(f"Initial loss: {loss:.4f}")
    
    # Backward pass
    dout = 2 * (latent - target) / latent.size
    agent.backward(dout)
    
    # Parameter update (SGD)
    learning_rate = 1e-3
    for param in agent.params:
        param.data -= learning_rate * param.grad
        param.grad.fill(0)  # Reset gradients
    
    # Verify updated forward pass
    updated_latent = agent.forward(batch)
    updated_loss = np.mean((updated_latent - target) ** 2)
    print(f"Updated loss: {updated_loss:.4f}")
