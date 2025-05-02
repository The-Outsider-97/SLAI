"""
Perception Agent:
- Weight initialization
- Memory-efficient attention
- Batch processing
- Pretrained loading
- Gradient infrastructure
"""

import numpy as np
import math
from collections import OrderedDict

from src.agents.base_agent import BaseAgent
from src.agents.perception.utils.common import Parameter, TensorOps

class PerceptionAgent(BaseAgent):
    def __init__(self, config, shared_memory, agent_factory,
                 audio_encoder=None,
                 args=(),
                 kwargs={}):
        super().__init__(shared_memory=shared_memory, agent_factory=agent_factory)
        from src.agents.perception.encoders.vision_encoder import VisionEncoder
        from src.agents.perception.encoders.text_encoder import TextEncoder
        from src.agents.perception.encoders.audio_encoder import AudioEncoder

        self.modalities = config.get('modalities', ['text'])
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self.config = config
        self.pretrainer = self.PretrainingTasks(self)
        self.encoders = OrderedDict()
        embed_dim = config.get('embed_dim', 512)
        projection_dim = config.get('projection_dim', 256)

        
        if 'vision' in self.modalities:
            self.encoders['vision'] = VisionEncoder()
        if 'text' in self.modalities:
            self.encoders['text'] = TextEncoder()
        if 'audio' in self.modalities:
            self.encoders['audio'] = AudioEncoder()

        audio_weights = {}
        for layer in range(6):
            audio_weights.update({
                f'transformer_encoder_{layer}_attn_q_proj': np.random.randn(512, 512),
                f'transformer_encoder_{layer}_attn_k_proj': np.random.randn(512, 512),
                f'transformer_encoder_{layer}_attn_v_proj': np.random.randn(512, 512),
                f'transformer_encoder_{layer}_attn_out_proj': np.random.randn(512, 512),
                f'transformer_encoder_{layer}_ffn_w1': np.random.randn(512, 2048),
                f'transformer_encoder_{layer}_ffn_w2': np.random.randn(2048, 512),
                f'transformer_encoder_{layer}_norm1': np.ones(512),
                f'transformer_encoder_{layer}_norm2': np.ones(512),
            })
        audio_weights.update({
            'conv_proj': np.random.randn(512, 1, 400),
            'cls_token': np.random.randn(1, 1, 512),
            'pos_embed': np.random.randn(1, 41, 512),
        })
        converted = self.convert_audio_weights(audio_weights)
        prefixed = {f'transformer_{k}': v for k, v in converted.items()}
        self.load_pretrained('audio', prefixed)

        self.fusion = MultimodalFusion(embed_dim=embed_dim, modalities=self.modalities)
        self.projection = Parameter(
            TensorOps.he_init((embed_dim, projection_dim), embed_dim)
        )

        # Initialize gradients
        self.params = self._collect_parameters()

    def convert_audio_weights(self, weights):
        """Convert custom audio weights to HF-style for all layers"""
        new_w = {}
        for layer in range(6):
            for k, v in weights.items():
                if f'transformer_encoder_{layer}_attn_q_proj' in k: new_w[f'encoder.layer.{layer}.attention.self.query.weight'] = v
                elif f'transformer_encoder_{layer}_attn_k_proj' in k: new_w[f'encoder.layer.{layer}.attention.self.key.weight'] = v
                elif f'transformer_encoder_{layer}_attn_v_proj' in k: new_w[f'encoder.layer.{layer}.attention.self.value.weight'] = v
                elif f'transformer_encoder_{layer}_attn_out_proj' in k: new_w[f'encoder.layer.{layer}.attention.output.dense.weight'] = v
                elif f'transformer_encoder_{layer}_ffn_w1' in k: new_w[f'encoder.layer.{layer}.intermediate.dense.weight'] = v
                elif f'transformer_encoder_{layer}_ffn_w2' in k: new_w[f'encoder.layer.{layer}.output.dense.weight'] = v
                elif f'transformer_encoder_{layer}_norm1' in k: new_w[f'encoder.layer.{layer}.attention.output.LayerNorm.weight'] = v
                elif f'transformer_encoder_{layer}_norm2' in k: new_w[f'encoder.layer.{layer}.output.LayerNorm.weight'] = v
        return new_w

    def load_pretrained(self, modality, weights):
        if modality in self.encoders:
            self.encoders[modality].load_pretrained(weights)
        elif modality == 'multimodal':
            self._load_bert_style(weights)
        elif modality == 'clip':
            self._load_clip_style(weights)
        else:
            self.logger.warning(f"No encoder found for modality: {modality}")

    def _load_bert_style(self, weights):
        if 'vision' in self.encoders and 'visual' in weights:
            self.encoders['vision'].load_pretrained(weights['visual'])
    
        if 'text' in self.encoders and 'textual' in weights:
            self.encoders['text'].load_pretrained(weights['textual'])
    
        if 'audio' in self.encoders and 'audial' in weights:
            self.encoders['audio'].load_pretrained(weights['audial'])
    
        if 'fusion' in weights and hasattr(self.fusion, 'load_pretrained'):
            self.fusion.load_pretrained(weights['fusion'])
    
        if 'projection' in weights:
            self.projection.data = weights['projection']

    def _load_clip_style(self, weights):
        """Handle vision-language pretrained weights"""
        # CLIP-style loading
        self.encoders['vision'].load_pretrained(weights['visual'])
        self.encoders['text'].load_pretrained(weights['textual'])
        
        # Load fusion weights if available
        if 'fusion' in weights:
            self.fusion.load_pretrained(weights['fusion'])
        
        # Projection layer alignment
        if 'proj' in weights:
            self.projection.data = weights['proj']

    def _collect_parameters(self):
        params = []
        # Collect encoder parameters
        for encoder in self.encoders.values():
            params.extend(encoder.parameters())
        # Collect fusion parameters
        params.extend(self.fusion.parameters())
        # Collect projection parameter
        params.append(self.projection)
        return params

    def forward(self, inputs):
        embeddings = {}
        for modality, encoder in self.encoders.items():
            embeddings[modality] = encoder.forward(inputs[modality])
        fused = self.fusion.forward(embeddings)
        self._cache = {'fused': fused}
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

    def align_with_slailm(self, slailm_instance):
        """Create bidirectional gradient pathways between perception and LLM"""
        self.adapter = SLAILMAdapter(
            self.config['projection_dim'],
            slailm_instance.embed_dim
        )
        
        # Create shared parameter registry
        self.shared_params = {
            'cross_attn': Parameter(TensorOps.he_init((slailm_instance.embed_dim, self.config['projection_dim'])))
        }

    def zero_grad(self):
        for param in self.params:
            param.grad.fill(0)

    def step(self, learning_rate=1e-3):
        for param in self.params:
            param.data -= learning_rate * param.grad
        self.track_param_metrics()

    def save_params(self, path):
        weights = {f"param_{i}": param.data for i, param in enumerate(self.params)}
        np.savez(path, **weights)

    def load_params(self, path):
        weights = np.load(path)
        for i, param in enumerate(self.params):
            key = f"param_{i}"
            if key in weights:
                param.data[:] = weights[key]

    def track_param_metrics(self):
        norms = [np.linalg.norm(p.data) for p in self.params]
        grads = [np.linalg.norm(p.grad) for p in self.params]
        self.performance_metrics['param_norms'].append(float(np.mean(norms)))
        self.performance_metrics['grad_norms'].append(float(np.mean(grads)))

    class PretrainingTasks:
        """Multimodal pretraining objectives for perception agent"""
        def __init__(self, agent):
            self.agent = agent
            self.masking_ratio = 0.15  # Default masking ratio

        def masked_modality_modeling(self, inputs):
            """
            Joint masked reconstruction across multiple modalities
            Args:
                inputs: Dict of modality tensors {
                    'vision': (batch, channels, H, W),
                    'text': (batch, seq_len),
                    'audio': (batch, samples)
                }
            Returns:
                dict: Reconstruction losses per modality
            """
            losses = {}
            
            # Vision masking
            if 'vision' in inputs:
                masked_vision, mask_vision = self._apply_masking(inputs['vision'], mode='patch')
                rec_vision = self.agent.encoders['vision'].forward(masked_vision)
                losses['vision'] = self._calc_reconstruction_loss(rec_vision, inputs['vision'], mask_vision)

            # Text masking
            if 'text' in inputs:
                masked_text, mask_text = self._apply_masking(inputs['text'], mode='token')
                rec_text = self.agent.encoders['text'].forward(masked_text)
                losses['text'] = self._calc_reconstruction_loss(rec_text, inputs['text'], mask_text)

            # Audio masking
            if 'audio' in inputs:
                masked_audio, mask_audio = self._apply_masking(inputs['audio'], mode='frame')
                rec_audio = self.agent.encoders['audio'].forward(masked_audio)
                losses['audio'] = self._calc_reconstruction_loss(rec_audio, inputs['audio'], mask_audio)

            return losses

        def crossmodal_matching(self, embeddings):
            """
            Verify alignment between modality embeddings using contrastive learning
            Args:
                embeddings: Dict of modality embeddings {
                    'vision': (batch, embed_dim),
                    'text': (batch, embed_dim),
                    'audio': (batch, embed_dim)
                }
            Returns:
                dict: Matching accuracy and loss metrics
            """
            metrics = {}
            
            # Vision-text alignment
            if 'vision' in embeddings and 'text' in embeddings:
                metrics['vision_text'] = self._calc_crossmodal_similarity(
                    embeddings['vision'], 
                    embeddings['text']
                )

            # Audio-text alignment
            if 'audio' in embeddings and 'text' in embeddings:
                metrics['audio_text'] = self._calc_crossmodal_similarity(
                    embeddings['audio'],
                    embeddings['text']
                )

            return metrics

        def temporal_consistency(self, sequences):
            """
            Enforce temporal coherence in sequential data (video/audio)
            Args:
                sequences: Dict of temporal sequences {
                    'video': (batch, frames, H, W, C),
                    'audio': (batch, seq_len, features)
                }
            Returns:
                dict: Temporal coherence losses
            """
            losses = {}
            
            # Video temporal consistency
            if 'video' in sequences:
                frame_embeddings = [self.agent.encoders['vision'].forward(f) 
                                  for f in sequences['video']]
                losses['video'] = self._calc_temporal_loss(frame_embeddings)

            # Audio temporal consistency
            if 'audio' in sequences:
                audio_embeddings = [self.agent.encoders['audio'].forward(a) 
                                  for a in sequences['audio']]
                losses['audio'] = self._calc_temporal_loss(audio_embeddings)

            return losses

        # Helper methods placeholder
        def _apply_masking(self, data, mode):
            """Apply modality-specific masking pattern"""
            mask = np.random.rand(*data.shape) < self.masking_ratio
            masked = data.copy()
            masked[mask] = 0
            return masked, mask

        def _calc_reconstruction_loss(self, reconstructed, original, mask):
            """Calculate reconstruction loss with masking"""
            diff = (reconstructed - original) * mask
            return np.mean(diff ** 2)

        def _calc_crossmodal_similarity(self, emb1, emb2):
            """Compute contrastive similarity between modalities"""
            # Cosine similarity
            norm1 = np.linalg.norm(emb1, axis=1, keepdims=True)
            norm2 = np.linalg.norm(emb2, axis=1, keepdims=True)
            similarity = np.sum(emb1 * emb2, axis=1) / (norm1 * norm2 + 1e-8)
            loss = 1 - similarity.mean()
            return {"similarity": similarity.mean(), "contrastive_loss": loss}

        def _calc_temporal_loss(self, sequence_embeddings):
            """Calculate temporal coherence loss"""
            deltas = []
            for i in range(1, len(sequence_embeddings)):
                delta = np.mean((sequence_embeddings[i] - sequence_embeddings[i-1]) ** 2)
                deltas.append(delta)
            return np.mean(deltas)
        
        def update_projection(self, rewards, lr):
            if isinstance(rewards, list):
                rewards = np.array(rewards)
            grad = rewards.mean()  # Simplified reward-based scaling
            self.projection.grad += grad * np.sign(self.projection.data)
            self.projection.data -= lr * self.projection.grad

class CrossModalAttention:
    def __init__(self, embed_dim):
        self.query = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        self.key = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))
        self.value = Parameter(TensorOps.he_init((embed_dim, embed_dim), embed_dim))

    def parameters(self):
        return [self.query, self.key, self.value]

    def forward(self, modality1, modality2):
        q = np.matmul(modality1, self.query.data)
        k = np.matmul(modality2, self.key.data)
        v = np.matmul(modality2, self.value.data)
        
        attn_scores = np.matmul(q, k.T) / math.sqrt(self.query.data.shape[-1])
        attn_probs = self._softmax(attn_scores)
        return np.matmul(attn_probs, v)

    def _softmax(self, x):
        max_x = np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x - max_x)
        return exp_x / exp_x.sum(axis=-1, keepdims=True)

class MultimodalMetrics:
    @staticmethod
    def modality_importance_score(embeddings):
        """
        Compute L2 norm-based importance score per modality.
        Returns dict of normalized scores.
        """
        norms = {k: np.linalg.norm(v, axis=-1).mean() for k, v in embeddings.items()}
        total = sum(norms.values())
        return {k: v / total for k, v in norms.items()}

    @staticmethod
    def crossmodal_consistency(features):
        """
        Compute pairwise cosine similarity between all modality embeddings.
        Returns a matrix of pairwise consistency scores.
        """
        keys = list(features.keys())
        matrix = {}
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                a = features[keys[i]]
                b = features[keys[j]]
                sim = np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-8)
                matrix[f'{keys[i]}-{keys[j]}'] = np.mean(sim)
        return matrix

    @staticmethod
    def embedding_stability(embeddings):
        """
        Track change in embedding norms over batch.
        Could be used to detect saturation or vanishing dynamics.
        """
        stability = {}
        for k, v in embeddings.items():
            norm_diff = np.abs(np.diff(np.linalg.norm(v, axis=-1))).mean()
            stability[k] = norm_diff
        return stability

class MultimodalFusion:
    """Weighted feature fusion module using modality-specific importance with optional dropout and masking"""
    def __init__(self, embed_dim, modalities, dropout_rate=0.1, masking_prob=0.15):
        self.embed_dim = embed_dim
        self.modalities = modalities
        self.modal_weights = OrderedDict()
        self._shapes = {}
        self.cross_attn = CrossModalAttention(embed_dim)
        self.gating_weights = Parameter(np.ones(len(self.modalities)))  # Dynamic
        self.dropout_rate = dropout_rate
        self.masking_prob = masking_prob
        self.training = True

    def parameters(self):
        return list(self.modal_weights.values()) + [self.gating_weights] + self.cross_attn.parameters()

    def forward(self, embeddings):
        self._cached_inputs = embeddings
        self._mask_flags = {}
    
        pooled = []
        alphas = []
    
        # Dropout + Gating
        for idx, (modality, tensor) in enumerate(embeddings.items()):
            if modality not in self.modal_weights:
                self.modal_weights[modality] = Parameter(np.array([1.0]))
    
            # Drop modality
            if self.training and np.random.rand() < self.masking_prob:
                self._mask_flags[modality] = True
                tensor = np.zeros_like(tensor)
            else:
                self._mask_flags[modality] = False
    
            pooled_mod = tensor.mean(axis=1)  # (B, D)
    
            # Apply dropout
            if self.training and self.dropout_rate > 0:
                dropout_mask = (np.random.rand(*pooled_mod.shape) > self.dropout_rate).astype(np.float32)
                pooled_mod *= dropout_mask
    
            weight = self.modal_weights[modality].data
            pooled.append(weight * pooled_mod)
            alphas.append(weight)
    
        # Combine
        fused = sum(pooled) / max(len(pooled), 1)
        self._alphas = {mod: alpha for mod, alpha in zip(embeddings.keys(), alphas)}
        return fused

    def backward(self, d_fused):
        grads = {}
        for modality, input_tensor in self._cached_inputs.items():
            B, T, D = input_tensor.shape
            grad = np.repeat(d_fused[:, None, :], T, axis=1) * self.modal_weights[modality].data
            grads[modality] = grad

            if not self._mask_flags.get(modality, False):
                pooled_input = input_tensor.mean(axis=1)  # shape (B, D)
                d_alpha = np.sum(pooled_input * d_fused) / len(self._cached_inputs)
                self.modal_weights[modality].grad += d_alpha

        return grads

    def parameters(self):
        return list(self.modal_weights.values()) + [self.gating_weights]

class SLAILMAdapter:
    def __init__(self, perception_dim, llm_dim):
        self.projection = Parameter(TensorOps.he_init((perception_dim, llm_dim), perception_dim))
        self.layer_norm = Parameter(np.ones(llm_dim))
        self._cache = {}  # initially empty

    def parameters(self):
        return [self.projection, self.layer_norm]

    def set_cache(self, multimodal_emb, projected):
        self._cache['multimodal_emb'] = multimodal_emb
        self._cache['projected'] = projected

    def forward(self, multimodal_emb):
        projected = np.matmul(multimodal_emb, self.projection.data)
        self.set_cache(multimodal_emb, projected)  # ensure cache is populated
        return TensorOps.layer_norm(projected) * self.layer_norm.data
