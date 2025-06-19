__version__ = "1.9.0"

"""
Perception Agent:
- Initializes and manages multimodal encoders (text, vision, audio) and decoders.
- Implements various pretraining objectives:
    - Masked Language Modeling (MLM)
    - Masked Patch Modeling (MPM)
    - Masked Audio Modeling (MAM)
    - Cross-modal Contrastive Learning
    - Temporal Coherence Learning
- Supports loading of pretrained weights, including conversion for custom formats.
- Integrates with PerceptionMemory for caching and efficient computation.
- Designed for robustness, flexibility, and consistency with the existing agent architecture.
"""

from datetime import timedelta
import random
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple, Union

from src.agents.base.utils.main_config_loader import load_global_config, get_config_section
from src.agents.base_agent import BaseAgent
from src.agents.perception.utils.common import Parameter, TensorOps
from src.agents.perception.modules.transformer import Transformer
from src.agents.perception.encoders.text_encoder import TextEncoder
from src.agents.perception.encoders.vision_encoder import VisionEncoder
from src.agents.perception.encoders.audio_encoder import AudioEncoder
from src.agents.perception.decoders.text_decoder import TextDecoder
from src.agents.perception.decoders.vision_decoder import VisionDecoder
from src.agents.perception.decoders.audio_decoder import AudioDecoder
from src.agents.perception.modules.tokenizer import Tokenizer
from src.agents.perception.utils.taskheads import ClassificationHead, RegressionHead, Seq2SeqHead
from logs.logger import get_logger, PrettyPrinter

logger = get_logger("Perception Agent")
printer = PrettyPrinter

class PerceptionAgent(BaseAgent, nn.Module):
    def __init__(self, shared_memory, agent_factory, config=None):
        BaseAgent.__init__(self, shared_memory, agent_factory, config=config)
        nn.Module.__init__(self)

        self.perceptive_agent = []
        self.shared_memory = shared_memory
        self.agent_factory = agent_factory
        self._init_configs()
        self._init_components()
        self._init_shared_memory_keys()
        #self.optimizer = self._configure_optimizer()

        #self.global_projection_param = Parameter(torch.randn(1, device=self.device) * 0.01, name="global_projection_param")
        assert self.text_encoder.embed_dim == self.embed_dim, "Text encoder dim mismatch!"
        assert self.audio_encoder.embed_dim == self.embed_dim, "Audio encoder dim mismatch!"
        assert self.vision_encoder.embed_dim == self.embed_dim, "Vision encoder dim mismatch!"

        logger.info(f"PerceptionAgent initialized on device: {self.device}")

    def _init_configs(self):
        """Load global and perception-specific configurations."""
        self.config = load_global_config()
        self.perception_config = get_config_section('perception_agent')

        self.device = self.perception_config.get('device', 'cpu')
        self.embed_dim = self.perception_config.get('embed_dim', 512)
        self.masking_ratio = self.perception_config.get('masking_ratio', 0.15)
        self.encoder_type  = self.perception_config.get('encoder_type ', 'transformer')
        self.contrastive_temp = self.perception_config.get('contrastive_temperature', 0.07)
        
        # Optimizer related configs
        self.learning_rate = self.perception_config.get('learning_rate', 1e-4)
        self.weight_decay = self.perception_config.get('weight_decay', 1e-2)
        self.adam_betas = tuple(self.perception_config.get('adam_betas', [0.9, 0.999]))
        self.adam_eps = self.perception_config.get('adam_eps', 1e-8)
        self.training = self.perception_config.get('training', True)

        # Temporal Coherence
        self.loss_type = self.perception_config.get('loss_type')
        self.max_scale = self.perception_config.get('max_scale')
        self.temperature = self.perception_config.get('temperature')
        self.mse_weight = self.perception_config.get('mse_weight')
        self.contrastive_weight = self.perception_config.get('contrastive_weight')

    def _init_components(self):
        """Initialize encoders, decoders, tokenizer, memory, and projection heads."""
        self.tokenizer = Tokenizer() # Uses its own config loading

        # Modality Encoders
        self.text_encoder = TextEncoder().to(self.device)
        self.vision_encoder = VisionEncoder().to(self.device)
        self.audio_encoder = AudioEncoder().to(self.device)

        # Audio encoder with transformer configured to return hidden states
        self.audio_encoder = AudioEncoder().to(self.device)
        if hasattr(self.audio_encoder, 'transformer'):
            self.audio_encoder.transformer.return_hidden = True  # Force hidden state output

        # Full decoders for generation tasks would be more complex
        self.text_prediction_head = nn.Linear(self.embed_dim, self.tokenizer.get_vocab_size()).to(self.device)

        # For Vision, predict flattened patch values.
        # patch_dim depends on VisionEncoder's patch_size and in_channels
        vision_patch_dim = self.vision_encoder.in_channels * (self.vision_encoder.patch_size ** 2)
        self.vision_prediction_head = nn.Linear(self.embed_dim, vision_patch_dim).to(self.device)

        # For Audio, predict flattened patch values or features (e.g., MFCCs)
        # audio_patch_dim depends on AudioEncoder's patch_size and in_channels
        audio_patch_dim = self.audio_encoder.in_channels * (self.audio_encoder.patch_size ** 2) # If patch-based
        self.audio_prediction_head = nn.Linear(self.embed_dim, audio_patch_dim).to(self.device)

        # Full decoders for generation tasks (optional, can be loaded on demand)
        self.text_generator = TextDecoder(encoder=self.text_encoder).to(self.device)
        self.vision_generator = VisionDecoder().to(self.device) # Assumes VisionDecoder can init without encoder
        self.audio_generator = AudioDecoder().to(self.device)   # Assumes AudioDecoder can init without encoder

        self.global_projection_param = Parameter(torch.randn(self.embed_dim, self.embed_dim), requires_grad=True)

        # Projection heads for contrastive learning (to a common dimension)
        self.contrastive_projection_dim = self.perception_config.get('contrastive_projection_dim', 256)
        self.text_contrastive_proj = nn.Linear(self.embed_dim, self.contrastive_projection_dim).to(self.device)
        self.vision_contrastive_proj = nn.Linear(self.embed_dim, self.contrastive_projection_dim).to(self.device)
        self.audio_contrastive_proj = nn.Linear(self.embed_dim, self.contrastive_projection_dim).to(self.device)

        # Use each encoder's actual embed_dim for prediction heads
        self.text_prediction_head = nn.Linear(
            self.text_encoder.embed_dim,  # Use text encoder's dim
            self.tokenizer.get_vocab_size()
        ).to(self.device)

        vision_patch_dim = self.vision_encoder.in_channels * (self.vision_encoder.patch_size ** 2)
        self.vision_prediction_head = nn.Linear(
            self.vision_encoder.embed_dim,  # Use vision encoder's dim
            vision_patch_dim
        ).to(self.device)

        audio_patch_dim = self.audio_encoder.in_channels * self.audio_encoder.patch_size
        self.audio_prediction_head = nn.Linear(
            self.audio_encoder.embed_dim,  # Use audio encoder's dim
            audio_patch_dim
        ).to(self.device)

        # Initialize mask tokens for each modality
        self.mask_tokens = nn.ParameterDict({
            'text': Parameter(torch.zeros(1, self.embed_dim)),
            'vision': Parameter(torch.zeros(1, self.embed_dim)),
            'audio': Parameter(torch.zeros(1, self.embed_dim))
        })
        for key in self.mask_tokens:
            nn.init.normal_(self.mask_tokens[key], mean=0.0, std=0.02)

        # Reference position embeddings from encoders
        self.position_embeddings = {
            'text': self.text_encoder.position_embeddings,
            'vision': self.vision_encoder.position_embed,
            'audio': self.audio_encoder.position_embed
        }

        self.multi_modal_projector = nn.ModuleDict({
            'text': nn.Linear(self.text_encoder.embed_dim, self.embed_dim),
            'vision': nn.Linear(self.vision_encoder.embed_dim, self.embed_dim),
            'audio': nn.Linear(self.audio_encoder.embed_dim, self.embed_dim)
        }).to(self.device)

        # Move components to device
        self.mask_tokens = self.mask_tokens.to(self.device)

        # Optimizer
        self.optimizer = self._configure_optimizer()

    def _configure_optimizer(self):
        """Configures the optimizer for training."""
        params_to_optimize = list(self.text_encoder.parameters()) + \
                             list(self.vision_encoder.parameters()) + \
                             list(self.audio_encoder.parameters()) + \
                             list(self.text_prediction_head.parameters()) + \
                             list(self.vision_prediction_head.parameters()) + \
                             list(self.audio_prediction_head.parameters()) + \
                             list(self.text_contrastive_proj.parameters()) + \
                             list(self.vision_contrastive_proj.parameters()) + \
                             list(self.audio_contrastive_proj.parameters()) + \
                             list(self.text_generator.parameters()) + \
                             list(self.vision_generator.parameters()) + \
                             list(self.audio_generator.parameters()) + \
                             [self.global_projection_param]

        # Add parameters from task heads if they are dynamically created and stored
        # For example, if self.task_heads is a nn.ModuleDict:
        # if hasattr(self, 'task_heads'):
        #     params_to_optimize.extend(self.task_heads.parameters())

        return torch.optim.AdamW(
            filter(lambda p: p.requires_grad, params_to_optimize),
            lr=self.learning_rate,
            betas=self.adam_betas,
            eps=self.adam_eps,
            weight_decay=self.weight_decay
        )

    def _init_shared_memory_keys(self):
        """Initialize standardized keys for shared memory access"""
        self.sm_keys = {
            'weights_cache': f"perception:weights:{self.name}",
            'model_snapshot': f"perception:snapshot:{self.name}",
            'embeddings': f"perception:embeddings:{self.name}",
            'training_state': f"perception:training:{self.name}"
        }

    # --- Masking Helpers ---
    def _apply_masking_text(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies masking for Masked Language Modeling."""
        mask_token_id = self.tokenizer.token_to_id(self.tokenizer.mask_token)
        rand = torch.rand(input_ids.shape, device=self.device)
        # Where to mask (bernoulli trial for each token)
        mask_arr = (rand < self.masking_ratio)
        
        # Ensure we don't mask special tokens like [CLS], [SEP], [PAD]
        pad_token_id = self.tokenizer.token_to_id(self.tokenizer.pad_token)
        cls_token_id = self.tokenizer.token_to_id(self.tokenizer.cls_token)
        sep_token_id = self.tokenizer.token_to_id(self.tokenizer.sep_token)

        special_tokens_mask = (input_ids == pad_token_id) | \
                              (input_ids == cls_token_id) | \
                              (input_ids == sep_token_id)
        mask_arr &= ~special_tokens_mask # Don't mask special tokens

        masked_input_ids = input_ids.clone()
        masked_input_ids[mask_arr] = mask_token_id
        return masked_input_ids, mask_arr

    def _apply_masking_vision(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies masking for Masked Patch Modeling. Patches shape: (B, NumPatches, PatchDim)."""
        batch_size, num_patches, _ = patches.shape
        mask = torch.rand(
            batch_size,
            num_patches,
            device=self.device
        ) < self.masking_ratio
        masked_patches = patches.clone()
        masked_patches[mask] = 0
        return masked_patches, mask

    def _apply_masking_audio(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies masking for Masked Audio Modeling (e.g., on spectrogram patches or features)."""
        batch_size, num_frames, _ = features.shape
        mask = torch.rand(batch_size, num_frames, device=self.device) < self.masking_ratio
        masked_features = features.clone()
        masked_features[mask] = 0
        return masked_features, mask

    # --- Loss Calculation Helpers ---
    def _calc_reconstruction_loss_text(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculates MLM loss. Predictions: (B, SeqLen, VocabSize), Targets: (B, SeqLen), Mask: (B, SeqLen)"""
        predictions_masked = predictions[mask] # Select logits for masked tokens
        targets_masked = targets[mask]       # Select target token IDs for masked tokens
        if predictions_masked.numel() == 0: # No tokens were masked
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        loss = F.cross_entropy(predictions_masked.reshape(-1, predictions.size(-1)), targets_masked.reshape(-1))
        return loss

    def _calc_reconstruction_loss_vision_audio(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Calculates reconstruction loss for vision/audio. Predictions/Targets/Mask: (B, NumPatches/Frames, FeatureDim)"""
        if mask.sum() == 0 : # No elements masked
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        expanded_mask = mask.unsqueeze(-1).expand_as(targets)
        loss = F.mse_loss(predictions[expanded_mask], targets[expanded_mask])
        return loss
        
    def _calc_contrastive_loss(self, emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
        """Calculates InfoNCE contrastive loss. emb1, emb2: (B, ProjDim)"""
        emb1 = F.normalize(emb1, p=2, dim=-1)
        emb2 = F.normalize(emb2, p=2, dim=-1)

        # logits: (B, B)
        logits = torch.matmul(emb1, emb2.t()) / self.contrastive_temp
        labels = torch.arange(logits.size(0), device=self.device) # Positive pairs are on the diagonal
        loss = F.cross_entropy(logits, labels)
        return loss

    def _calc_temporal_coherence_loss(self, sequence_embeddings: torch.Tensor) -> torch.Tensor:
        """Calculates temporal coherence loss. sequence_embeddings: (B, SeqLen, EmbedDim)"""
        if sequence_embeddings.size(1) < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        # Difference between consecutive frame embeddings
        diffs = sequence_embeddings[:, 1:, :] - sequence_embeddings[:, :-1, :]
        loss = torch.mean(diffs.pow(2)) # MSE of differences
        return loss

    # --- Pretraining Steps ---
    def _pretrain_masked_modality(
        self,
        modality: str,
        inputs: Dict[str, torch.Tensor],
        mask_ratio: float = 0.15,
        return_reconstructions: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Performs masked modality pretraining (MLM, MPM, MAM) using modality-specific
        encoders and simple prediction heads.
        """
        valid_modalities = ['vision', 'audio', 'text']
        if modality not in valid_modalities:
            raise ValueError(f"Invalid modality '{modality}'. Must be one of {valid_modalities}")

        device = self.device
        training = self.training # Agent's training state

        # Get modality-specific components
        encoder = {
            'text': self.text_encoder,
            'vision': self.vision_encoder,
            'audio': self.audio_encoder
        }[modality]
    
        prediction_head = {
            'text': self.text_prediction_head,
            'vision': self.vision_prediction_head,
            'audio': self.audio_prediction_head
        }[modality]
    
        raw_input_key = self._get_input_key_for_modality(modality)
        raw_input = inputs[raw_input_key].to(device)
        style_id = inputs.get('style_id')
        if style_id is not None:
            style_id = style_id.to(device)
            
        reconstruction_loss = torch.tensor(0.0, device=device, requires_grad=training)
        reconstructions_dict = {}
    
        if modality in ["audio", "vision"] and encoder.encoder_type != "transformer":
            return 0.0

        with torch.set_grad_enabled(training):
            if modality == 'text':
                masked_input_ids, target_mask_indices = self._apply_masking_text(raw_input)
                encoder_output = encoder(masked_input_ids, style_id=style_id, output_type="full_sequence")
                predictions_logits = prediction_head(encoder_output) # (B, S, VocabSize)
 
                if target_mask_indices.sum() > 0:
                    # Select logits and labels for masked positions
                    masked_predictions_logits = predictions_logits[target_mask_indices] # (NumMasked, VocabSize)
                    masked_target_labels = raw_input[target_mask_indices] # (NumMasked)
                    
                    reconstruction_loss = F.cross_entropy(
                        masked_predictions_logits,
                        masked_target_labels
                    )
                
                if return_reconstructions:
                    reconstructions_dict = {
                        'original_ids': raw_input.detach(),
                        'masked_input_ids': masked_input_ids.detach(),
                        'reconstructed_logits': predictions_logits.detach(),
                        'mask_indices': target_mask_indices.detach()
                    }
    
            elif modality == 'vision' or modality == 'audio':
                # This implements a BERT-style masked patch/frame prediction.
                # 1. Extract raw patches/frames.
                # 2. Project them to initial embeddings.
                # 3. Mask some of these initial embeddings.
                # 4. Add CLS token and Positional Embeddings.
                # 5. Pass through the encoder's main transformer.
                # 6. Predict original raw patches/frames from the transformer's output at masked positions.
    
                if modality == 'vision':
                    # raw_input: (B, C, H, W)
                    raw_patches = encoder.extract_patches(raw_input) # (B, NumPatches, RawPatchDim)
                    initial_patch_embeddings = torch.matmul(raw_patches, encoder.projection) # (B, NumPatches, EmbedDim)
                    pos_embed_full = encoder.position_embed # (1, BaseNumPatches+1, EmbedDim) or (1, MaxPosEmbed, EmbedDim)
                    cls_token_emb = encoder.cls_token # (1, 1, EmbedDim)
                    mask_token_for_modality = self.mask_tokens['vision'] # (1, EmbedDim)
                else: # modality == 'audio'
                    # raw_input: (B, C, T_audio) or (B, T_audio)
                    raw_patches = encoder.extract_patches(raw_input) # (B, NumFrames, RawFrameDim)
                    initial_patch_embeddings = torch.matmul(raw_patches, encoder.projection) # (B, NumFrames, EmbedDim)
                    pos_embed_full = encoder.position_embed # (1, MaxPosEmbed, EmbedDim)
                    cls_token_emb = encoder.cls_token # (1, 1, EmbedDim) - Assuming audio encoder also has it
                    mask_token_for_modality = self.mask_tokens['audio'] # (1, EmbedDim)
    
                batch_size, num_patches_or_frames, _ = initial_patch_embeddings.shape
                
                # Create mask for actual patches/frames
                target_mask_indices = torch.rand(batch_size, num_patches_or_frames, device=device) < mask_ratio
                
                masked_initial_embeddings = initial_patch_embeddings.clone()
                # Efficiently apply mask token using torch.where or direct assignment
                # Ensure mask_token_for_modality is correctly broadcasted/expanded
                expanded_mask_fill = mask_token_for_modality.expand(batch_size, num_patches_or_frames, -1)
                masked_initial_embeddings = torch.where(target_mask_indices.unsqueeze(-1), 
                                                        expanded_mask_fill, 
                                                        masked_initial_embeddings)
    
                # Add CLS token
                cls_tokens_expanded = cls_token_emb.expand(batch_size, -1, -1)
                transformer_input_embeddings = torch.cat([cls_tokens_expanded, masked_initial_embeddings], dim=1) # (B, NumPatches+1, EmbedDim)
                
                current_seq_len = transformer_input_embeddings.size(1)
                
                # Add positional embeddings (handle potential size mismatch carefully)
                # Encoders should ideally handle dynamic PE sizing or use PE up to max_position_embeddings
                if pos_embed_full.size(1) < current_seq_len:
                    # Fallback: Use available part and warn. Proper fix is PE interpolation or larger PE table.
                    logger.warning(f"Positional embedding table size ({pos_embed_full.size(1)}) is smaller than "
                                   f"current sequence length ({current_seq_len}) for modality {modality}. Truncating/Padding PE.")
                    pe_to_add = torch.zeros_like(transformer_input_embeddings)
                    len_to_copy = min(pos_embed_full.size(1), current_seq_len)
                    pe_to_add[:, :len_to_copy, :] = pos_embed_full[:, :len_to_copy, :]
                else:
                    pe_to_add = pos_embed_full[:, :current_seq_len, :]
                
                transformer_input_with_pe = transformer_input_embeddings + pe_to_add
                
                # Pass through the encoder's transformer component
                encoder_transformer_output = encoder.transformer(transformer_input_with_pe, style_id=style_id) # (B, NumPatches+1, EmbedDim)
                
                if target_mask_indices.sum() > 0:
                    output_at_masked_positions = encoder_transformer_output[:, 1:, :][target_mask_indices] # (TotalNumMasked, EmbedDim)
                    
                    # Predict raw patches/frames from these outputs
                    predictions_raw = prediction_head(output_at_masked_positions) # (TotalNumMasked, RawPatchDim)
                    
                    # Targets are the original raw patches/frames at masked positions
                    target_raw_values = raw_patches[target_mask_indices] # (TotalNumMasked, RawPatchDim)
                    
                    reconstruction_loss = F.mse_loss(predictions_raw, target_raw_values)
                
                if return_reconstructions:
                    reconstructions_dict = {
                        'original_raw_patches_or_frames': raw_patches.detach(),
                        'masked_input_embeddings_to_transformer': masked_initial_embeddings.detach(), # These are EmbedDim
                        'reconstructed_raw_predictions_at_mask': predictions_raw.detach() if target_mask_indices.sum() > 0 else torch.empty(0),
                        'mask_indices': target_mask_indices.detach()
                    }
            
            # Ensure loss requires grad if in training and it turned out to be 0 (e.g. no tokens masked)
            if training and not reconstruction_loss.requires_grad and reconstruction_loss.item() == 0.0:
                reconstruction_loss = reconstruction_loss.clone().requires_grad_(True)
                
            if return_reconstructions:
                return reconstruction_loss, reconstructions_dict
            
            return reconstruction_loss

    def _pretrain_contrastive(self, data_mod1: Dict, data_mod2: Dict, mod1_type: str, mod2_type: str) -> torch.Tensor:
        """Performs cross-modal contrastive learning."""
        encoders = {'text': self.text_encoder, 'vision': self.vision_encoder, 'audio': self.audio_encoder}
        projections = {'text': self.text_contrastive_proj, 
                       'vision': self.vision_contrastive_proj, 
                       'audio': self.audio_contrastive_proj}

        # Extract input data for each modality
        input1 = data_mod1[self._get_input_key_for_modality(mod1_type)].to(self.device)
        input2 = data_mod2[self._get_input_key_for_modality(mod2_type)].to(self.device)

        style_id1 = data_mod1.get('style_id')
        style_id2 = data_mod2.get('style_id')

        # Encode features
        emb1_full = encoders[mod1_type](input1, style_id=style_id1)
        emb2_full = encoders[mod2_type](input2, style_id=style_id2)

        emb1 = emb1_full[:, 0, :] if emb1_full.ndim == 3 else emb1_full # (B, EmbedDim)
        emb2 = emb2_full[:, 0, :] if emb2_full.ndim == 3 else emb2_full # (B, EmbedDim)

        # Project to common contrastive space
        proj_emb1 = projections[mod1_type](emb1)
        proj_emb2 = projections[mod2_type](emb2)
        
        return self._calc_contrastive_loss(proj_emb1, proj_emb2)

    def _get_input_key_for_modality(self, modality_type: str) -> str:
        if modality_type == 'text': return 'input_ids'
        if modality_type == 'vision': return 'pixel_values'
        if modality_type == 'audio': return 'audio_values'
        raise ValueError(f"Unknown modality type: {modality_type}")

    def _pretrain_temporal_coherence(self, sequence_data: torch.Tensor, modality: str) -> torch.Tensor:
        """
        Enhanced temporal coherence learning with configurable loss types,
        multi-scale coherence, and efficient batched processing.
        
        Args:
            sequence_data: Input tensor (B, NumFrames, ...) 
            modality: 'vision' or 'audio'
        
        Returns:
            Temporal coherence loss tensor
        """
        encoders = {'vision': self.vision_encoder, 'audio': self.audio_encoder}
        if modality not in encoders:
            raise ValueError(f"Temporal coherence not supported for modality: {modality}")
        
        batch_size, num_frames = sequence_data.shape[:2]
        
        # Handle short sequences
        if num_frames < 2:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Efficient batched encoding - (B*T, ...) -> (B*T, D) -> (B, T, D)
        flat_sequence = sequence_data.view(-1, *sequence_data.shape[2:])
        encoded_frames = encoders[modality](flat_sequence)  # (B*T, ...)
        
        # Handle different encoder output types
        if encoded_frames.dim() == 3:  # Sequence output (B*T, S, D)
            frame_embeddings = encoded_frames[:, 0, :]  # Use CLS token
        else:  # Pooled output (B*T, D)
            frame_embeddings = encoded_frames
        
        embeddings = frame_embeddings.view(batch_size, num_frames, -1)  # (B, T, D)
        
        total_loss = torch.tensor(0.0, device=self.device)
        
        # Multi-scale MSE loss (captures both short and long-term coherence)
        if self.loss_type in ['mse', 'hybrid']:
            mse_loss = torch.tensor(0.0, device=self.device)
            valid_scale_count = 0
            
            for scale in range(1, min(self.max_scale + 1, num_frames)):
                if num_frames - scale < 1:
                    continue
                    
                # Calculate differences at current temporal scale
                emb1 = embeddings[:, :-scale, :]
                emb2 = embeddings[:, scale:, :]
                diffs = emb2 - emb1
                
                # Accumulate MSE loss
                scale_loss = torch.mean(diffs.pow(2))
                mse_loss += scale_loss
                valid_scale_count += 1
            
            if valid_scale_count > 0:
                mse_loss /= valid_scale_count
                total_loss += self.mse_weight * mse_loss
        
        # Contrastive loss (distinguishes adjacent vs distant frames)
        if self.loss_type in ['contrastive', 'hybrid']:
            # Only use consecutive pairs for anchor-positive
            anchors = embeddings[:, :-1, :].reshape(-1, embeddings.size(-1))  # (B*(T-1), D)
            positives = embeddings[:, 1:, :].reshape(-1, embeddings.size(-1))  # (B*(T-1), D)
            
            # Generate negatives - same sequence but distant frames
            all_negatives = []
            for i in range(batch_size):
                # Get random distant frames from same sequence
                time_indices = torch.randint(0, num_frames, (num_frames - 1,))
                distant_frames = embeddings[i, time_indices, :]
                all_negatives.append(distant_frames)
            
            negatives = torch.cat(all_negatives, dim=0)  # (B*(T-1), D)
            
            # Normalize embeddings
            anchors_norm = F.normalize(anchors, dim=-1)
            positives_norm = F.normalize(positives, dim=-1)
            negatives_norm = F.normalize(negatives, dim=-1)
            
            # Calculate similarities
            pos_sim = torch.sum(anchors_norm * positives_norm, dim=-1) / self.temperature
            neg_sim = torch.sum(anchors_norm[:, None] * negatives_norm[None, :], dim=-1) / self.temperature
            
            # Contrastive loss (InfoNCE)
            logits = torch.cat([pos_sim[:, None], neg_sim], dim=1)
            labels = torch.zeros(logits.size(0), dtype=torch.long, device=self.device)
            contrastive_loss = F.cross_entropy(logits, labels)
            
            total_loss += self.contrastive_weight * contrastive_loss
        
        return total_loss

    def _pretraining_step(self, task_data: Dict) -> Dict[str, torch.Tensor]:
        """Orchestrates a single pretraining step based on the objective."""
        if self.shared_memory.get(self.sm_keys['training_state']):
            logger.info("Training paused - another agent is currently training")
            return {'status': 'paused'}
            
        try:
            # Set training lock
            self.shared_memory.put(self.sm_keys['training_state'], True)

            self.train() # Set agent to training mode
            self.optimizer.zero_grad()
            
            objective = task_data['objective']
            total_loss = torch.tensor(0.0, device=self.device)

            if objective == 'mlm': # Masked Language Modeling
                loss = self._pretrain_masked_modality(task_data['text_data'], 'text')
                total_loss = total_loss + loss
            elif objective == 'mpm': # Masked Patch Modeling
                loss = self._pretrain_masked_modality(task_data['vision_data'], 'vision')
                total_loss = total_loss + loss
            elif objective == 'mam': # Masked Audio Modeling
                loss = self._pretrain_masked_modality(task_data['audio_data'], 'audio')
                total_loss = total_loss + loss
            elif objective == 'contrastive_text_image':
                loss = self._pretrain_contrastive(task_data['text_data'], task_data['vision_data'], 'text', 'vision')
                total_loss = total_loss + loss
            elif objective == 'contrastive_text_audio':
                loss = self._pretrain_contrastive(task_data['text_data'], task_data['audio_data'], 'text', 'audio')
                total_loss = total_loss + loss
            elif objective == 'contrastive_vision_audio':
                loss = self._pretrain_contrastive(task_data['vision_data'], task_data['audio_data'], 'vision', 'audio')
                total_loss = total_loss + loss
            elif objective == 'temporal_vision': # Temporal coherence for video
                loss = self._pretrain_temporal_coherence(task_data['video_data']['frame_sequence'], 'vision') # video_data.frame_sequence: (B, NumFrames, C, H, W)
                total_loss = total_loss + loss
            elif objective == 'temporal_audio': # Temporal coherence for audio
                loss = self._pretrain_temporal_coherence(task_data['audio_sequence_data']['segment_sequence'], 'audio') # audio_sequence_data.segment_sequence: (B, NumSegments, C, T_segment)
                total_loss = total_loss + loss
            else:
                logger.warning(f"Unknown pretraining objective: {objective}")
                return {'loss': total_loss, 'status': 'unknown_objective'}

            if total_loss.requires_grad: # Ensure loss requires grad before backward
                total_loss.backward()
                self.optimizer.step()
            
            return {'loss': total_loss.item(), 'status': 'success'}

        finally:
            # Release training lock
            self.shared_memory.put(self.sm_keys['training_state'], False)

    def train(self):
        """Set all model components to training mode."""
        for name in ['encoder', 'decoder', 'projection_layer']:
            module = getattr(self, name, None)
            if isinstance(module, torch.nn.Module):
                module.train()

    def _finetune_step(self, task_data: Dict) -> Dict[str, Any]:
        self.train()
        self.optimizer.zero_grad()

        downstream_task = task_data['downstream_task'] # e.g., 'image_classification', 'text_regression'
        labels = task_data['labels'].to(self.device)
        
        embeddings = None
        if 'text_data' in task_data:
            input_ids = task_data['text_data']['input_ids'].to(self.device)
            embeddings = self.text_encoder(input_ids, style_id=task_data['text_data'].get('style_id'))
        elif 'vision_data' in task_data:
            pixel_values = task_data['vision_data']['pixel_values'].to(self.device)
            embeddings = self.vision_encoder(pixel_values, style_id=task_data['vision_data'].get('style_id'))
        elif 'audio_data' in task_data:
            waveform = task_data['audio_data']['waveform'].to(self.device)
            embeddings = self.audio_encoder(waveform, style_id=task_data['audio_data'].get('style_id'))
        # Add handling for multimodal fine-tuning if necessary, by concatenating/fusing embeddings
        
        if embeddings is None:
            return {'loss': 0, 'status': 'no_input_data'}

        # Use CLS token or mean pooled output for sequence tasks
        if embeddings.ndim == 3: # (B, SeqLen, EmbedDim)
            pooled_embeddings = embeddings[:, 0, :] # Assuming CLS token
        else: # (B, EmbedDim)
            pooled_embeddings = embeddings

        # Get task head
        num_classes = task_data.get('num_classes') # Required for classification
        task_head = self.transformer.select_taskhead(
            downstream_task, 
            input_dim=self.embed_dim,
            num_classes=num_classes
        )
        
        logits_or_values = task_head(pooled_embeddings)

        loss = None
        if "classification" in downstream_task.lower():
            loss = F.cross_entropy(logits_or_values, labels)
        elif "regression" in downstream_task.lower():
            loss = F.mse_loss(logits_or_values.squeeze(), labels.float()) # Ensure labels are float for MSE
        else:
            return {'loss': 0, 'status': f'unsupported_task_type:{downstream_task}'}
        
        if loss is not None and loss.requires_grad:
            loss.backward()
            self.optimizer.step()
            return {'loss': loss.item(), 'status': 'success', 'predictions': logits_or_values.detach()}
        return {'loss': 0, 'status': 'loss_not_computed', 'predictions': logits_or_values.detach()}

    def _inference_step(self, task_data: Dict) -> Dict[str, Any]:
        self.eval() # Set agent to evaluation mode
        with torch.no_grad():
            modality = task_data['modality']
            input_data = task_data['input_data']
            downstream_task = task_data.get('downstream_task', None) # Optional: for task-specific heads

            embeddings = None
            style_id = input_data.get('style_id')

            if modality == 'text':
                input_ids = input_data['input_ids'].to(self.device)
                embeddings = self.text_encoder(input_ids, style_id=style_id)
            elif modality == 'vision':
                pixel_values = input_data['pixel_values'].to(self.device)
                embeddings = self.vision_encoder(pixel_values, style_id=style_id)
            elif modality == 'audio':
                waveform = input_data['waveform'].to(self.device)
                embeddings = self.audio_encoder(waveform, style_id=style_id)
            elif modality == 'multimodal': # Example for multimodal inference
                text_emb = self.text_encoder(input_data['text']['input_ids'].to(self.device), style_id=input_data['text'].get('style_id'))[:,0,:]
                vis_emb = self.vision_encoder(input_data['vision']['pixel_values'].to(self.device), style_id=input_data['vision'].get('style_id'))[:,0,:]
                # Simple concatenation for fusion, more sophisticated fusion can be added
                embeddings = torch.cat((text_emb, vis_emb), dim=-1) 
                # For multimodal, a specific multimodal head should be used.
                # Here, we assume a generic task head that can take the concatenated embeddings.
                # The input_dim for such a head would be 2 * self.embed_dim.
                # This part requires a more defined multimodal architecture.
            else:
                return {'output': None, 'status': f'unknown_modality:{modality}'}

            if downstream_task:
                if embeddings.ndim == 3: # (B, SeqLen, EmbedDim)
                    pooled_embeddings = embeddings[:, 0, :] # CLS token
                else: # (B, EmbedDim)
                    pooled_embeddings = embeddings
                
                # Adjust input_dim for multimodal concatenated embeddings
                current_embed_dim = embeddings.size(-1) if modality == 'multimodal' else self.embed_dim

                task_head = self._get_task_head(downstream_task, input_dim=current_embed_dim, num_classes=task_data.get('num_classes'))
                output = task_head(pooled_embeddings)
            else: # Return raw embeddings or pass through a generator
                if task_data.get('generate', False): # For generative tasks like text generation
                    if modality == 'text' and 'memory_for_decoder' in input_data: # Assuming memory comes from another encoder
                        memory = input_data['memory_for_decoder'].to(self.device)
                        output = self.text_generator.inference(memory=memory, style_id=style_id)
                    # Add similar generation for vision/audio if applicable
                    else:
                        output = embeddings # Fallback to embeddings if generation setup is incomplete
                else:
                    output = embeddings
            
            return {'output': output.cpu(), 'status': 'success'}

    # --- Main perform_task method ---
    def perform_task(self, task_data: Dict) -> Dict[str, Any]:
        """
        Main entry point for the PerceptionAgent.
        Dispatches to pretraining, fine-tuning, or inference based on task_data.
        """
        # Try loading state from shared memory before execution
        if task_data.get('use_cached_state', False):
            if self.load_state_from_shared_memory():
                logger.info("Using model state from shared memory")

            task_type = task_data.get('task_type')
            if task_type == 'pretrain':
                return self._pretraining_step(task_data)
            elif task_type == 'finetune':
                return self._finetune_step(task_data)
            elif task_type == 'inference':
                return self._inference_step(task_data)
            else:
                logger.error(f"Unsupported task_type: {task_type}")
                return {'status': 'error', 'message': f"Unsupported task_type: {task_type}"}

        # Save state after critical operations
        if task_data.get('save_state_after', False):
            self.save_state_to_shared_memory()
            
        return result

    def update_projection(self, rewards: Union[List[float], torch.Tensor], lr: float):
        """
        Updates the global_projection_param using a custom rule.
        This is separate from the main optimizer.
        """
        if not isinstance(rewards, torch.Tensor):
            rewards = torch.tensor(rewards, device=self.device, dtype=torch.float3T)
        
        # Ensure global_projection_param requires grad for this update logic
        if not self.global_projection_param.requires_grad:
            self.global_projection_param.requires_grad_(True)

        # Simplified reward-based scaling for the gradient
        # The gradient here is a pseudo-gradient based on rewards
        pseudo_grad = rewards.mean() * torch.sign(self.global_projection_param)
        
        if self.global_projection_param.grad is None:
            self.global_projection_param.grad = pseudo_grad
        else:
            self.global_projection_param.grad.data.add_(pseudo_grad) # Accumulate pseudo-gradient

        # Apply update (manual SGD step for this parameter)
        with torch.no_grad():
            self.global_projection_param.sub_(lr * self.global_projection_param.grad.data)
        
        # Zero out the pseudo-gradient after update
        if self.global_projection_param.grad is not None:
            self.global_projection_param.grad.detach_() # Detach from computation graph
            self.global_projection_param.grad.zero_()
        
        logger.debug(f"Updated global_projection_param: {self.global_projection_param.item()}")

    def load_pretrained_weights(self, checkpoint_path: Union[str, Path], source_format: str = "custom_audio"):
        """Loads pretrained weights, handling different source formats."""
        logger.info(f"Loading pretrained weights from: {checkpoint_path} (format: {source_format})")

        # First check shared memory cache
        cache_key = f"{self.sm_keys['weights_cache']}:{source_format}:{checkpoint_path}"
        cached_weights = self.shared_memory.get(cache_key)
    
        path = Path(checkpoint_path)
        if not path.exists():
            logger.warning(f"Checkpoint file not found: {checkpoint_path}. Creating new checkpoint.")
            checkpoint_dir = path.parent
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            weights_data = {
                'model_state_dict': {
                    'text_encoder': self.text_encoder.state_dict(),
                    'vision_encoder': self.vision_encoder.state_dict(),
                    'audio_encoder': self.audio_encoder.state_dict(),
                    #'transformer': self.transformer.state_dict(),
                    'text_decoder': self.text_generator.state_dict(),
                    'vision_decoder': self.vision_generator.state_dict(),
                    'audio_decoder': self.audio_generator.state_dict(),
                },
                'optimizer_state_dict': self.optimizer.state_dict() if hasattr(self, 'optimizer') else {}
            }
            torch.save(weights_data, path)
            logger.info(f"New checkpoint created at: {checkpoint_path}")
            return weights_data
        if path.is_dir():
            logger.error(f"Checkpoint path is a directory, not a file: {checkpoint_path}")
            return

        if not Path(checkpoint_path).exists():
            logger.error(f"Checkpoint file not found: {checkpoint_path}")
            return
        if cached_weights:
            logger.info(f"Using cached weights from shared memory: {cache_key}")
            weights_data = cached_weights
        else:
            # Load weights if not cached
            weights_data = torch.load(checkpoint_path, map_location=self.device)
            # Store loaded weights in shared memory
            self.shared_memory.put(cache_key, weights_data, ttl=timedelta(days=7))
            logger.info(f"Cached weights in shared memory: {cache_key}")
        
        if source_format == "custom_audio":
            converted_weights = self._convert_custom_audio_weights(weights_data)
            self.audio_encoder.load_pretrained(converted_weights)
            # Potentially load for audio_generator too if structure is similar
            # self.audio_generator.load_pretrained(weights)
        elif source_format == "vision_language_model":
            vision_weights, text_weights = self._convert_vision_language_weights(weights_data)
            self.vision_encoder.load_pretrained(vision_weights)
            self.text_encoder.load_pretrained_embeddings(text_weights.get('token_embeddings.weight')) # Example
            # self.text_encoder.transformer.load_pretrained(weights) if text transformer weights are separate
        elif source_format == "text_encoder_only":
            self.text_encoder.load_pretrained_embeddings(weights_data.get('token_embeddings.weight'))
            # Potentially load transformer part of text_encoder
            # self.text_encoder.transformer.load_pretrained(weights_data.get('transformer_weights', {}))
        elif source_format == "perception_agent_checkpoint": # Loading a full agent checkpoint
            self.load_state_dict(weights_data['model_state_dict'])
            self.optimizer.load_state_dict(weights_data['optimizer_state_dict'])
            logger.info("Loaded full PerceptionAgent checkpoint.")
        else:
            logger.warning(f"Unsupported pretrained weight format: {source_format}")

    def _convert_custom_audio_weights(self, custom_weights: Dict) -> Dict:
        """Converts custom audio model weights to a format expected by AudioEncoder/Transformer.
        
        Handles common architectures:
        - Wav2Vec 2.0
        - HuBERT
        - Speech2Vec
        - Custom CNN-RNN hybrids
        
        Conversion strategies:
        1. Direct key mapping for standard architectures
        2. Tensor reshaping for dimension mismatches
        3. Layer skipping for incompatible components
        """
        hf_style_weights = {}
        config = self.perception_config.get('weight_conversion', {})
        skip_mismatched = config.get('skip_mismatched', True)
        reshape_mode = config.get('reshape_mode', 'auto')
        
        # Architecture detection
        arch = None
        if any(k.startswith('w2v2.') for k in custom_weights):
            arch = 'wav2vec2'
        elif any(k.startswith('hubert.') for k in custom_weights):
            arch = 'hubert'
        elif any('cnn' in k and 'rnn' in k for k in custom_weights):
            arch = 'cnn_rnn'

        # Mapping tables for known architectures
        mapping_tables = {
            'wav2vec2': {
                'w2v2.encoder.pos_conv.0.weight': 'position_embed',
                'w2v2.feature_extractor.conv_layers.0.conv.weight': 'conv_layers.0.weight',
                'w2v2.post_extract_proj.weight': 'projection.weight',
                'w2v2.mask_emb': 'mask_tokens.audio',
                'w2v2.encoder.layers.{}.self_attn.k_proj.weight': 'transformer.layers.{}.attention.k_proj.weight',
                'w2v2.encoder.layers.{}.self_attn.out_proj.weight': 'transformer.layers.{}.attention.out_proj.weight',
                'w2v2.encoder.layers.{}.fc1.weight': 'transformer.layers.{}.ff.0.weight'
            },
            'hubert': {
                'hubert.encoder.pos_conv.0.weight': 'position_embed',
                'hubert.feature_projection.projection.weight': 'projection.weight',
                'hubert.mask_emb': 'mask_tokens.audio',
                'hubert.encoder.layers.{}.self_attn.k_proj.weight': 'transformer.layers.{}.attention.k_proj.weight'
            },
            'cnn_rnn': {
                'cnn.conv1.weight': 'conv_layers.0.weight',
                'rnn.rnn.weight_ih_l0': 'recurrent_layer.weight_ih',
                'rnn.rnn.weight_hh_l0': 'recurrent_layer.weight_hh',
                'projection_layer.weight': 'projection.weight'
            }
        }
        
        # Handle unknown architectures with pattern matching
        if not arch:
            logger.warning("Custom audio architecture not recognized - using heuristic mapping")
            mapping_tables['unknown'] = {
                r'conv(\d+)\.weight': 'conv_layers.\\1.weight',
                r'pos_?emb': 'position_embed',
                r'proj': 'projection',
                r'transformer\.layer_(\d+)\.attention': 'transformer.layers.\\1.attention',
                r'mask_?token': 'mask_tokens.audio'
            }
            arch = 'unknown'
        
        # Conversion process
        for custom_name, tensor in custom_weights.items():
            matched = False
            target_name = None
            
            # Try known mappings first
            for pattern, target_pattern in mapping_tables[arch].items():
                if arch in ['wav2vec2', 'hubert']:
                    # Handle layer-indexed patterns
                    if '{}' in pattern:
                        for layer_idx in range(self.audio_encoder.transformer.num_layers):
                            if pattern.format(layer_idx) in custom_name:
                                target_name = target_pattern.format(layer_idx)
                                matched = True
                                break
                    elif pattern in custom_name:
                        target_name = target_pattern
                        matched = True
                elif arch == 'cnn_rnn':
                    # Direct mapping
                    if pattern in custom_name:
                        target_name = target_pattern
                        matched = True
                else:  # Heuristic matching
                    import re
                    match = re.match(pattern, custom_name)
                    if match:
                        target_name = re.sub(pattern, target_pattern, custom_name)
                        matched = True
            
            # Skip unmapped parameters
            if not matched:
                if config.get('log_skipped_weights', False):
                    logger.debug(f"Skipping unmapped audio weight: {custom_name}")
                continue
            
            # Check tensor compatibility
            current_shape = tensor.shape
            try:
                target_param = self.audio_encoder.get_parameter(target_name)
                target_shape = target_param.shape
            except AttributeError:
                logger.warning(f"Target parameter {target_name} not found in model")
                continue
            
            # Reshape tensors if needed
            if current_shape != target_shape:
                if reshape_mode == 'skip':
                    logger.info(f"Skipping {custom_name} due to shape mismatch "
                               f"({current_shape} vs {target_shape})")
                    continue
                    
                tensor = self._reshape_tensor(tensor, target_shape, 
                                             mode=reshape_mode,
                                             custom_name=custom_name,
                                             target_name=target_name)
            
            hf_style_weights[target_name] = tensor
        
        logger.info(f"Converted {len(hf_style_weights)}/{len(custom_weights)} "
                   f"audio weights ({arch} format)")
        return hf_style_weights
    
    def _reshape_tensor(self, tensor: torch.Tensor, target_shape: Tuple[int], 
                       mode: str = 'auto', **kwargs) -> torch.Tensor:
        """Reshapes tensors using various strategies"""
        if mode == 'auto':
            if tensor.dim() == 2 and target_shape[0] == target_shape[1]:    # Automatic reshaping heuristics
                if tensor.shape[0] == target_shape[1] and tensor.shape[1] == target_shape[0]:    # Handle square matrix transpose
                    return tensor.t()
            elif tensor.numel() == math.prod(target_shape):
                return tensor.view(target_shape)
        
        elif mode == 'pad':
            # Zero-padding strategy
            new_tensor = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
            slices = tuple(slice(0, min(dim, t_dim)) for dim, t_dim in zip(tensor.shape, target_shape))
            new_tensor[slices] = tensor[slices]
            return new_tensor
        
        elif mode == 'crop':
            # Cropping strategy
            slices = tuple(slice(0, min(dim, t_dim)) for dim, t_dim in zip(target_shape, tensor.shape))
            return tensor[slices].clone()
        
        logger.warning(f"Could not reshape {kwargs.get('custom_name')} from "
                      f"{tensor.shape} to {target_shape} using {mode} mode")
        return tensor

    def _convert_vision_language_weights(self, custom_weights: Dict) -> Tuple[Dict, Dict]:
        """Separates and converts vision-language weights for VisionEncoder and TextEncoder.
        
        Supports models:
        - CLIP
        - ALIGN
        - FLAVA
        - ALBEF
        - Custom dual-encoder architectures
        
        Handles:
        - Component separation (vision/text)
        - Cross-attention redistribution
        - Modality-specific projection layers
        """
        vision_weights = {}
        text_weights = {}
        config = self.perception_config.get('weight_conversion', {})
        fusion_handling = config.get('fusion_handling', 'distribute')
        
        # Architecture detection
        arch = None
        if any(k.startswith('visual.') or k.startswith('text.') for k in custom_weights):
            arch = 'clip_style'
        elif any('image_encoder' in k and 'text_encoder' in k for k in custom_weights):
            arch = 'dual_encoder'
        elif any('cross_attention' in k for k in custom_weights):
            arch = 'fusion_model'
        
        # Key classification patterns
        vision_patterns = [
            'visual.', 'image_encoder.', 'vision.', 
            'conv', 'resblocks', 'patch_embed', 'pos_embed',
            'img_', 'spatial.', 'pixel_'
        ]
        text_patterns = [
            'text.', 'token_embed', 'positional_embedding',
            'transformer.text.', 'word_embed', 'txt_',
            'language_encoder', 'bert.'
        ]
        fusion_patterns = [
            'cross_attention', 'fusion', 'multihead_attn',
            'modality_combine', 'concat'
        ]
        
        # Special handling for known architectures
        if arch == 'clip_style':
            # CLIP-style explicit naming
            for k, v in custom_weights.items():
                if k.startswith('visual.'):
                    vision_weights[k.replace('visual.', '')] = v
                elif k.startswith('text.'):
                    text_weights[k.replace('text.', '')] = v
                elif 'positional_embedding' in k:
                    if 'visual.positional_embedding' in k:
                        vision_weights['position_embed'] = v
                    else:
                        text_weights['position_embeddings'] = v
        
        else:
            # Generic heuristic-based separation
            for custom_name, tensor in custom_weights.items():
                # Classify as vision, text, or fusion
                is_vision = any(p in custom_name for p in vision_patterns)
                is_text = any(p in custom_name for p in text_patterns)
                is_fusion = any(p in custom_name for p in fusion_patterns)
                
                # Fusion component handling
                if is_fusion:
                    if fusion_handling == 'distribute':
                        # Distribute fusion components proportionally
                        if is_vision or ('image' in custom_name.lower()):
                            vision_weights[custom_name] = tensor
                        elif is_text or ('text' in custom_name.lower()):
                            text_weights[custom_name] = tensor
                        else:
                            # Split evenly between modalities
                            vision_weights[custom_name] = tensor[:tensor.shape[0]//2]
                            text_weights[custom_name] = tensor[tensor.shape[0]//2:]
                    elif fusion_handling == 'discard':
                        logger.info(f"Discarding fusion weight: {custom_name}")
                        continue
                    elif fusion_handling == 'vision':
                        vision_weights[custom_name] = tensor
                    elif fusion_handling == 'text':
                        text_weights[custom_name] = tensor
                
                # Direct classification
                elif is_vision:
                    vision_weights[custom_name] = tensor
                elif is_text:
                    text_weights[custom_name] = tensor
                else:
                    logger.warning(f"Unclassified weight: {custom_name} - assigning to both")
                    vision_weights[custom_name] = tensor
                    text_weights[custom_name] = tensor.clone()
        
        # Post-processing for each modality
        vision_weights = self._post_process_vision_weights(vision_weights)
        text_weights = self._post_process_text_weights(text_weights)
        
        logger.info(f"Converted weights: {len(vision_weights)} vision, "
                   f"{len(text_weights)} text, {arch} format")
        return vision_weights, text_weights
    
    def _post_process_vision_weights(self, weights: Dict) -> Dict:
        """Applies vision-specific weight transformations"""
        processed = {}
        for k, v in weights.items():
            # Handle different position embedding formats
            if 'pos_embed' in k and v.dim() == 2:
                processed[k] = v.unsqueeze(0)  # Add batch dimension
            
            # Adapt convolutional weights
            elif 'conv' in k and v.dim() == 4:
                # Convert from (out, in, h, w) to (out, in, h, w)
                if self.vision_encoder.encoder_type == 'transformer' and 'patch_embed' not in k:
                    # No transformation needed
                    processed[k] = v
                else:
                    # Permute dimensions if needed
                    processed[k] = v.permute(2, 3, 1, 0) if self.config.get('channels_last') else v
            
            # Handle projection layers
            elif 'proj' in k and v.dim() == 2:
                target_shape = self.vision_encoder.projection.shape
                if v.shape != target_shape:
                    if v.shape[1] == target_shape[1]:
                        processed[k] = v[:target_shape[0]]
                    else:
                        logger.warning(f"Projection shape mismatch: {v.shape} vs {target_shape}")
            
            else:
                processed[k] = v
        
        return processed
    
    def _post_process_text_weights(self, weights: Dict) -> Dict:
        """Applies text-specific weight transformations"""
        processed = {}
        vocab_size = self.tokenizer.get_vocab_size()
        
        for k, v in weights.items():
            # Handle embedding layers
            if 'embed' in k:
                # Adapt to our vocabulary size
                if v.shape[0] > vocab_size:
                    processed[k] = v[:vocab_size]
                elif v.shape[0] < vocab_size:
                    # Pad with random initialization
                    new_emb = torch.randn(vocab_size, v.shape[1])
                    new_emb[:v.shape[0]] = v
                    processed[k] = new_emb
                else:
                    processed[k] = v
            
            # Handle position embeddings
            elif 'position' in k and v.dim() == 1:
                processed[k] = v.unsqueeze(0)  # Add sequence dimension
            
            else:
                processed[k] = v
        
        return processed

    def extract_performance_metrics(self, result: Any) -> dict:
        """Extracts performance metrics from the result of perform_task."""
        # This depends on what perform_task returns.
        # For pretraining, it returns {'loss': ..., 'status': ...}
        # For finetuning, it returns {'loss': ..., 'status': ..., 'predictions': ...}
        # For inference, it returns {'output': ..., 'status': ...}
        
        metrics = {}
        if isinstance(result, dict):
            if 'loss' in result:
                metrics['loss'] = result['loss']
            if 'status' in result and result['status'] == 'success':
                metrics['task_successful'] = 1.0
            else:
                metrics['task_successful'] = 0.0
            
            # If finetuning with labels and predictions, one could compute accuracy, F1, etc.
            # This requires 'labels' to be passed or stored from task_data.
            # Example (pseudo-code, needs actual labels):
            # if result.get('predictions') is not None and task_data.get('labels') is not None:
            #     if "classification" in task_data.get('downstream_task','').lower():
            #         preds = torch.argmax(result['predictions'], dim=-1)
            #         acc = (preds == task_data['labels']).float().mean().item()
            #         metrics['accuracy'] = acc
        
        return metrics

    def save_state_to_shared_memory(self):
        """Save current model state to shared memory"""
        snapshot = {
            'model_state_dict': {
                'text_encoder': self.text_encoder.state_dict(),
                'vision_encoder': self.vision_encoder.state_dict(),
                'audio_encoder': self.audio_encoder.state_dict(),
                #'transformer': self.transformer.state_dict(),
                'text_decoder': self.text_generator.state_dict(),
                'vision_decoder': self.vision_generator.state_dict(),
                'audio_decoder': self.audio_generator.state_dict(),
            },
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.perception_config
        }
        self.shared_memory.put(
            self.sm_keys['model_snapshot'], 
            snapshot,
            ttl=timedelta(days=1))
        logger.info(f"Saved model snapshot to shared memory")

    def load_state_from_shared_memory(self):
        """Load model state from shared memory"""
        snapshot = self.shared_memory.get(self.sm_keys['model_snapshot'])
        if snapshot:
            self.load_state_dict(snapshot['model_state_dict'])
            self.optimizer.load_state_dict(snapshot['optimizer_state'])
            logger.info(f"Loaded model state from shared memory")
            return True
        return False

    def cache_embeddings(self, modality: str, inputs: torch.Tensor, embeddings: torch.Tensor):
        """Cache computed embeddings in shared memory"""
        key = f"{self.sm_keys['embeddings']}:{modality}:{hash(inputs.cpu().numpy().tobytes())}"
        self.shared_memory.put(key, embeddings.detach().cpu())
        logger.debug(f"Cached embeddings: {key}")

    def get_cached_embeddings(self, modality: str, inputs: torch.Tensor):
        """Retrieve cached embeddings if available"""
        key = f"{self.sm_keys['embeddings']}:{modality}:{hash(inputs.cpu().numpy().tobytes())}"
        return self.shared_memory.get(key)

if __name__ == "__main__":
    printer.status("MAIN", "Starting PerceptionAgent Test", "info")
    from src.agents.collaborative.shared_memory import SharedMemory
    from src.agents.agent_factory import AgentFactory

    shared_mem = SharedMemory()
    agent_factory = AgentFactory()

    perception_agent = PerceptionAgent(shared_memory=shared_mem, agent_factory=agent_factory)
    #print(perception_agent)
    printer.status("INIT", "PerceptionAgent initialized successfully.", "success")

    print("\n* * * * * Phase 2 - Masking * * * * *\n")
    input_ids = torch.tensor([[1343]], dtype=torch.long, device=perception_agent.device)
    patch_size = 16
    feature_dim = 64
    encoded_output = torch.randn(1, patch_size, feature_dim, device=perception_agent.device)
    features = encoded_output
    patches  = encoded_output

    printer.pretty("TEXT", perception_agent._apply_masking_text(input_ids=input_ids), "success")
    printer.pretty("AUDIO", perception_agent._apply_masking_audio(features=features), "success")
    printer.pretty("VISION", perception_agent._apply_masking_vision(patches=patches), "success")

    print("\n* * * * * Phase 3 - Modalities * * * * *\n")
    mask_ratio = 0.15
    return_reconstructions = False

    printer.pretty("PRE1", perception_agent._pretrain_masked_modality(
        inputs={"input_ids": torch.tensor([[1343]], dtype=torch.long, device=perception_agent.device)},
        modality="text",
        mask_ratio=mask_ratio,
        return_reconstructions=return_reconstructions), "success")
    
    printer.pretty("PRE2", perception_agent._pretrain_masked_modality(
        inputs={"pixel_values": torch.randn(1, 3, 224, 224, device=perception_agent.device)},
        modality="vision",
        mask_ratio=mask_ratio,
        return_reconstructions=return_reconstructions), "success")
    
    printer.pretty("PRE3", perception_agent._pretrain_masked_modality(
        inputs={"audio_values": torch.randn(1, 1, 16000, device=perception_agent.device)},
        modality="audio",
        mask_ratio=mask_ratio,
        return_reconstructions=return_reconstructions), "success")

    print("\n* * * * * Phase 4 - Pretrain 1 * * * * *\n")
    data_mod1={'input_ids': torch.tensor([[1343]], dtype=torch.long, device=perception_agent.device)}
    data_mod2={'audio_values': torch.randn(1, 1, 16000, device=perception_agent.device)}
    mod1_type='text'
    mod2_type='audio'

    contra = perception_agent._pretrain_contrastive(
        data_mod1=data_mod1,
        data_mod2=data_mod2,
        mod1_type=mod1_type,
        mod2_type=mod2_type
    )

    printer.pretty("PRETRAIN", contra, "success" if contra else "error")

    print("\n* * * * * Phase 5 - Pretrain 2 * * * * *\n")
    checkpoint_path = "src/agents/perception/checkpoints/example_checkpoint.pt"
    source_format = "custom_audio"

    weight = perception_agent.load_pretrained_weights(
        checkpoint_path=checkpoint_path,
        source_format=source_format
    )

    printer.pretty("LOAD", weight, "success" if weight else "error")
    printer.pretty("LOAD", perception_agent.save_state_to_shared_memory(), "success" if perception_agent.save_state_to_shared_memory() else "error")

    printer.status("MAIN", "PerceptionAgent Test Finished", "info")
