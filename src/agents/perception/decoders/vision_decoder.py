import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger

logger = get_logger("Vision Decoder")

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

class VisionDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        vision_cfg = config['vision_encoder']
        self.decoder_type = vision_cfg['encoder_type']  # Mirror encoder type
        
        if self.decoder_type == "transformer":
            self.projection = Parameter(
                TensorOps.he_init((vision_cfg['embed_dim'], vision_cfg['patch_size']**2 * vision_cfg['in_channels']))
            self.transformer = Transformer(config)  # Reuse transformer
            self.patch_size = vision_cfg['patch_size']
        elif self.decoder_type == "cnn":
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2),
                nn.Sigmoid()  # Normalize to [0,1]
            )

    def forward(self, x):
        if self.decoder_type == "transformer":
            x = self.transformer(x)
            x = torch.matmul(x[:, 1:], self.projection.data)  # Skip CLS token
            x = self._patch2img(x)
            return x
        else:
            return self.decoder(x)

    def _patch2img(self, x):
        # Reverse of extract_patches: (B, num_patches, patch_dim) → (B, C, H, W)
        B, N, D = x.shape
        p = self.patch_size
        c = D // (p**2)
        hw = int(math.sqrt(N))
        x = x.reshape(B, hw, hw, c, p, p)
        x = x.permute(0, 3, 1, 4, 2, 5).reshape(B, c, hw*p, hw*p)
        return x
