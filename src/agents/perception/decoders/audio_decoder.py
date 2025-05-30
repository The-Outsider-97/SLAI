
import yaml, json
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any

from src.agents.perception.utils.common import TensorOps, Parameter
from src.agents.perception.modules.transformer import Transformer
from logs.logger import get_logger

logger = get_logger("Audio Decoder")

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

class AudioDecoder(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        audio_cfg = config['audio_encoder']
        self.patch_size = audio_cfg['patch_size']
        self.in_channels = audio_cfg['in_channels']
        
        self.projection = Parameter(
            TensorOps.he_init((audio_cfg['embed_dim'], self.patch_size * self.in_channels))
        self.transformer = Transformer(config)  # Shared weights if desired

    def forward(self, x):
        x = self.transformer(x)
        x = torch.matmul(x[:, 1:], self.projection.data)  # Remove CLS token
        x = x.permute(0, 2, 1).reshape(-1, self.in_channels, x.shape[1]*self.patch_size)
        return x  # (B, in_channels, audio_length)
