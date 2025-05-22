import yaml
import torch
import random
import librosa
import numpy as np

from typing import Dict, List, Tuple, Optional, Union
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image

# Import existing components
from src.agents.perception.modules.tokenizer import Tokenizer
from logs.logger import get_logger

logger = get_logger("DataLoader")

CONFIG_PATH = "src/agents/perception/configs/perception_config.yaml"

def load_config(config_path=CONFIG_PATH):
    with open(config_path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class MultiModalDataset(Dataset):
    def __init__(self, config, mode: str = 'train'):
        self.config = load_config()
        self.modalities = self._detect_modalities()
        
        # Initialize modality-specific components
        if 'text' in self.modalities:
            self.tokenizer = Tokenizer(self.config)
        
        # Setup transforms
        self.transforms = self._create_transforms()
        self.mode = mode
        self.training_ext = TrainingExtensions(config)
        self.inference_opt = InferenceOptimizer(config)
        
        if self.mode == 'inference':
            self.transforms = self._create_inference_transforms()
           
    def _create_inference_transforms(self) -> Dict:
        """Create lightweight transforms for inference"""
        transforms_dict = {}
        
        if 'vision' in self.modalities:
            vision_cfg = self.config['vision_encoder']
            transforms_dict['vision'] = transforms.Compose([
                transforms.Resize(vision_cfg['img_size']),
                transforms.CenterCrop(vision_cfg['img_size']),
                transforms.ToTensor()
            ])
            
        return transforms_dict
    
    def _process_vision(self, image_path: str) -> torch.Tensor:
        """Enhanced processing with augmentation support"""
        image = super()._process_vision(image_path)
        if self.mode == 'train' and 'vision' in self.training_ext.augmentation:
            image = self.training_ext.augmentation['vision'](image)
        return image
    
    def _process_audio(self, audio_path: str) -> torch.Tensor:
        """Enhanced audio processing with noise injection"""
        waveform = super()._process_audio(audio_path)
        if self.mode == 'train' and 'audio' in self.training_ext.augmentation:
            waveform = self.training_ext.augmentation['audio'](waveform)
        return waveform

    def _detect_modalities(self) -> List[str]:
        modalities = []
        if 'vision_encoder' in self.config:
            modalities.append('vision')
        if 'audio_encoder' in self.config:
            modalities.append('audio')
        if 'text_encoder' in self.config:
            modalities.append('text')
        return modalities

    def _create_transforms(self) -> Dict:
        """Create modality-specific transformation pipelines"""
        transforms_dict = {}
        
        # Vision transforms
        if 'vision' in self.modalities:
            vision_cfg = self.config['vision_encoder']
            transforms_dict['vision'] = transforms.Compose([
                transforms.Resize(vision_cfg['img_size']),
                transforms.CenterCrop(vision_cfg['img_size']),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
        
        # Audio transforms
        if 'audio' in self.modalities:
            audio_cfg = self.config['audio_encoder']
            self.audio_length = audio_cfg['audio_length']
            self.sample_rate = audio_cfg['mfcc']['sample_rate']
        
        return transforms_dict

    def _process_vision(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        return self.transforms['vision'](image)

    def _process_audio(self, audio_path: str) -> torch.Tensor:
        # Load and preprocess audio
        waveform, _ = librosa.load(
            audio_path, 
            sr=self.sample_rate,
            duration=self.audio_length/self.sample_rate
        )
        
        # Pad/truncate to fixed length
        if len(waveform) > self.audio_length:
            waveform = waveform[:self.audio_length]
        elif len(waveform) < self.audio_length:
            waveform = np.pad(waveform, (0, max(0, self.audio_length - len(waveform))))
        
        return torch.from_numpy(waveform).float()

    def _process_text(self, text: str) -> Dict:
        return self.tokenizer(text)

    def __getitem__(self, idx: int) -> Dict:
        """Implement this method in concrete datasets"""
        raise NotImplementedError("Subclasses must implement __getitem__")

    def __len__(self) -> int:
        """Implement this method in concrete datasets"""
        raise NotImplementedError("Subclasses must implement __len__")

class MultiModalDataLoader:
    def __init__(self, dataset: Dataset, batch_size: int = 32, shuffle: bool = True):
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """Custom collate function for multi-modal batches"""
        collated = {'metadata': []}
        
        # Process each modality
        for modality in ['vision', 'audio', 'text']:
            if modality in batch[0]:
                items = [item[modality] for item in batch]
                
                if modality == 'vision':
                    collated['vision'] = torch.stack(items)
                elif modality == 'audio':
                    collated['audio'] = torch.nn.utils.rnn.pad_sequence(
                        items, batch_first=True
                    )
                elif modality == 'text':
                    collated['text'] = {
                        'input_ids': torch.stack([x['input_ids'] for x in items]),
                        'attention_mask': torch.stack([x['attention_mask'] for x in items])
                    }
        
        # Add metadata if present
        if 'metadata' in batch[0]:
            collated['metadata'] = [item['metadata'] for item in batch]
            
        return collated

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)


class TrainingExtensions:
    """Enhanced training capabilities integrated with existing DataLoader"""
    def __init__(self, config):
        self.config = load_config()
        self.augmentation = self._create_augmentations()
        
    def _create_augmentations(self) -> Dict:
        """Create advanced data augmentations from config"""
        aug_config = self.config.get('training', {}).get('augmentations', {})
        aug_dict = {}
        self.modalities = MultiModalDataset._detect_modalities(self)
        
        # Vision augmentations
        if 'vision' in self.modalities:
            vision_aug = []
            if aug_config.get('random_crop', False):
                vision_aug.append(transforms.RandomResizedCrop(
                    self.config['vision_encoder']['img_size']
                ))
            if aug_config.get('color_jitter', False):
                vision_aug.append(transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ))
            if aug_config.get('horizontal_flip', False):
                vision_aug.append(transforms.RandomHorizontalFlip())
                
            aug_dict['vision'] = transforms.Compose(vision_aug)
        
        # Audio augmentations
        if 'audio' in self.modalities and aug_config.get('audio_noise', False):
            aug_dict['audio'] = lambda x: self._add_audio_noise(x)
            
        return aug_dict
    
    def _add_audio_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        """Add Gaussian noise to audio waveform"""
        noise = torch.randn_like(waveform) * 0.005
        return waveform + noise

class InferenceOptimizer:
    """Optimizations for inference scenarios"""
    def __init__(self, config):
        self.config = load_config()
        self._init_optimizations()

    def _init_optimizations(self):
        """Initialize inference-specific optimizations"""
        self.cache = {}
        self.batch_sizes = {
            'vision': self.config['inference'].get('vision_batch', 8),
            'audio': self.config['inference'].get('audio_batch', 16),
            'text': self.config['inference'].get('text_batch', 32)
        }

    def dynamic_batching(self, samples: List[Dict]) -> List[Dict]:
        """Batch samples by modality for optimal inference performance"""
        batched = {'vision': [], 'audio': [], 'text': []}
        
        for sample in samples:
            for modality in sample.keys():
                if modality in batched:
                    batched[modality].append(sample[modality])

        return {
            mod: [batched[mod][i:i + self.batch_sizes[mod]]] 
            for mod in batched 
            for i in range(0, len(batched[mod]), self.batch_sizes[mod])
        }

# Enhanced training loop integration
class TrainingManager:
    def __init__(self, config):
        self.config = load_config()
        self._init_components()
        
    def _init_components(self):
        """Initialize model components from existing architecture"""
        from src.agents.perception.encoders.vision_encoder import VisionEncoder
        from src.agents.perception.encoders.audio_encoder import AudioEncoder
        from src.agents.perception.encoders.text_encoder import TextEncoder
        
        self.models = {
            'vision': VisionEncoder(self.config),
            'audio': AudioEncoder(self.config),
            'text': TextEncoder(self.config)
        }
        
    def train_step(self, batch: Dict) -> Dict:
        """Complete training step handling multi-modal data"""
        outputs = {}

        # Process each modality
        if 'vision' in batch:
            outputs['vision'] = self.models['vision'](batch['vision'])
        if 'audio' in batch:
            outputs['audio'] = self.models['audio'](batch['audio'])
        if 'text' in batch:
            outputs['text'] = self.models['text'](batch['text']['input_ids'])

        # Fusion and loss calculation
        fused_output = self.fuse_outputs(outputs)
        loss = self.calculate_loss(fused_output, batch['metadata'])
        return loss

    def fuse_outputs(self, outputs: Dict) -> torch.Tensor:
        """Fuse multi-modal outputs according to config"""
        fusion_method = self.config['task_heads']['multimodal']['fusion_method']

        if fusion_method == 'concat':
            return torch.cat([outputs[k] for k in outputs.keys()], dim=-1)
        elif fusion_method == 'mean':
            return torch.mean(torch.stack([outputs[k] for k in outputs.keys()]), dim=0)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

# Enhanced inference handler
class InferenceEngine:
    def __init__(self, config):
        self.config = load_config()
        self.models = TrainingManager(config).models
        self.dataset = MultiModalDataset(config, mode='inference')
        
    def process_single(self, sample: Dict) -> Dict:
        """Process single sample for inference"""
        batch = self.dataset.inference_opt.dynamic_batching([sample])
        return self.process_batch(batch)
    
    def process_batch(self, batch: Dict) -> Dict:
        """Process batched inputs for inference"""
        results = {}

        with torch.no_grad():
            for modality in batch.keys():
                if modality == 'vision':
                    results[modality] = self.models['vision'](batch[modality])
                elif modality == 'audio':
                    results[modality] = self.models['audio'](batch[modality])
                elif modality == 'text':
                    results[modality] = self.models['text'](batch[modality]['input_ids'])

        return self.fuse_results(results)

    def fuse_results(self, results: Dict) -> torch.Tensor:
        """Fuse results using task head configuration"""
        task_head_cfg = self.config['task_heads'][self.config['training']['task']]
        return torch.cat([results[k] for k in results.keys()], dim=-1)


if __name__ == "__main__":

    # Example concrete dataset implementation
    class ExampleMultiModalDataset(MultiModalDataset):
        def __init__(self, data_root: str, config):
            super().__init__(config)
            self.data_root = Path(data_root)
            self.samples = self._load_samples()
            
        def _load_samples(self) -> List[Dict]:
            """Implement actual sample loading logic here"""
            return [
                {
                    'vision': 'path/to/image.jpg',
                    'audio': 'path/to/audio.wav',
                    'text': 'example text',
                    'metadata': {'label': 0}
                }
            ]  # Replace with actual data loading

        def __getitem__(self, idx: int) -> Dict:
            sample = self.samples[idx]
            item = {}
            
            if 'vision' in sample:
                item['vision'] = self._process_vision(sample['vision'])
            if 'audio' in sample:
                item['audio'] = self._process_audio(sample['audio'])
            if 'text' in sample:
                item['text'] = self._process_text(sample['text'])
            
            if 'metadata' in sample:
                item['metadata'] = sample['metadata']
                
            return item

        def __len__(self) -> int:
            return len(self.samples)

    print("\n=== Running Shared Memory ===\n")
    config = load_config(CONFIG_PATH)
    dataset=ExampleMultiModalDataset(data_root="data", config=config)
    batch_size = 32
    shuffle = True
    
    loader = MultiModalDataLoader(dataset, batch_size, shuffle)
    set = MultiModalDataset(config)


    print(loader)
    print(set)
    print("\n=== Successfully Ran Shared Memory ===\n")
