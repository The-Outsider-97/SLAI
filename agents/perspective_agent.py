import cv2
import numpy as np
import torchaudio
from transformers import CLIPProcessor, CLIPModel

class PerceptionAgent:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def process_image(self, image_path):
        image = cv2.imread(image_path)
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.clip_model.get_image_features(**inputs)
        return outputs.detach().numpy()

    def process_audio(self, audio_path):
        waveform, sample_rate = torchaudio.load(audio_path)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate)(waveform)
        return mel_spectrogram.mean(dim=-1).squeeze().numpy()
