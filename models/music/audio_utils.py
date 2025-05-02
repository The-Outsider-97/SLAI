import numpy as np
import soundfile as sf
import os
from scipy.signal import resample

class AudioUtils:
    TARGET_SR = 16000
    TARGET_LENGTH = 16000  # 1 second at 16kHz

    @staticmethod
    def load_audio(file_path, target_sr=TARGET_SR):
        """
        Load an audio file, resample if necessary, and return waveform as float32.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        waveform, sr = sf.read(file_path)
        
        # Resample if needed
        if sr != target_sr:
            num_samples = int(len(waveform) * target_sr / sr)
            waveform = resample(waveform, num_samples)

        # Ensure waveform is float32
        waveform = waveform.astype(np.float32)

        return waveform

    @staticmethod
    def reshape_for_model(waveform, target_length=TARGET_LENGTH):
        """
        Reshape waveform for ML model input: (batch, channels, samples)
        - Trims or pads to target_length.
        """
        if len(waveform) < target_length:
            pad_width = target_length - len(waveform)
            waveform = np.pad(waveform, (0, pad_width), mode='constant')
        else:
            waveform = waveform[:target_length]

        return waveform.reshape(1, 1, -1)  # (batch, channels, samples)

    @staticmethod
    def validate_audio(file_path):
        """
        Validate that a file is a readable audio file.
        """
        try:
            sf.info(file_path)
            return True
        except Exception:
            return False
