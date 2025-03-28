import asyncio
import logging
from multiprocessing import Pool, cpu_count
from functools import partial

import gc
import cv2
import torch
import torchaudio
import sounddevice as sd
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipModel
from typing import Optional, Dict, AsyncGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('perception_agent.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerceptionAgent:
    def __init__(self, device: Optional[str] = None, max_workers: Optional[int] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing PerceptionAgent on device: {self.device}")
        
        # Initialize BLIP model
        try:
            self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.model = BlipModel.from_pretrained("Salesforce/blip-image-captioning-base").eval().to(self.device)
            logger.info("BLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BLIP model: {e}")
            raise

        # Audio configuration
        self.sample_rate = 16000
        self.n_fft = 1024
        self.hop_length = 512
        self.n_mels = 64
        self.audio_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
        ).to(self.device)

        # Multiprocessing pool for CPU-bound tasks
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.pool = Pool(processes=self.max_workers)
        logger.info(f"Initialized multiprocessing pool with {self.max_workers} workers")

        # Memory management
        self.max_feature_cache = 1000  # Maximum features to cache
        self.feature_cache = {}
        self._shutdown = False

    async def process_image(self, image_path: str) -> torch.Tensor:
        """Process image asynchronously using multiprocessing"""
        try:
            # Offload to process pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # Uses default executor (thread pool)
                partial(self._process_image_sync, image_path)
            )
            return result
        except Exception as e:
            logger.error(f"Image processing failed for {image_path}: {e}")
            return torch.zeros(1, device="cpu")

    def _process_image_sync(self, image_path: str) -> torch.Tensor:
        """Synchronous image processing (runs in worker process)"""
        try:
            if image_path in self.feature_cache:
                return self.feature_cache[image_path]

            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.vision_model(**inputs)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu()

            # Cache management
            if len(self.feature_cache) >= self.max_feature_cache:
                oldest_key = next(iter(self.feature_cache))
                del self.feature_cache[oldest_key]
            self.feature_cache[image_path] = features

            return features
        except Exception as e:
            logger.error(f"Sync image processing failed: {e}")
            return torch.zeros(1, device="cpu")

    async def process_audio(self, audio_path: str) -> torch.Tensor:
        """Process audio asynchronously using multiprocessing"""
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                partial(self._process_audio_sync, audio_path)
            )
            return result
        except Exception as e:
            logger.error(f"Audio processing failed for {audio_path}: {e}")
            return torch.zeros(self.n_mels, device="cpu")

    def _process_audio_sync(self, audio_path: str) -> torch.Tensor:
        """Synchronous audio processing (runs in worker process)"""
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=self.sample_rate)
            waveform = waveform.to(self.device)
            mel = self.audio_transform(waveform)
            return torch.mean(mel, dim=-1).squeeze().cpu()
        except Exception as e:
            logger.error(f"Sync audio processing failed: {e}")
            return torch.zeros(self.n_mels, device="cpu")

    async def stream_camera(self, camera_index: int = 0, frame_skip: int = 0) -> AsyncGenerator[torch.Tensor, None]:
        """Asynchronous camera streaming with memory management"""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error(f"Could not open camera {camera_index}")
            return

        frame_count = 0
        try:
            while not self._shutdown:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Camera frame read failed")
                    await asyncio.sleep(0.1)
                    continue

                frame_count += 1
                if frame_skip > 0 and frame_count % (frame_skip + 1) != 0:
                    continue

                # Process frame asynchronously
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                
                with torch.no_grad():
                    features = self.model.vision_model(**inputs).last_hidden_state.mean(dim=1).squeeze().cpu()
                
                # Explicit cleanup
                del frame, img, inputs
                if frame_count % 10 == 0:
                    gc.collect()
                
                yield features
                await asyncio.sleep(0)  # Yield control to event loop

        except Exception as e:
            logger.error(f"Camera streaming failed: {e}")
        finally:
            cap.release()
            logger.info("Camera stream released")

    async def stream_audio(self, duration: float = 1.0) -> AsyncGenerator[torch.Tensor, None]:
        """Asynchronous audio streaming with memory management"""
        audio_queue = asyncio.Queue(maxsize=5)
        self._shutdown = False

        def callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio stream status: {status}")
            if not self._shutdown:
                try:
                    waveform = torch.from_numpy(indata.T.copy()).float()
                    waveform = waveform.to(self.device)
                    mel = self.audio_transform(waveform)
                    features = torch.mean(mel, dim=-1).squeeze().cpu()
                    
                    # Non-blocking put with timeout
                    try:
                        audio_queue.put_nowait(features)
                    except asyncio.QueueFull:
                        logger.warning("Audio queue full - dropping frame")
                except Exception as e:
                    logger.error(f"Audio callback error: {e}")

        try:
            with sd.InputStream(
                callback=callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=int(self.sample_rate * duration)
            ):
                logger.info("Audio stream started")
                while not self._shutdown:
                    try:
                        features = await asyncio.wait_for(audio_queue.get(), timeout=duration * 2)
                        yield features
                    except asyncio.TimeoutError:
                        logger.warning("Audio stream timeout")
                        continue
        except Exception as e:
            logger.error(f"Audio streaming failed: {e}")
        finally:
            logger.info("Audio stream stopped")

    async def shutdown(self):
        """Clean shutdown of all resources"""
        logger.info("Shutting down PerceptionAgent")
        self._shutdown = True
        
        # Cleanup multiprocessing pool
        self.pool.close()
        self.pool.join()
        
        # Clear caches
        self.feature_cache.clear()
        gc.collect()
        logger.info("Shutdown complete")

    def fuse_modalities(self, image_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """Fuse features with memory efficiency"""
        with torch.no_grad():
            image_features = image_features / image_features.norm() if image_features is not None else torch.zeros(1)
            audio_features = audio_features / audio_features.norm() if audio_features is not None else torch.zeros(1)
            fused = torch.cat([image_features, audio_features])
            return fused

    async def attention_router(self, context: Dict[str, torch.Tensor], weights: tuple = (0.5, 0.5)) -> torch.Tensor:
        """Async attention routing with memory management"""
        try:
            # Process features in parallel
            image_task = asyncio.create_task(
                self.process_image(context["image"]) if "image" in context 
                else asyncio.sleep(0)
            )
            audio_task = asyncio.create_task(
                self.process_audio(context["audio"]) if "audio" in context 
                else asyncio.sleep(0)
            )
            
            image_feat, audio_feat = await asyncio.gather(image_task, audio_task)
            
            with torch.no_grad():
                result = weights[0] * image_feat + weights[1] * audio_feat
                return result
        except Exception as e:
            logger.error(f"Attention routing failed: {e}")
            return torch.zeros(1)
