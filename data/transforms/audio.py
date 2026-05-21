"""
Audio transforms: resampling, mono conversion, MFCC and mel-spectrogram extraction.
"""
from __future__ import annotations
 
import numpy as np  # type: ignore
 
from typing import Any, Dict, Optional
 
from ..utils.config_loader import get_config_section, load_global_config
from ..utils.data_error import *
from ..utils.data_helpers import *
from .base_transform import Transform
from .registry import register_transform
from logs.logger import get_logger, PrettyPrinter  # pyright: ignore[reportMissingImports]
 
logger = get_logger("Audio Transform")
printer = PrettyPrinter()
 
# Optional audio libraries — guarded at import time so the module loads even
# when neither is installed; each transform raises DataConfigError at call
# time if its required library is absent.
try:
    import librosa  # type: ignore
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
 
try:
    import torchaudio  # type: ignore
    TORCHAUDIO_AVAILABLE = True
except ImportError:
    TORCHAUDIO_AVAILABLE = False
 
 
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_librosa(transform_name: str) -> None:
    if not LIBROSA_AVAILABLE:
        raise DataConfigError(
            f"{transform_name} requires librosa: pip install librosa",
            context={"transform": transform_name},
        )
 
 
def _require_torchaudio(transform_name: str) -> None:
    if not TORCHAUDIO_AVAILABLE:
        raise DataConfigError(
            f"{transform_name} requires torchaudio: pip install torchaudio",
            context={"transform": transform_name},
        )
 
 
def _extract_audio_fields(
    record: Dict[str, Any],
    transform_name: str,
) -> Optional[tuple[np.ndarray, int]]:
    """Return (audio_array, sample_rate) or None when fields are absent."""
    audio = record.get("audio")
    sr = record.get("sample_rate")
    if audio is None or sr is None:
        logger.debug({
            "event": "audio_fields_missing",
            "transform": transform_name,
            "has_audio": audio is not None,
            "has_sample_rate": sr is not None,
        })
        return None
    if not isinstance(audio, np.ndarray):
        raise DataTransformError(
            f"{transform_name} expected audio as np.ndarray",
            context={"transform": transform_name, "got": type(audio).__name__},
        )
    return audio, int(sr)
 
 
# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------
# @register_transform("to_mono")
class ToMono(Transform):
    """Convert stereo (2, N) audio to mono by averaging channels.
 
    If the array is already 1-D (mono) it is returned unchanged.
    Multi-channel arrays with more than 2 channels are also averaged
    across axis 0.
    """
 
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "audio":
            return record
        fields = _extract_audio_fields(record, "ToMono")
        if fields is None:
            return record
        audio, sr = fields
        if audio.ndim > 1:
            record["audio"] = np.mean(audio, axis=0)
        return record
 
    def _get_params(self) -> Dict[str, Any]:
        return {}
 
 
# @register_transform("resample_audio")
class ResampleAudio(Transform):
    """Resample audio waveform to *target_sr*.
 
    Two backends are supported:
 
    * **librosa** (default) — pure-Python, widely available.
    * **torchaudio** — GPU-accelerated, preferred in torch-based pipelines.
 
    Set ``use_librosa: false`` in ``transforms.audio`` to switch backends.
    """
 
    def __init__(self) -> None:
        super().__init__()
        self.audio_cfg: Dict[str, Any] = get_config_section("transforms").get("audio", {})
        self.target_sr: int = int(self.audio_cfg.get("target_sr", 16000))
        self.res_type: str = str(self.audio_cfg.get("res_type", "kaiser_best"))
        self.use_librosa: bool = bool(self.audio_cfg.get("use_librosa", True))
 
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "audio":
            return record
        fields = _extract_audio_fields(record, "ResampleAudio")
        if fields is None:
            return record
        audio, sr = fields
 
        if sr == self.target_sr:
            return record
 
        try:
            if self.use_librosa:
                _require_librosa("ResampleAudio")
                resampled: np.ndarray = librosa.resample(
                    audio, orig_sr=sr, target_sr=self.target_sr, res_type=self.res_type
                )
            else:
                _require_torchaudio("ResampleAudio")
                import torch
                audio_t = torch.from_numpy(audio).float()
                resampled_t = torchaudio.functional.resample(audio_t, sr, self.target_sr)
                resampled = resampled_t.numpy()
        except (DataConfigError, DataTransformError):
            raise
        except Exception as exc:
            raise DataTransformError(
                f"ResampleAudio failed: {sr} Hz → {self.target_sr} Hz",
                context={"modality": modality, "orig_sr": sr, "target_sr": self.target_sr},
                cause=exc,
            ) from exc
 
        record["audio"] = resampled
        record["sample_rate"] = self.target_sr
        logger.debug({
            "event": "audio_resampled",
            "orig_sr": sr,
            "target_sr": self.target_sr,
            "frames_in": len(audio),
            "frames_out": len(resampled),
        })
        return record
 
    def _get_params(self) -> Dict[str, Any]:
        return {
            "target_sr": self.target_sr,
            "res_type": self.res_type,
            "use_librosa": self.use_librosa,
        }
 
 
# @register_transform("extract_mfcc")
class ExtractMFCC(Transform):
    """Extract MFCC feature matrix from a waveform.
 
    The result is stored as ``record["mfcc"]`` with shape ``(T, n_mfcc)``
    where *T* is the number of time frames.
 
    Requires **librosa**.  ``use_librosa: false`` raises ``DataConfigError``
    because no alternative backend is currently implemented.
    """
 
    def __init__(self) -> None:
        super().__init__()
        self.mfcc_cfg: Dict[str, Any] = get_config_section("transforms").get("mfcc", {})
        self.n_mfcc: int = int(self.mfcc_cfg.get("n_mfcc", 13))
        self.n_fft: int = int(self.mfcc_cfg.get("n_fft", 2048))
        self.hop_length: int = int(self.mfcc_cfg.get("hop_length", 512))
        self.use_librosa: bool = bool(self.mfcc_cfg.get("use_librosa", True))
 
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "audio":
            return record
        fields = _extract_audio_fields(record, "ExtractMFCC")
        if fields is None:
            return record
        audio, sr = fields
 
        if not self.use_librosa:
            raise DataConfigError(
                "ExtractMFCC: non-librosa backend not implemented; set use_librosa: true",
                context={"n_mfcc": self.n_mfcc},
            )
        _require_librosa("ExtractMFCC")
 
        try:
            mfcc: np.ndarray = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
        except Exception as exc:
            raise DataTransformError(
                "ExtractMFCC failed",
                context={"modality": modality, "n_mfcc": self.n_mfcc, "n_fft": self.n_fft},
                cause=exc,
            ) from exc
 
        record["mfcc"] = mfcc.T  # shape (T, n_mfcc)
        logger.debug({
            "event": "mfcc_extracted",
            "n_mfcc": self.n_mfcc,
            "shape": list(record["mfcc"].shape),
        })
        return record
 
    def _get_params(self) -> Dict[str, Any]:
        return {
            "n_mfcc": self.n_mfcc,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "use_librosa": self.use_librosa,
        }
 
 
# @register_transform("extract_mel_spectrogram")
class ExtractMelSpectrogram(Transform):
    """Compute a log-mel spectrogram and store it as ``record["mel_spectrogram"]``.
 
    Shape of result: ``(T, n_mels)``.  Requires **librosa**.
    """
 
    def __init__(self) -> None:
        super().__init__()
        self.mel_cfg: Dict[str, Any] = get_config_section("transforms").get("mel_spectrogram", {})
        self.n_mels: int = int(self.mel_cfg.get("n_mels", 128))
        self.n_fft: int = int(self.mel_cfg.get("n_fft", 2048))
        self.hop_length: int = int(self.mel_cfg.get("hop_length", 512))
        self.fmin: float = float(self.mel_cfg.get("fmin", 0.0))
        self.fmax: Optional[float] = self.mel_cfg.get("fmax", None)
 
    def __call__(self, record: Dict[str, Any], modality: str) -> Dict[str, Any]:
        if modality != "audio":
            return record
        fields = _extract_audio_fields(record, "ExtractMelSpectrogram")
        if fields is None:
            return record
        audio, sr = fields
        _require_librosa("ExtractMelSpectrogram")
 
        try:
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
            )
            mel_db: np.ndarray = librosa.power_to_db(mel, ref=np.max)
        except Exception as exc:
            raise DataTransformError(
                "ExtractMelSpectrogram failed",
                context={"modality": modality, "n_mels": self.n_mels},
                cause=exc,
            ) from exc
 
        record["mel_spectrogram"] = mel_db.T  # (T, n_mels)
        return record
 
    def _get_params(self) -> Dict[str, Any]:
        return {
            "n_mels": self.n_mels,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "fmin": self.fmin,
            "fmax": self.fmax,
        }


if __name__ == "__main__":
    print("\n=== Running audio ===\n")
    printer.status("TEST", "audio initialized", "info")
 
    SR = 16000
 
    def _make_record(sr: int = SR, channels: int = 1) -> Dict[str, Any]:
        shape = (channels, SR) if channels > 1 else (SR,)
        return {"audio": np.zeros(shape, dtype=np.float32), "sample_rate": sr}
 
    # ToMono — stereo → mono
    rec = _make_record(channels=2)
    assert rec["audio"].shape == (2, SR)
    out = ToMono()(rec, "audio")
    assert out["audio"].ndim == 1 and out["audio"].shape[0] == SR
    printer.status("PASS", "ToMono stereo→mono", "success")
 
    # ToMono — already mono, unchanged
    rec_mono = _make_record(channels=1)
    out_mono = ToMono()(rec_mono, "audio")
    assert out_mono["audio"].ndim == 1
    printer.status("PASS", "ToMono mono unchanged", "success")
 
    # ToMono — wrong modality skipped
    rec_text = {"text": "hello"}
    out_t = ToMono()(rec_text, "text")
    assert "audio" not in out_t
    printer.status("PASS", "ToMono skips non-audio modality", "success")
 
    # ResampleAudio — same SR is a no-op
    rec_sr = _make_record(sr=SR)
    out_sr = ResampleAudio()(rec_sr, "audio")
    assert out_sr["sample_rate"] == SR
    printer.status("PASS", "ResampleAudio no-op when SR already matches", "success")
 
    # ExtractMFCC — missing audio fields → passthrough
    rec_empty = {"modality": "audio"}
    out_empty = ExtractMFCC()(rec_empty, "audio")
    assert "mfcc" not in out_empty
    printer.status("PASS", "ExtractMFCC skips records without audio/sample_rate", "success")
 
    # ExtractMelSpectrogram — missing audio fields → passthrough
    out_mel_empty = ExtractMelSpectrogram()(rec_empty, "audio")
    assert "mel_spectrogram" not in out_mel_empty
    printer.status("PASS", "ExtractMelSpectrogram skips missing fields", "success")
 
    # _get_params coverage
    assert ResampleAudio()._get_params()["target_sr"] == 16000
    assert ExtractMFCC()._get_params()["n_mfcc"] == 13
    assert ExtractMelSpectrogram()._get_params()["n_mels"] == 128
    printer.status("PASS", "_get_params returns correct defaults", "success")
 
    # DataTransformError on bad audio type
    bad_rec = {"audio": "not_an_array", "sample_rate": SR}
    try:
        ResampleAudio()(bad_rec, "audio")
        assert False
    except DataTransformError:
        printer.status("PASS", "ResampleAudio raises DataTransformError on bad audio type", "success")
 
    print("\n=== Test ran successfully ===\n")