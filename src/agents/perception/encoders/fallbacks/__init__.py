"""Fallback encoder backends for SLAI perception."""

from .cnn_encoder import *
from .mfcc_encoder import *
from .base_encoder import *


from .cnn_encoder import __all__ as _cnn_encoder_exports
from .mfcc_encoder import __all__ as _mfcc_encoder_exports
from .base_encoder import __all__ as _base_encoder_exports


__all__ = [
    *_cnn_encoder_exports,
    *_mfcc_encoder_exports,
    *_base_encoder_exports,
] # type: ignore