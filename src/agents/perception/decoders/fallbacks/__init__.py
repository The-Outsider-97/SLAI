"""Audio decoder fallback backends for SLAI perception."""

from .cnn_decoder import *
from .mfcc_decoder import *
from .base_decoder import *


from .cnn_decoder import __all__ as _cnn_decoder_exports
from .mfcc_decoder import __all__ as _mfcc_decoder_exports
from .base_decoder import __all__ as _base_decoder_exports


__all__ = [
    *_cnn_decoder_exports,
    *_mfcc_decoder_exports,
    *_base_decoder_exports,
] # type: ignore