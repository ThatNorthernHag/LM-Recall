"""Decoder backends.

Each backend implements the Decoder protocol from base.py. The
pipeline.py module accepts any Decoder; per-decoder wrapper scripts in
the parent directory instantiate the right one and hand it off.
"""

from .base import Decoder, DecoderError

__all__ = ["Decoder", "DecoderError"]
