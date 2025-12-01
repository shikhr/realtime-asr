"""
Utility functions for audio processing.
"""

from typing import Iterator


def frame_generator(
    frame_ms: int, audio_bytes: bytes, sample_rate: int
) -> Iterator[bytes]:
    """Yield frames (bytes) of length frame_ms from audio_bytes."""
    bytes_per_sample = 2  # int16
    frame_size = int(sample_rate * (frame_ms / 1000.0)) * bytes_per_sample
    offset = 0
    total = len(audio_bytes)
    while offset + frame_size <= total:
        yield audio_bytes[offset : offset + frame_size]
        offset += frame_size
