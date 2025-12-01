"""
Audio stream handling: input streams and callbacks.
"""

import time
from typing import Any

import pyaudio

from . import config
from .queues import record_queue

# PyAudio instance
pa = pyaudio.PyAudio()


def record_callback(
    in_data: bytes | None,
    frame_count: int,
    time_info: dict[str, float],
    status_flags: int,
) -> tuple[None, int]:
    """PyAudio callback that puts recorded chunks into record_queue."""
    if in_data is not None:
        ts = time.time()
        try:
            record_queue.put_nowait((in_data, ts))
        except Exception:
            # Drop policy when overwhelmed
            pass
    return (None, pyaudio.paContinue)


def create_input_stream() -> pyaudio.Stream:
    """Create and return the input audio stream."""
    return pa.open(
        format=config.FORMAT,
        channels=config.CHANNELS,
        rate=config.RATE,
        input=True,
        frames_per_buffer=config.FRAMES_PER_BUFFER,
        stream_callback=record_callback,
    )


def terminate() -> None:
    """Terminate the PyAudio instance."""
    pa.terminate()
