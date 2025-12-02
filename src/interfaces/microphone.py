"""
Microphone input interface using PyAudio.
"""

import time
from typing import Callable, Protocol

import pyaudio

from ..core import config


class AudioCallback(Protocol):
    """Protocol for audio chunk callbacks."""

    def __call__(self, audio_bytes: bytes, timestamp: float) -> None: ...


class MicrophoneInput:
    """
    Microphone input using PyAudio.

    Captures audio from the default microphone and sends chunks
    to a callback function.
    """

    def __init__(
        self,
        on_audio: AudioCallback | None = None,
        sample_rate: int = config.SAMPLE_RATE,
        channels: int = config.CHANNELS,
        chunk_ms: int = 100,
    ):
        self.on_audio = on_audio
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_ms = chunk_ms
        self.frames_per_buffer = int(sample_rate * chunk_ms / 1000)

        self._pa: pyaudio.PyAudio | None = None
        self._stream: pyaudio.Stream | None = None

    def _callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict[str, float],
        status_flags: int,
    ) -> tuple[None, int]:
        """PyAudio callback."""
        if in_data is not None and self.on_audio:
            ts = time.time()
            try:
                self.on_audio(in_data, ts)
            except Exception:
                pass  # Don't crash the audio thread
        return (None, pyaudio.paContinue)

    def start(self) -> None:
        """Start capturing audio from microphone."""
        if self._stream is not None:
            return  # Already running

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            stream_callback=self._callback,
        )
        self._stream.start_stream()

    def stop(self) -> None:
        """Stop capturing audio."""
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None

        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    def is_active(self) -> bool:
        """Check if the stream is active."""
        return self._stream is not None and self._stream.is_active()

    def __enter__(self) -> "MicrophoneInput":
        self.start()
        return self

    def __exit__(self, *args) -> None:
        self.stop()
