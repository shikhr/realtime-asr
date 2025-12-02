"""
Voice Activity Detection (VAD) processor.
Processes audio frames and assembles utterances.
This module is independent of any transport or UI.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterator

import webrtcvad

from . import config


@dataclass
class Utterance:
    """Represents a detected speech utterance."""

    audio_bytes: bytes
    start_ts: float
    end_ts: float

    @property
    def duration_ms(self) -> float:
        return (self.end_ts - self.start_ts) * 1000.0


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


class VADProcessor:
    """
    VAD-based utterance detector.

    Processes audio chunks and emits complete utterances via callback.
    This class maintains internal state for utterance assembly.
    """

    def __init__(
        self,
        on_utterance: Callable[[Utterance], None] | None = None,
        sample_rate: int = config.SAMPLE_RATE,
        aggressiveness: int = config.VAD_AGGRESSIVENESS,
        frame_ms: int = config.VAD_FRAME_MS,
        min_utterance_ms: int = config.UTTERANCE_MIN_MS,
        max_utterance_ms: int = config.UTTERANCE_MAX_MS,
        end_padding_ms: int = config.UTTERANCE_END_PADDING_MS,
        pre_roll_ms: int = config.PRE_ROLL_MS,
        streaming_split_after_ms: int = config.STREAMING_SPLIT_AFTER_MS,
        streaming_split_silence_ms: int = config.STREAMING_SPLIT_SILENCE_MS,
    ):
        self.on_utterance = on_utterance
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.min_utterance_ms = min_utterance_ms
        self.max_utterance_ms = max_utterance_ms
        self.end_padding_ms = end_padding_ms
        self.pre_roll_ms = pre_roll_ms
        self.streaming_split_after_ms = streaming_split_after_ms
        self.streaming_split_silence_ms = streaming_split_silence_ms

        self.vad = webrtcvad.Vad(int(aggressiveness))
        self.pre_roll_buf: deque[tuple[bytes, float]] = deque(
            maxlen=max(1, int(math.ceil(pre_roll_ms / frame_ms)))
        )

        # State
        self._collecting = False
        self._utterance_bytes = bytearray()
        self._utter_start_ts: float | None = None
        self._last_speech_ts: float | None = None
        self._accumulated_ms = 0

    def reset(self) -> None:
        """Reset internal state."""
        self.pre_roll_buf.clear()
        self._collecting = False
        self._utterance_bytes = bytearray()
        self._utter_start_ts = None
        self._last_speech_ts = None
        self._accumulated_ms = 0

    def process_chunk(self, audio_bytes: bytes, chunk_ts: float) -> list[Utterance]:
        """
        Process an audio chunk and return any completed utterances.

        Args:
            audio_bytes: Raw PCM audio data (int16, mono)
            chunk_ts: Timestamp of the chunk

        Returns:
            List of completed utterances (may be empty)
        """
        utterances: list[Utterance] = []
        frames = list(frame_generator(self.frame_ms, audio_bytes, self.sample_rate))

        for i, frame in enumerate(frames):
            frame_ts = chunk_ts + (i * self.frame_ms) / 1000.0
            self.pre_roll_buf.append((frame, frame_ts))

            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
            except Exception:
                is_speech = False

            if is_speech:
                if not self._collecting:
                    # Start new utterance with pre-roll
                    self._collecting = True
                    self._utterance_bytes = bytearray()
                    for pr_frame, _ in self.pre_roll_buf:
                        self._utterance_bytes.extend(pr_frame)
                    self._utter_start_ts = (
                        self.pre_roll_buf[0][1]
                        if len(self.pre_roll_buf) > 0
                        else frame_ts
                    )
                    self._accumulated_ms = 0
                    self._last_speech_ts = frame_ts

                self._utterance_bytes.extend(frame)
                self._accumulated_ms += self.frame_ms
                self._last_speech_ts = frame_ts

                # Check max length
                if self._accumulated_ms >= self.max_utterance_ms:
                    utterance = self._finalize_utterance(frame_ts)
                    if utterance:
                        utterances.append(utterance)

            else:
                # Silence frame
                if self._collecting:
                    self._utterance_bytes.extend(frame)
                    self._accumulated_ms += self.frame_ms

                    if self._last_speech_ts is not None:
                        silence_ms = (frame_ts - self._last_speech_ts) * 1000.0

                        # Determine split threshold
                        if self._accumulated_ms >= self.streaming_split_after_ms:
                            split_threshold = self.streaming_split_silence_ms
                        else:
                            split_threshold = self.end_padding_ms

                        if silence_ms >= split_threshold:
                            if self._accumulated_ms >= self.min_utterance_ms:
                                utterance = self._finalize_utterance(frame_ts)
                                if utterance:
                                    utterances.append(utterance)
                            else:
                                # Too short, discard
                                self._reset_collection()

        # Emit utterances via callback
        for utt in utterances:
            if self.on_utterance:
                self.on_utterance(utt)

        return utterances

    def _finalize_utterance(self, end_ts: float) -> Utterance | None:
        """Finalize current utterance and reset state."""
        if not self._utterance_bytes or self._utter_start_ts is None:
            self._reset_collection()
            return None

        utterance = Utterance(
            audio_bytes=bytes(self._utterance_bytes),
            start_ts=self._utter_start_ts,
            end_ts=end_ts,
        )
        self._reset_collection()
        return utterance

    def _reset_collection(self) -> None:
        """Reset collection state."""
        self._collecting = False
        self._utterance_bytes = bytearray()
        self._utter_start_ts = None
        self._last_speech_ts = None
        self._accumulated_ms = 0

    def update_config(
        self,
        aggressiveness: int | None = None,
        min_utterance_ms: int | None = None,
        max_utterance_ms: int | None = None,
        end_padding_ms: int | None = None,
        pre_roll_ms: int | None = None,
        streaming_split_after_ms: int | None = None,
        streaming_split_silence_ms: int | None = None,
    ) -> None:
        """
        Update VAD configuration at runtime.
        Only non-None values are updated.
        """
        if aggressiveness is not None:
            self.vad.set_mode(int(aggressiveness))

        if min_utterance_ms is not None:
            self.min_utterance_ms = min_utterance_ms

        if max_utterance_ms is not None:
            self.max_utterance_ms = max_utterance_ms

        if end_padding_ms is not None:
            self.end_padding_ms = end_padding_ms

        if pre_roll_ms is not None:
            self.pre_roll_ms = pre_roll_ms
            self.pre_roll_buf = deque(
                maxlen=max(1, int(math.ceil(pre_roll_ms / self.frame_ms)))
            )

        if streaming_split_after_ms is not None:
            self.streaming_split_after_ms = streaming_split_after_ms

        if streaming_split_silence_ms is not None:
            self.streaming_split_silence_ms = streaming_split_silence_ms
