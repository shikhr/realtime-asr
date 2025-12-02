"""
STT Pipeline - Orchestrates audio input → VAD → ASR flow.
"""

import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable

from ..core.asr import get_asr_engine
from ..core.vad import VADProcessor, Utterance
from ..core import config


@dataclass
class TranscriptResult:
    """Result of transcription."""

    text: str
    duration_s: float
    timestamp: float


class STTPipeline:
    """
    Speech-to-Text pipeline that processes audio → VAD → ASR.

    This class is independent of the audio source - it receives
    audio chunks and emits transcription results.
    """

    def __init__(
        self,
        on_transcript: Callable[[TranscriptResult], None] | None = None,
        on_utterance: Callable[[Utterance], None] | None = None,
        sample_rate: int = config.SAMPLE_RATE,
    ):
        self.on_transcript = on_transcript
        self.on_utterance = on_utterance
        self.sample_rate = sample_rate

        # ASR engine (lazy loaded)
        self._asr_engine = None

        # VAD processor
        self._vad = VADProcessor(
            on_utterance=self._handle_utterance,
            sample_rate=sample_rate,
        )

        # Internal queues for async processing
        self._utterance_queue: queue.Queue[Utterance] = queue.Queue(maxsize=50)
        self._stop_event = threading.Event()
        self._asr_thread: threading.Thread | None = None

    def initialize(self) -> None:
        """Initialize the pipeline (loads ASR model)."""
        if self._asr_engine is None:
            self._asr_engine = get_asr_engine()

    def start(self) -> None:
        """Start the ASR processing thread."""
        self._stop_event.clear()
        self._asr_thread = threading.Thread(target=self._asr_worker, daemon=True)
        self._asr_thread.start()

    def stop(self) -> None:
        """Stop the pipeline."""
        self._stop_event.set()
        if self._asr_thread:
            self._asr_thread.join(timeout=1.0)
            self._asr_thread = None

        # Clear queues
        while not self._utterance_queue.empty():
            try:
                self._utterance_queue.get_nowait()
            except Exception:
                break

        # Reset VAD state
        self._vad.reset()

    def process_audio(self, audio_bytes: bytes, timestamp: float) -> None:
        """
        Process an audio chunk.

        Args:
            audio_bytes: Raw PCM audio (int16, mono)
            timestamp: Timestamp of the chunk
        """
        self._vad.process_chunk(audio_bytes, timestamp)

    def _handle_utterance(self, utterance: Utterance) -> None:
        """Handle detected utterance from VAD."""
        if self.on_utterance:
            self.on_utterance(utterance)

        # Queue for ASR processing
        try:
            self._utterance_queue.put_nowait(utterance)
        except queue.Full:
            # Drop oldest if full
            try:
                self._utterance_queue.get_nowait()
                self._utterance_queue.put_nowait(utterance)
            except Exception:
                pass

    def _asr_worker(self) -> None:
        """ASR worker thread."""
        if self._asr_engine is None:
            self._asr_engine = get_asr_engine()

        while not self._stop_event.is_set():
            try:
                utterance = self._utterance_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
                text = self._asr_engine.transcribe_bytes(utterance.audio_bytes)
                result = TranscriptResult(
                    text=text,
                    duration_s=utterance.duration_ms / 1000.0,
                    timestamp=time.time(),
                )
                if self.on_transcript:
                    self.on_transcript(result)
            except Exception:
                pass  # Silently handle ASR errors

    def update_vad_config(self, **kwargs) -> None:
        """Update VAD configuration at runtime."""
        self._vad.update_config(**kwargs)


class TranscriptBatcher:
    """
    Batches transcripts and emits after silence timeout.
    """

    def __init__(
        self,
        on_batch: Callable[[list[str]], None] | None = None,
        silence_timeout_ms: int = config.LLM_BATCH_SILENCE_MS,
    ):
        self.on_batch = on_batch
        self.silence_timeout_s = silence_timeout_ms / 1000.0

        self._batch: list[str] = []
        self._last_transcript_time: float | None = None
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the batcher thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the batcher."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        with self._lock:
            self._batch.clear()
            self._last_transcript_time = None

    def add_transcript(self, text: str) -> None:
        """Add a transcript to the batch."""
        with self._lock:
            self._batch.append(text)
            self._last_transcript_time = time.time()

    def update_silence_timeout(self, timeout_ms: int) -> None:
        """Update the silence timeout at runtime."""
        with self._lock:
            self.silence_timeout_s = timeout_ms / 1000.0

    def _worker(self) -> None:
        """Batcher worker thread."""
        while not self._stop_event.is_set():
            time.sleep(0.1)

            with self._lock:
                if self._batch and self._last_transcript_time is not None:
                    elapsed = time.time() - self._last_transcript_time
                    if elapsed >= self.silence_timeout_s:
                        batch = self._batch.copy()
                        self._batch.clear()
                        self._last_transcript_time = None

                        if self.on_batch and batch:
                            self.on_batch(batch)
