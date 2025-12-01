"""
Utterance handling: enqueueing finalized utterances for ASR.
"""

import queue

from . import config
from .queues import asr_queue


def finalize_and_enqueue_utterance(
    utter_bytes: bytes, start_ts: float, end_ts: float
) -> None:
    """Enqueue utterance for ASR processing (if longer than min duration)."""
    duration_ms = (end_ts - start_ts) * 1000.0 if (start_ts and end_ts) else 0.0
    if duration_ms < config.UTTERANCE_MIN_MS:
        return

    # Enqueue for ASR (non-blocking)
    try:
        asr_queue.put_nowait((utter_bytes, start_ts, end_ts))
    except queue.Full:
        try:
            asr_queue.get_nowait()
            asr_queue.put_nowait((utter_bytes, start_ts, end_ts))
        except Exception:
            print("Dropped utterance due to full ASR queue")
