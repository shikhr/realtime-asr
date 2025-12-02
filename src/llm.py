"""
LLM integration module with transcript batching.
"""

import time
from datetime import datetime
from typing import NoReturn

from . import config
from .queues import batcher_queue, llm_queue, llm_response_queue


def transcript_batcher() -> NoReturn:
    """
    Batch transcripts and send to LLM after silence timeout.

    Collects transcripts as they arrive. After LLM_BATCH_SILENCE_MS of no new
    transcripts, batches them and sends to LLM queue.
    """
    batch: list[str] = []
    last_transcript_time: float | None = None
    silence_timeout_s = config.LLM_BATCH_SILENCE_MS / 1000.0

    while True:
        try:
            # Check for new transcripts with short timeout
            text = batcher_queue.get(timeout=0.1)
            batch.append(text)
            last_transcript_time = time.time()

        except Exception:
            # Timeout or empty - check if we should batch
            pass

        # Check if we have a batch and silence timeout exceeded
        if batch and last_transcript_time is not None:
            silence_elapsed = time.time() - last_transcript_time
            if silence_elapsed >= silence_timeout_s:
                # Send batch to LLM
                try:
                    llm_queue.put_nowait(batch.copy())
                except Exception:
                    pass  # Queue full
                batch.clear()
                last_transcript_time = None


def llm_worker() -> NoReturn:
    """
    Process batched transcripts through LLM (dummy implementation).
    """
    while True:
        batch = llm_queue.get()  # blocking

        # Combine transcripts into a single input
        combined_text = " ".join(batch)

        # Dummy LLM response
        timestamp = datetime.now().strftime("%H:%M:%S")
        response = f'[{timestamp}] User said: "{combined_text}"\n\n🤖 LLM Response: This is a dummy response. Received {len(batch)} utterance(s).'

        # Send response to UI
        try:
            llm_response_queue.put_nowait(response)
        except Exception:
            pass  # Queue full


def process_with_llm(text: str) -> str:
    """
    Process text with LLM (dummy implementation for now).

    Args:
        text: Combined transcript text

    Returns:
        LLM response string
    """
    # TODO: Replace with actual LLM API call
    return f'Acknowledged: "{text}"'
