"""
Shared queues for inter-thread communication.
"""

import queue
from . import config

# Filled by callback: (bytes, ts)
record_queue: queue.Queue[tuple[bytes, float]] = queue.Queue(
    maxsize=config.RECORD_QUEUE_MAX
)

# VAD processing
processing_queue: queue.Queue[tuple[bytes, float]] = queue.Queue(
    maxsize=config.PROCESSING_QUEUE_MAX
)

# Finalized utterances to ASR
asr_queue: queue.Queue[tuple[bytes, float, float]] = queue.Queue(
    maxsize=config.ASR_QUEUE_MAX
)

# UI updates (transcript text, duration)
ui_queue: queue.Queue[tuple[str, float]] = queue.Queue(maxsize=100)

# Transcript batching queue (text only, for LLM batching)
batcher_queue: queue.Queue[str] = queue.Queue(maxsize=100)

# LLM batched input (list of transcript strings)
llm_queue: queue.Queue[list[str]] = queue.Queue(maxsize=50)

# LLM responses for UI
llm_response_queue: queue.Queue[str] = queue.Queue(maxsize=50)
