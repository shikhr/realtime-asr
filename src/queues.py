"""
Shared queues for inter-thread communication.
"""

import queue
from . import config

# Filled by callback: (bytes, ts)
record_queue = queue.Queue(maxsize=config.RECORD_QUEUE_MAX)

# VAD processing
processing_queue = queue.Queue(maxsize=config.PROCESSING_QUEUE_MAX)

# Finalized utterances to ASR
asr_queue = queue.Queue(maxsize=config.ASR_QUEUE_MAX)
