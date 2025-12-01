"""
Configuration constants for the real-time STT system.
"""

import pyaudio

# -------------------------
# AUDIO CONFIG
# -------------------------
RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16

# chunking
CHUNK_SECONDS = 0.1  # 100 ms chunks provided by callback
FRAMES_PER_BUFFER = int(RATE * CHUNK_SECONDS)

# -------------------------
# VAD CONFIG (webrtcvad)
# -------------------------
VAD_AGGRESSIVENESS = 2  # 0..3
VAD_FRAME_MS = 20  # must be 10, 20, or 30

# -------------------------
# UTTERANCE ASSEMBLY TUNING
# -------------------------
UTTERANCE_END_PADDING_MS = (
    300  # hangover after last speech frame before candidate finalize
)
UTTERANCE_MIN_MS = 500  # minimum duration to accept (0.5s) - lowered for streaming
UTTERANCE_MAX_MS = 30_000  # cap utterance at 30s

# Streaming split: after this duration, split at shorter pauses for real-time output
STREAMING_SPLIT_AFTER_MS = 2000  # after 2s of speech, look for split points
STREAMING_SPLIT_SILENCE_MS = 150  # split on 150ms pause when streaming

# merge short bursts behavior
MERGE_WAIT_MS = 500  # wait up to this for more speech to merge into short utterance
PRE_ROLL_MS = 200  # prepend ~200 ms before VAD start to avoid chopping

# -------------------------
# QUEUE SIZES
# -------------------------
RECORD_QUEUE_MAX = 800
PROCESSING_QUEUE_MAX = 800
ASR_QUEUE_MAX = 50
