"""
Core configuration constants for audio processing.
These are transport-agnostic settings.
"""

# -------------------------
# AUDIO CONFIG
# -------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (int16)

# -------------------------
# VAD CONFIG (webrtcvad)
# -------------------------
VAD_AGGRESSIVENESS = 2  # 0..3
VAD_FRAME_MS = 20  # must be 10, 20, or 30

# -------------------------
# UTTERANCE ASSEMBLY TUNING
# -------------------------
UTTERANCE_END_PADDING_MS = 300  # hangover after last speech frame before finalize
UTTERANCE_MIN_MS = 500  # minimum duration to accept
UTTERANCE_MAX_MS = 30_000  # cap utterance at 30s

# Streaming split: after this duration, split at shorter pauses
STREAMING_SPLIT_AFTER_MS = 2000  # after 2s of speech, look for split points
STREAMING_SPLIT_SILENCE_MS = 150  # split on 150ms pause when streaming

# merge short bursts behavior
MERGE_WAIT_MS = 500  # wait for more speech to merge into short utterance
PRE_ROLL_MS = 200  # prepend ~200 ms before VAD start

# -------------------------
# LLM BATCHING
# -------------------------
LLM_BATCH_SILENCE_MS = 2000  # batch transcripts after 2s of silence
