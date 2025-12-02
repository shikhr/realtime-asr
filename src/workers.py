"""
Worker threads for audio processing pipeline.
"""

import math
import time
import queue
from collections import deque
from typing import NoReturn

import webrtcvad

from . import config
from .queues import (
    record_queue,
    processing_queue,
    asr_queue,
    ui_queue,
    batcher_queue,
)
from .utils import frame_generator
from .utterance import finalize_and_enqueue_utterance
from .asr import get_asr_engine


def forward_to_processing() -> NoReturn:
    """Forward audio chunks from record queue to processing queue."""
    while True:
        chunk_bytes, ts = record_queue.get()
        # Forward to processing queue
        try:
            processing_queue.put_nowait((chunk_bytes, ts))
        except queue.Full:
            try:
                processing_queue.get_nowait()
                processing_queue.put_nowait((chunk_bytes, ts))
            except Exception:
                pass


def processing_worker() -> NoReturn:
    """VAD-based utterance assembly with pre-roll and merge wait."""
    vad = webrtcvad.Vad(int(config.VAD_AGGRESSIVENESS))
    pre_roll_buf = deque(
        maxlen=max(1, int(math.ceil(config.PRE_ROLL_MS / config.VAD_FRAME_MS)))
    )

    collecting = False
    utterance_bytes = bytearray()
    utter_start_ts = None
    last_speech_ts = None
    accumulated_ms = 0
    frame_duration_ms = config.VAD_FRAME_MS

    while True:
        chunk_bytes, chunk_ts = processing_queue.get()  # blocking
        frames = list(frame_generator(frame_duration_ms, chunk_bytes, config.RATE))
        if not frames:
            continue

        # Push frames into pre-roll as we see them
        for i, frame in enumerate(frames):
            frame_ts = chunk_ts + (i * frame_duration_ms) / 1000.0
            pre_roll_buf.append((frame, frame_ts))

            try:
                is_speech = vad.is_speech(frame, config.RATE)
            except Exception:
                is_speech = False

            if is_speech:
                if not collecting:
                    # Start new utterance, include pre-roll frames
                    collecting = True
                    utterance_bytes = bytearray()
                    # Prepend pre-roll frames in order
                    for pr_frame, _pr_ts in pre_roll_buf:
                        utterance_bytes.extend(pr_frame)
                    utter_start_ts = (
                        pre_roll_buf[0][1] if len(pre_roll_buf) > 0 else frame_ts
                    )
                    accumulated_ms = 0
                    last_speech_ts = frame_ts

                # Append current speech frame
                utterance_bytes.extend(frame)
                accumulated_ms += frame_duration_ms
                last_speech_ts = frame_ts

                # Guard max utterance length
                if accumulated_ms >= config.UTTERANCE_MAX_MS:
                    finalize_and_enqueue_utterance(
                        bytes(utterance_bytes), utter_start_ts, frame_ts
                    )
                    collecting = False
                    utterance_bytes = bytearray()
                    utter_start_ts = None
                    last_speech_ts = None
                    accumulated_ms = 0

            else:
                # Silence frame
                if collecting:
                    utterance_bytes.extend(frame)
                    accumulated_ms += frame_duration_ms
                    if last_speech_ts is not None:
                        silence_ms = (frame_ts - last_speech_ts) * 1000.0

                        # Determine split threshold based on utterance length
                        # After STREAMING_SPLIT_AFTER_MS, use shorter silence threshold
                        if accumulated_ms >= config.STREAMING_SPLIT_AFTER_MS:
                            split_silence_threshold = config.STREAMING_SPLIT_SILENCE_MS
                        else:
                            split_silence_threshold = config.UTTERANCE_END_PADDING_MS

                        if silence_ms >= split_silence_threshold:
                            # Candidate finalize
                            if accumulated_ms >= config.UTTERANCE_MIN_MS:
                                # Good length, finalize
                                finalize_and_enqueue_utterance(
                                    bytes(utterance_bytes), utter_start_ts, frame_ts
                                )
                                collecting = False
                                utterance_bytes = bytearray()
                                utter_start_ts = None
                                last_speech_ts = None
                                accumulated_ms = 0
                            else:
                                # Too short: wait for more speech up to MERGE_WAIT_MS
                                wait_deadline = time.time() + (
                                    config.MERGE_WAIT_MS / 1000.0
                                )
                                got_more_speech = False
                                # Transient local buffer of frames consumed during wait
                                while time.time() < wait_deadline:
                                    try:
                                        nxt_chunk_bytes, nxt_chunk_ts = (
                                            processing_queue.get(timeout=0.05)
                                        )
                                    except queue.Empty:
                                        continue
                                    nxt_frames = list(
                                        frame_generator(
                                            frame_duration_ms,
                                            nxt_chunk_bytes,
                                            config.RATE,
                                        )
                                    )
                                    if not nxt_frames:
                                        continue
                                    for j, nxt_frame in enumerate(nxt_frames):
                                        nxt_frame_ts = (
                                            nxt_chunk_ts
                                            + (j * frame_duration_ms) / 1000.0
                                        )
                                        pre_roll_buf.append((nxt_frame, nxt_frame_ts))
                                        try:
                                            nxt_is_speech = vad.is_speech(
                                                nxt_frame, config.RATE
                                            )
                                        except Exception:
                                            nxt_is_speech = False

                                        if nxt_is_speech:
                                            got_more_speech = True
                                            utterance_bytes.extend(nxt_frame)
                                            accumulated_ms += frame_duration_ms
                                            last_speech_ts = nxt_frame_ts
                                        else:
                                            # Silence frame during wait: append padding
                                            utterance_bytes.extend(nxt_frame)
                                            accumulated_ms += frame_duration_ms

                                    if got_more_speech:
                                        break

                                if got_more_speech:
                                    # Continue collecting normally
                                    continue
                                else:
                                    # Drop short utterance (no additional speech)
                                    collecting = False
                                    utterance_bytes = bytearray()
                                    utter_start_ts = None
                                    last_speech_ts = None
                                    accumulated_ms = 0


def asr_worker() -> NoReturn:
    """ASR worker that transcribes utterances using NeMo Parakeet model."""
    # Initialize ASR engine (loads model on first call)
    asr_engine = get_asr_engine()

    while True:
        utter_bytes, start_ts, end_ts = asr_queue.get()
        duration_s = end_ts - start_ts if (start_ts and end_ts) else 0.0

        try:
            # Transcribe using NeMo
            text = asr_engine.transcribe_bytes(utter_bytes)
            # Send to UI queue
            try:
                ui_queue.put_nowait((text, duration_s))
            except Exception:
                pass  # UI queue full, drop update
            # Send to batcher queue for LLM batching
            try:
                batcher_queue.put_nowait(text)
            except Exception:
                pass  # Batcher queue full, drop
        except Exception:
            pass  # Silently handle ASR errors
