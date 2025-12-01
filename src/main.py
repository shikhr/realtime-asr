#!/usr/bin/env python3
"""
Real-time STT with VAD-based utterance assembly.

- pyaudio callback -> record_queue
- processing worker -> VAD-based utterance assembly with pre-roll + merge wait
- ASR worker -> NeMo Parakeet TDT transcription
"""

import time
import threading

from . import config
from .audio import create_input_stream, terminate
from .workers import forward_to_processing, processing_worker, asr_worker


def main():
    """Main entry point for the STT application."""
    # Pre-load ASR model before starting streams
    print("Initializing ASR engine...")
    from .asr import get_asr_engine

    get_asr_engine()

    # Setup input stream
    input_stream = create_input_stream()

    # Start worker threads
    threads = []

    t1 = threading.Thread(target=forward_to_processing, daemon=True)
    t1.start()
    threads.append(t1)

    t2 = threading.Thread(target=processing_worker, daemon=True)
    t2.start()
    threads.append(t2)

    t3 = threading.Thread(target=asr_worker, daemon=True)
    t3.start()
    threads.append(t3)

    print(
        f"Real-time STT started. VAD-based utterance detection (min {config.UTTERANCE_MIN_MS/1000.0:.1f}s)."
    )
    print("Speak into your microphone. Press Ctrl+C to stop.\n")

    input_stream.start_stream()

    try:
        while input_stream.is_active():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        input_stream.stop_stream()
        input_stream.close()
        terminate()


if __name__ == "__main__":
    main()
