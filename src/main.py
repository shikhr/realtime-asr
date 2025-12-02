#!/usr/bin/env python3
"""
Real-time STT with VAD-based utterance assembly and Gradio UI.

- pyaudio callback -> record_queue
- processing worker -> VAD-based utterance assembly with pre-roll + merge wait
- ASR worker -> NeMo Parakeet TDT transcription
- Gradio UI -> live recording control and transcript display
"""

from .ui import launch


def main():
    """Main entry point for the STT application with Gradio UI."""
    launch()


if __name__ == "__main__":
    main()
