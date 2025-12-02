#!/usr/bin/env python3
"""
Real-time STT with VAD-based utterance assembly and Gradio UI.
"""

from .app.gradio_ui import launch


def main():
    """Main entry point for the STT application with Gradio UI."""
    launch()


if __name__ == "__main__":
    main()
