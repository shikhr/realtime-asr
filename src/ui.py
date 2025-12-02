"""
Gradio UI for real-time speech-to-text.
"""

import threading
import time
from datetime import datetime
from typing import Generator

import gradio as gr
import numpy as np

from . import config
from .audio import create_input_stream, terminate, pa
from .queues import (
    ui_queue,
    batcher_queue,
    llm_response_queue,
    stop_event,
    clear_all_queues,
)
from .workers import forward_to_processing, processing_worker, asr_worker
from .llm import transcript_batcher, llm_worker
from .asr import get_asr_engine


class STTApp:
    """Real-time STT application with Gradio UI."""

    def __init__(self):
        self.is_recording = False
        self.input_stream = None
        self.threads: list[threading.Thread] = []
        self.transcripts: list[tuple[str, str, float]] = (
            []
        )  # (timestamp, text, duration)
        self.llm_responses: list[str] = []

    def start_recording(self) -> str:
        """Start the recording and processing pipeline."""
        if self.is_recording:
            return "Already recording..."

        # Clear previous data
        self.transcripts.clear()
        self.llm_responses.clear()

        # Clear stop event and all queues
        stop_event.clear()
        clear_all_queues()

        # Create input stream
        self.input_stream = create_input_stream()

        # Start worker threads
        self.threads = []

        t1 = threading.Thread(target=forward_to_processing, daemon=True)
        t1.start()
        self.threads.append(t1)

        t2 = threading.Thread(target=processing_worker, daemon=True)
        t2.start()
        self.threads.append(t2)

        t3 = threading.Thread(target=asr_worker, daemon=True)
        t3.start()
        self.threads.append(t3)

        # Start LLM workers
        t4 = threading.Thread(target=transcript_batcher, daemon=True)
        t4.start()
        self.threads.append(t4)

        t5 = threading.Thread(target=llm_worker, daemon=True)
        t5.start()
        self.threads.append(t5)

        # Start stream
        self.input_stream.start_stream()
        self.is_recording = True

        return "🎙️ Recording started..."

    def stop_recording(self) -> str:
        """Stop the recording pipeline."""
        if not self.is_recording:
            return "Not recording."

        self.is_recording = False

        # Signal workers to stop
        stop_event.set()

        # Stop audio stream
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
            self.input_stream = None

        # Wait for threads to finish (with timeout)
        for t in self.threads:
            t.join(timeout=0.5)
        self.threads.clear()

        # Clear all queues
        clear_all_queues()

        return "⏹️ Recording stopped."

    def get_transcripts(self) -> str:
        """Get all transcripts as formatted text."""
        # Check for new transcripts from queue
        while True:
            try:
                text, duration = ui_queue.get_nowait()
                timestamp = datetime.now().strftime("%H:%M:%S")
                self.transcripts.append((timestamp, text, duration))
            except Exception:
                break

        if not self.transcripts:
            return ""

        # Format transcripts - show each on new line with timestamp
        lines = []
        for ts, text, dur in self.transcripts:
            lines.append(f"[{ts}] {text}")
        return "\n".join(lines)

    def get_llm_responses(self) -> str:
        """Get all LLM responses."""
        # Check for new LLM responses
        while True:
            try:
                response = llm_response_queue.get_nowait()
                self.llm_responses.append(response)
            except Exception:
                break

        if not self.llm_responses:
            return ""

        return "\n\n---\n\n".join(self.llm_responses)

    def get_status(self) -> str:
        """Get current recording status."""
        if self.is_recording:
            return "🔴 Recording..."
        return "⚪ Stopped"


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    app = STTApp()

    # Pre-load ASR model
    print("Initializing ASR engine...")
    get_asr_engine()
    print("ASR engine ready.")

    with gr.Blocks(
        title="Real-time Speech-to-Text with LLM",
    ) as demo:
        gr.Markdown("# 🎤 Real-time Speech-to-Text with LLM")
        gr.Markdown(
            "Using NVIDIA NeMo Parakeet TDT 0.6B model. "
            f"Transcripts are batched after {config.LLM_BATCH_SILENCE_MS}ms of silence and sent to LLM."
        )

        with gr.Row():
            status_text = gr.Textbox(
                label="Status",
                value="⚪ Stopped",
                interactive=False,
                lines=1,
                scale=1,
            )
            start_btn = gr.Button("🎙️ Start Recording", variant="primary", size="lg")
            stop_btn = gr.Button("⏹️ Stop Recording", variant="stop", size="lg")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Live Transcripts")
                transcript_box = gr.Textbox(
                    label="Transcripts",
                    placeholder="Speak into your microphone...",
                    lines=12,
                    max_lines=15,
                    interactive=False,
                    autoscroll=True,
                )

            with gr.Column(scale=1):
                gr.Markdown("### 🤖 LLM Responses")
                llm_box = gr.Textbox(
                    label="LLM Output",
                    placeholder="LLM responses will appear here after silence...",
                    lines=12,
                    max_lines=15,
                    interactive=False,
                    autoscroll=True,
                )

        # Button handlers
        def on_start():
            msg = app.start_recording()
            return msg, "", app.get_status()

        def on_stop():
            msg = app.stop_recording()
            return app.get_transcripts(), app.get_llm_responses(), app.get_status()

        start_btn.click(
            fn=on_start,
            outputs=[transcript_box, llm_box, status_text],
        )

        stop_btn.click(
            fn=on_stop,
            outputs=[transcript_box, llm_box, status_text],
        )

        # Auto-refresh both transcripts and LLM responses
        def refresh_all():
            return app.get_transcripts(), app.get_llm_responses(), app.get_status()

        # Use a timer to poll for updates
        timer = gr.Timer(value=0.3, active=True)
        timer.tick(
            fn=refresh_all,
            outputs=[transcript_box, llm_box, status_text],
        )

    return demo


def launch():
    """Launch the Gradio UI."""
    demo = create_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
