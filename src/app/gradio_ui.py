"""
Gradio UI for real-time speech-to-text.
"""

import threading
from datetime import datetime

import gradio as gr

from ..core.asr import get_asr_engine
from ..core import config
from .pipeline import STTPipeline, TranscriptBatcher, TranscriptResult
from .llm import LLMProcessor
from ..interfaces.microphone import MicrophoneInput


class STTApp:
    """Real-time STT application with Gradio UI."""

    def __init__(self):
        self.is_recording = False

        # Components
        self._mic: MicrophoneInput | None = None
        self._pipeline: STTPipeline | None = None
        self._batcher: TranscriptBatcher | None = None
        self._llm: LLMProcessor | None = None

        # Runtime config values (used when starting new sessions)
        self.vad_aggressiveness = config.VAD_AGGRESSIVENESS
        self.utterance_end_padding_ms = config.UTTERANCE_END_PADDING_MS
        self.utterance_min_ms = config.UTTERANCE_MIN_MS
        self.streaming_split_after_ms = config.STREAMING_SPLIT_AFTER_MS
        self.streaming_split_silence_ms = config.STREAMING_SPLIT_SILENCE_MS
        self.llm_batch_silence_ms = config.LLM_BATCH_SILENCE_MS

        # Results storage
        self._transcripts: list[tuple[str, str, float]] = (
            []
        )  # (timestamp, text, duration)
        self._llm_responses: list[str] = []
        self._lock = threading.Lock()

    def _on_transcript(self, result: TranscriptResult) -> None:
        """Handle transcript from pipeline."""
        with self._lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            self._transcripts.append((timestamp, result.text, result.duration_s))

        # Add to batcher
        if self._batcher:
            self._batcher.add_transcript(result.text)

    def _on_llm_response(self, response: str) -> None:
        """Handle LLM response."""
        with self._lock:
            self._llm_responses.append(response)

    def _on_batch(self, batch: list[str]) -> None:
        """Handle transcript batch."""
        if self._llm:
            self._llm.process_batch(batch)

    def start_recording(self) -> str:
        """Start the recording and processing pipeline."""
        if self.is_recording:
            return "Already recording..."

        # Clear previous data
        with self._lock:
            self._transcripts.clear()
            self._llm_responses.clear()

        # Create components with current config values
        self._pipeline = STTPipeline(on_transcript=self._on_transcript)
        self._pipeline.update_vad_config(
            aggressiveness=self.vad_aggressiveness,
            end_padding_ms=self.utterance_end_padding_ms,
            min_utterance_ms=self.utterance_min_ms,
            streaming_split_after_ms=self.streaming_split_after_ms,
            streaming_split_silence_ms=self.streaming_split_silence_ms,
        )
        self._pipeline.start()

        self._batcher = TranscriptBatcher(
            on_batch=self._on_batch,
            silence_timeout_ms=self.llm_batch_silence_ms,
        )
        self._batcher.start()

        self._llm = LLMProcessor(on_response=self._on_llm_response)

        self._mic = MicrophoneInput(on_audio=self._pipeline.process_audio)
        self._mic.start()

        self.is_recording = True
        return "🎙️ Recording started..."

    def stop_recording(self) -> str:
        """Stop the recording pipeline."""
        if not self.is_recording:
            return "Not recording."

        self.is_recording = False

        # Stop in reverse order
        if self._mic:
            self._mic.stop()
            self._mic = None

        if self._batcher:
            self._batcher.stop()
            self._batcher = None

        if self._pipeline:
            self._pipeline.stop()
            self._pipeline = None

        self._llm = None

        return "⏹️ Recording stopped."

    def get_transcripts(self) -> str:
        """Get all transcripts as formatted text."""
        with self._lock:
            if not self._transcripts:
                return ""

            lines = [f"[{ts}] {text}" for ts, text, _ in self._transcripts]
            return "\n".join(lines)

    def get_llm_responses(self) -> str:
        """Get all LLM responses."""
        with self._lock:
            if not self._llm_responses:
                return ""
            return "\n\n---\n\n".join(self._llm_responses)

    def get_status(self) -> str:
        """Get current recording status."""
        if self.is_recording:
            return "🔴 Recording..."
        return "⚪ Stopped"

    def update_vad_aggressiveness(self, value: int) -> str:
        """Update VAD aggressiveness."""
        self.vad_aggressiveness = int(value)
        if self._pipeline:
            self._pipeline.update_vad_config(aggressiveness=int(value))
        return f"VAD aggressiveness: {value}"

    def update_end_padding(self, value: int) -> str:
        """Update utterance end padding."""
        self.utterance_end_padding_ms = int(value)
        if self._pipeline:
            self._pipeline.update_vad_config(end_padding_ms=int(value))
        return f"End padding: {value}ms"

    def update_min_utterance(self, value: int) -> str:
        """Update minimum utterance length."""
        self.utterance_min_ms = int(value)
        if self._pipeline:
            self._pipeline.update_vad_config(min_utterance_ms=int(value))
        return f"Min utterance: {value}ms"

    def update_streaming_split_after(self, value: int) -> str:
        """Update streaming split threshold."""
        self.streaming_split_after_ms = int(value)
        if self._pipeline:
            self._pipeline.update_vad_config(streaming_split_after_ms=int(value))
        return f"Split after: {value}ms"

    def update_streaming_split_silence(self, value: int) -> str:
        """Update streaming split silence threshold."""
        self.streaming_split_silence_ms = int(value)
        if self._pipeline:
            self._pipeline.update_vad_config(streaming_split_silence_ms=int(value))
        return f"Split silence: {value}ms"

    def update_llm_batch_silence(self, value: int) -> str:
        """Update LLM batch silence timeout."""
        self.llm_batch_silence_ms = int(value)
        if self._batcher:
            self._batcher.update_silence_timeout(int(value))
        return f"LLM batch timeout: {value}ms"


def create_ui() -> gr.Blocks:
    """Create the Gradio UI."""
    app = STTApp()

    # Pre-load ASR model
    print("Initializing ASR engine...")
    get_asr_engine()
    print("ASR engine ready.")

    with gr.Blocks(title="Real-time Speech-to-Text with LLM") as demo:
        gr.Markdown("# 🎤 Real-time Speech-to-Text with LLM")
        gr.Markdown(
            "Using NVIDIA NeMo Parakeet TDT 0.6B model. "
            "Adjust settings below to tune VAD and batching behavior."
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

        # Settings panel
        with gr.Accordion("⚙️ Settings", open=False):
            gr.Markdown(
                "**Adjust these settings to tune VAD and batching behavior. Changes apply immediately.**"
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### VAD Settings")
                    vad_aggr_slider = gr.Slider(
                        minimum=0,
                        maximum=3,
                        step=1,
                        value=config.VAD_AGGRESSIVENESS,
                        label="VAD Aggressiveness",
                        info="0=least aggressive, 3=most aggressive at filtering non-speech",
                    )
                    end_padding_slider = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        step=50,
                        value=config.UTTERANCE_END_PADDING_MS,
                        label="End Padding (ms)",
                        info="Silence duration before finalizing utterance",
                    )
                    min_utterance_slider = gr.Slider(
                        minimum=100,
                        maximum=2000,
                        step=100,
                        value=config.UTTERANCE_MIN_MS,
                        label="Min Utterance (ms)",
                        info="Minimum utterance duration to accept",
                    )

                with gr.Column(scale=1):
                    gr.Markdown("#### Streaming Settings")
                    split_after_slider = gr.Slider(
                        minimum=500,
                        maximum=5000,
                        step=250,
                        value=config.STREAMING_SPLIT_AFTER_MS,
                        label="Split After (ms)",
                        info="Start looking for split points after this duration",
                    )
                    split_silence_slider = gr.Slider(
                        minimum=50,
                        maximum=500,
                        step=25,
                        value=config.STREAMING_SPLIT_SILENCE_MS,
                        label="Split Silence (ms)",
                        info="Silence duration to trigger split during streaming",
                    )
                    llm_batch_slider = gr.Slider(
                        minimum=500,
                        maximum=5000,
                        step=250,
                        value=config.LLM_BATCH_SILENCE_MS,
                        label="LLM Batch Timeout (ms)",
                        info="Silence duration before batching transcripts for LLM",
                    )

            # Wire up settings
            vad_aggr_slider.change(
                fn=app.update_vad_aggressiveness, inputs=[vad_aggr_slider]
            )
            end_padding_slider.change(
                fn=app.update_end_padding, inputs=[end_padding_slider]
            )
            min_utterance_slider.change(
                fn=app.update_min_utterance, inputs=[min_utterance_slider]
            )
            split_after_slider.change(
                fn=app.update_streaming_split_after, inputs=[split_after_slider]
            )
            split_silence_slider.change(
                fn=app.update_streaming_split_silence, inputs=[split_silence_slider]
            )
            llm_batch_slider.change(
                fn=app.update_llm_batch_silence, inputs=[llm_batch_slider]
            )

        def on_start():
            msg = app.start_recording()
            return msg, "", app.get_status()

        def on_stop():
            app.stop_recording()
            return app.get_transcripts(), app.get_llm_responses(), app.get_status()

        start_btn.click(fn=on_start, outputs=[transcript_box, llm_box, status_text])
        stop_btn.click(fn=on_stop, outputs=[transcript_box, llm_box, status_text])

        def refresh_all():
            return app.get_transcripts(), app.get_llm_responses(), app.get_status()

        timer = gr.Timer(value=0.3, active=True)
        timer.tick(fn=refresh_all, outputs=[transcript_box, llm_box, status_text])

    return demo


def launch():
    """Launch the Gradio UI."""
    demo = create_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
