"""
ASR module using NVIDIA NeMo Parakeet TDT model.
"""

import numpy as np
import torch
import nemo.collections.asr as nemo_asr


class ASREngine:
    """ASR engine using NeMo Parakeet TDT model."""

    def __init__(self, model_name: str = "nvidia/parakeet-tdt-0.6b-v2"):
        print(f"Loading ASR model: {model_name}...")
        self.model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

        # Optimize for inference
        self.model.eval()

        # Use GPU with half-precision if available
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            # Enable half-precision for faster inference on GPU
            self.model = self.model.half()
            print(f"ASR model loaded on GPU with FP16.")
        else:
            print("ASR model loaded on CPU.")

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw PCM audio bytes (int16, mono, 16kHz) to text.

        Args:
            audio_bytes: Raw PCM audio data (int16, mono, 16kHz)

        Returns:
            Transcribed text string
        """
        # Convert bytes to numpy array
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        # Normalize to [-1, 1] range
        audio_np = audio_np / 32768.0

        # Transcribe using NeMo with inference mode for speed
        with torch.inference_mode():
            output = self.model.transcribe([audio_np])
        # Extract text from output
        if hasattr(output[0], "text"):
            return output[0].text
        elif isinstance(output[0], list):
            return " ".join(output[0])
        else:
            return str(output[0])


# Global ASR engine instance (lazy loaded)
_asr_engine = None


def get_asr_engine() -> ASREngine:
    """Get or create the global ASR engine instance."""
    global _asr_engine
    if _asr_engine is None:
        _asr_engine = ASREngine()
    return _asr_engine


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe audio bytes to text using the global ASR engine."""
    return get_asr_engine().transcribe_bytes(audio_bytes)
