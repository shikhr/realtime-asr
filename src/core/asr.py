"""
ASR module using NVIDIA NeMo Parakeet TDT model.
This module is independent of any transport or UI.
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
            self.model = self.model.half()
            print("ASR model loaded on GPU with FP16.")
        else:
            print("ASR model loaded on CPU.")

    def transcribe_bytes(self, audio_bytes: bytes) -> str:
        """
        Transcribe raw PCM audio bytes (int16, mono) to text.

        Args:
            audio_bytes: Raw PCM audio data (int16, mono)

        Returns:
            Transcribed text string
        """
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
        audio_np /= 32768.0  # Normalize to [-1, 1]
        return self._transcribe(audio_np)

    def transcribe_numpy(self, audio_np: np.ndarray) -> str:
        """
        Transcribe numpy audio array (float32, normalized) to text.

        Args:
            audio_np: Audio as float32 numpy array, normalized to [-1, 1]

        Returns:
            Transcribed text string
        """
        return self._transcribe(audio_np)

    def _transcribe(self, audio_np: np.ndarray) -> str:
        """Internal transcription with NeMo model."""
        with torch.inference_mode():
            output = self.model.transcribe([audio_np])

        if hasattr(output[0], "text"):
            return output[0].text
        if isinstance(output[0], list):
            return " ".join(output[0])
        return str(output[0])


# Global ASR engine instance (lazy loaded)
_asr_engine: ASREngine | None = None


def get_asr_engine() -> ASREngine:
    """Get or create the global ASR engine instance."""
    global _asr_engine
    if _asr_engine is None:
        _asr_engine = ASREngine()
    return _asr_engine
