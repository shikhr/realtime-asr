"""
LLM integration module.
"""

from datetime import datetime
from typing import Callable


class LLMProcessor:
    """
    LLM processor for handling batched transcripts.
    Currently a dummy implementation.
    """

    def __init__(
        self,
        on_response: Callable[[str], None] | None = None,
    ):
        self.on_response = on_response

    def process_batch(self, transcripts: list[str]) -> str:
        """
        Process a batch of transcripts through LLM.

        Args:
            transcripts: List of transcript strings

        Returns:
            LLM response string
        """
        combined_text = " ".join(transcripts)

        # Dummy LLM response
        timestamp = datetime.now().strftime("%H:%M:%S")
        response = (
            f'[{timestamp}] User said: "{combined_text}"\n\n'
            f"🤖 LLM Response: This is a dummy response. "
            f"Received {len(transcripts)} utterance(s)."
        )

        if self.on_response:
            self.on_response(response)

        return response
