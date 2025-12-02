"""
Runtime configuration that can be modified during execution.
Thread-safe configuration store for tunable parameters.
"""

import threading
from dataclasses import dataclass, field
from typing import Callable

from . import config


@dataclass
class RuntimeConfig:
    """
    Runtime-tunable configuration values.
    These can be changed while the application is running.
    """

    # VAD settings
    vad_aggressiveness: int = config.VAD_AGGRESSIVENESS

    # Utterance assembly
    utterance_end_padding_ms: int = config.UTTERANCE_END_PADDING_MS
    utterance_min_ms: int = config.UTTERANCE_MIN_MS
    utterance_max_ms: int = config.UTTERANCE_MAX_MS

    # Streaming split
    streaming_split_after_ms: int = config.STREAMING_SPLIT_AFTER_MS
    streaming_split_silence_ms: int = config.STREAMING_SPLIT_SILENCE_MS

    # Pre-roll
    pre_roll_ms: int = config.PRE_ROLL_MS

    # LLM batching
    llm_batch_silence_ms: int = config.LLM_BATCH_SILENCE_MS


class ConfigStore:
    """
    Thread-safe configuration store with change notifications.
    """

    def __init__(self):
        self._config = RuntimeConfig()
        self._lock = threading.RLock()
        self._listeners: list[Callable[[RuntimeConfig], None]] = []

    def get(self) -> RuntimeConfig:
        """Get a copy of the current configuration."""
        with self._lock:
            return RuntimeConfig(
                vad_aggressiveness=self._config.vad_aggressiveness,
                utterance_end_padding_ms=self._config.utterance_end_padding_ms,
                utterance_min_ms=self._config.utterance_min_ms,
                utterance_max_ms=self._config.utterance_max_ms,
                streaming_split_after_ms=self._config.streaming_split_after_ms,
                streaming_split_silence_ms=self._config.streaming_split_silence_ms,
                pre_roll_ms=self._config.pre_roll_ms,
                llm_batch_silence_ms=self._config.llm_batch_silence_ms,
            )

    def update(self, **kwargs) -> None:
        """
        Update configuration values.

        Args:
            **kwargs: Configuration fields to update
        """
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)

            # Notify listeners
            config_copy = self.get()
            for listener in self._listeners:
                try:
                    listener(config_copy)
                except Exception:
                    pass

    def add_listener(self, callback: Callable[[RuntimeConfig], None]) -> None:
        """Add a listener for configuration changes."""
        with self._lock:
            self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[RuntimeConfig], None]) -> None:
        """Remove a configuration change listener."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)


# Global config store instance
_config_store: ConfigStore | None = None


def get_config_store() -> ConfigStore:
    """Get the global configuration store."""
    global _config_store
    if _config_store is None:
        _config_store = ConfigStore()
    return _config_store
