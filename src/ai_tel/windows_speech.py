"""Backward-compatible aliases for the old Windows speech module."""

from .openai_speech import MicrophoneRecorder, OpenAISpeechRecognizer

WindowsSpeechRecognizer = OpenAISpeechRecognizer

__all__ = ["MicrophoneRecorder", "OpenAISpeechRecognizer", "WindowsSpeechRecognizer"]