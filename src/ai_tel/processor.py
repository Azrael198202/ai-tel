"""Core text processing primitives for the ai-tel project."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

from langdetect import DetectorFactory, LangDetectException, detect, detect_langs
from textblob import TextBlob

DetectorFactory.seed = 0


class AITextProcessor:
    """Provide language detection, text analysis, and template-based generation."""

    SUPPORTED_LANGUAGES = {
        "zh-cn": "Simplified Chinese",
        "zh-tw": "Traditional Chinese",
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
        "ru": "Russian",
        "ar": "Arabic",
    }

    GENERATION_TEMPLATES = {
        "english": (
            "Regarding {prompt}, there are several useful angles to consider. "
            "{prompt} matters because it connects practical decisions, context, and outcomes."
        ),
        "chinese": (
            "About {prompt}, we can understand it from several angles. "
            "{prompt} matters because it is closely tied to real-world scenarios, decisions, and outcomes."
        ),
        "japanese": (
            "About {prompt}, we can organize the topic from several perspectives. "
            "{prompt} matters because it is closely connected to practical decisions, context, and outcomes."
        ),
    }

    _SENTENCE_SPLIT_RE = re.compile(r"[.!?。！？]+")
    _WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

    def __init__(self) -> None:
        self.processing_history: list[dict[str, Any]] = []

    def detect_language(self, text: str) -> dict[str, Any]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return self._error("Text must not be empty.")

        try:
            language_code = detect(cleaned_text)
            probabilities = detect_langs(cleaned_text)
        except LangDetectException as exc:
            return self._error(str(exc))

        confidence = max(prob.prob for prob in probabilities)
        result = {
            "status": "success",
            "language_code": language_code,
            "language_name": self.SUPPORTED_LANGUAGES.get(language_code, language_code),
            "confidence": round(confidence, 4),
            "timestamp": self._timestamp(),
        }
        self.processing_history.append(result)
        return result

    def analyze_text(self, text: str) -> dict[str, Any]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return self._error("Text must not be empty.")

        try:
            blob = TextBlob(cleaned_text)
            language_info = self.detect_language(cleaned_text)
            sentence_count = self._count_sentences(cleaned_text)
            word_count = self._count_words(cleaned_text)
            result = {
                "status": "success",
                "text": cleaned_text,
                "language": language_info.get("language_name", "Unknown"),
                "sentences_count": sentence_count,
                "words_count": word_count,
                "polarity": round(blob.sentiment.polarity, 4),
                "subjectivity": round(blob.sentiment.subjectivity, 4),
                "timestamp": self._timestamp(),
            }
        except Exception as exc:
            return self._error(str(exc))

        self.processing_history.append(result)
        return result

    def generate_text(self, prompt: str, language: str = "english", length: int = 100) -> dict[str, Any]:
        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            return self._error("Prompt must not be empty.")
        if length <= 0:
            return self._error("Length must be greater than zero.")

        template = self.GENERATION_TEMPLATES.get(language.lower(), self.GENERATION_TEMPLATES["english"])
        generated_text = template.format(prompt=cleaned_prompt)
        if len(generated_text) > length:
            generated_text = generated_text[: max(0, length - 3)].rstrip() + "..."

        result = {
            "status": "success",
            "prompt": cleaned_prompt,
            "language": language,
            "generated_text": generated_text,
            "text_length": len(generated_text),
            "timestamp": self._timestamp(),
        }
        self.processing_history.append(result)
        return result

    def get_history(self) -> list[dict[str, Any]]:
        return list(self.processing_history)

    def clear_history(self) -> None:
        self.processing_history = []

    def _count_sentences(self, text: str) -> int:
        parts = [part for part in self._SENTENCE_SPLIT_RE.split(text) if part.strip()]
        return max(1, len(parts))

    def _count_words(self, text: str) -> int:
        return len(self._WORD_RE.findall(text))

    def _error(self, message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "timestamp": self._timestamp(),
        }

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")