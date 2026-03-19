"""OpenAI text response helpers for the voice assistant UI."""

from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class OpenAITextResponder:
    """Generate a short assistant reply from user text."""

    model = "gpt-4o-mini"
    max_reply_characters = 100
    knowledge_base_dir_name = "knowledge_base"
    knowledge_base_extensions = ("*.txt", "*.md")

    def generate_reply(
        self,
        user_text: str,
        system_prompt: str | None = None,
        language_hint: str | None = None,
    ) -> dict[str, Any]:
        cleaned_text = user_text.strip()
        if not cleaned_text:
            return self._error("User text must not be empty.")

        knowledge_reply = self._reply_from_knowledge_base(cleaned_text, language_hint)
        if knowledge_reply is not None:
            return knowledge_reply

        api_key = self._get_api_key()
        if not api_key:
            return self._error("OPENAI_API_KEY is not set.")

        try:
            OpenAI = self._load_openai_client_class()
        except Exception as exc:
            return self._error(str(exc))

        client = OpenAI(api_key=api_key)
        messages = [
            {"role": "system", "content": self._build_system_prompt(system_prompt, language_hint)},
            {"role": "user", "content": cleaned_text},
        ]

        try:
            response = client.chat.completions.create(model=self.model, messages=messages)
        except Exception as exc:
            return self._error(f"OpenAI reply generation failed: {exc}")

        text = self._extract_text(response)
        if not text:
            return self._error("OpenAI reply generation returned no text.")

        result = {
            "status": "success",
            "text": self._truncate_text(text, self.max_reply_characters),
            "model": self.model,
            "language_hint": language_hint,
            "source": "openai",
            "timestamp": self._timestamp(),
        }

        usage = self._extract_usage(response)
        if usage is not None:
            result["usage"] = usage

        return result

    def _reply_from_knowledge_base(self, user_text: str, language_hint: str | None) -> dict[str, Any] | None:
        documents = self._load_knowledge_documents()
        if not documents:
            return None

        chunks = self._chunk_documents(documents)
        if not chunks:
            return None

        terms = self._build_query_terms(user_text)
        ranked = []
        for chunk in chunks:
            score = self._score_chunk(chunk["content"], terms)
            if score > 0:
                ranked.append((score, chunk))

        if not ranked:
            return None

        ranked.sort(key=lambda item: item[0], reverse=True)
        top_chunks = [item[1] for item in ranked[:2]]
        summary = self._summarize_knowledge_chunks(top_chunks)
        if not summary:
            return None

        return {
            "status": "success",
            "text": self._truncate_text(summary, self.max_reply_characters),
            "model": "local-knowledge-base",
            "language_hint": language_hint,
            "source": "knowledge_base",
            "knowledge_files": [chunk["path"] for chunk in top_chunks],
            "timestamp": self._timestamp(),
        }

    def _load_knowledge_documents(self) -> list[dict[str, str]]:
        base_dir = Path.cwd() / self.knowledge_base_dir_name
        if not base_dir.exists():
            return []

        documents: list[dict[str, str]] = []
        for pattern in self.knowledge_base_extensions:
            for path in sorted(base_dir.rglob(pattern)):
                try:
                    text = path.read_text(encoding="utf-8").strip()
                except Exception:
                    continue
                if text:
                    documents.append({"path": str(path), "content": text})
        return documents

    def _chunk_documents(self, documents: list[dict[str, str]]) -> list[dict[str, str]]:
        chunks: list[dict[str, str]] = []
        for document in documents:
            parts = [part.strip() for part in re.split(r"\n\s*\n", document["content"]) if part.strip()]
            if not parts:
                parts = [document["content"]]
            for part in parts:
                chunks.append({"path": document["path"], "content": part})
        return chunks

    def _build_system_prompt(self, system_prompt: str | None, language_hint: str | None) -> str:
        base = (
            "You are a helpful voice assistant. Reply clearly, naturally, and briefly. "
            "Keep the reply within 100 characters. "
            "If the user speaks in Japanese, reply in Japanese. If the user speaks in Chinese, reply in Chinese. "
            "Otherwise reply in the same language as the user."
        )
        hint = self._language_instruction(language_hint)
        custom = (system_prompt or "").strip()

        parts = [base]
        if hint:
            parts.append(hint)
        if custom:
            parts.append(f"Additional instruction: {custom}")
        return "\n".join(parts)

    def _language_instruction(self, language_hint: str | None) -> str | None:
        mapping = {
            "ja": "Prefer replying in Japanese unless the user clearly asks for another language.",
            "zh": "Prefer replying in Chinese unless the user clearly asks for another language.",
            "en": "Prefer replying in English unless the user clearly asks for another language.",
            "ko": "Prefer replying in Korean unless the user clearly asks for another language.",
            "fr": "Prefer replying in French unless the user clearly asks for another language.",
        }
        return mapping.get((language_hint or "").strip().lower())

    def _build_query_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        lowered = text.lower()

        for term in re.findall(r"[a-z0-9]{2,}", lowered):
            if term not in terms:
                terms.append(term)

        cjk_sequences = re.findall(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]{2,}", text)
        for sequence in cjk_sequences:
            max_size = min(4, len(sequence))
            for size in range(max_size, 1, -1):
                for index in range(0, len(sequence) - size + 1):
                    token = sequence[index:index + size]
                    if token not in terms:
                        terms.append(token)
                    if len(terms) >= 16:
                        return terms
        return terms

    def _score_chunk(self, chunk: str, terms: list[str]) -> int:
        haystack = chunk.lower()
        score = 0
        for term in terms:
            if term.lower() in haystack:
                score += max(1, len(term) - 1)
        return score

    def _summarize_knowledge_chunks(self, chunks: list[dict[str, str]]) -> str:
        texts = []
        for chunk in chunks:
            content = re.sub(r"\s+", " ", chunk["content"]).strip()
            if content and content not in texts:
                texts.append(content)

        if not texts:
            return ""

        return " ".join(texts)

    def _get_api_key(self) -> str | None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            return api_key

        env_path = Path.cwd() / ".env"
        if not env_path.exists():
            return None

        try:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                if key.strip() == "OPENAI_API_KEY":
                    parsed_value = value.strip().strip('"').strip("'")
                    if parsed_value:
                        os.environ["OPENAI_API_KEY"] = parsed_value
                        return parsed_value
        except Exception:
            return None

        return None

    def _load_openai_client_class(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Install the project requirements to enable reply generation."
            ) from exc
        return OpenAI

    def _extract_text(self, response: Any) -> str | None:
        if hasattr(response, "choices"):
            choices = getattr(response, "choices")
            if choices:
                first = choices[0]
                message = getattr(first, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if isinstance(content, str):
                        return content.strip()
        if isinstance(response, dict):
            choices = response.get("choices")
            if isinstance(choices, list) and choices:
                message = choices[0].get("message", {})
                content = message.get("content")
                if isinstance(content, str):
                    return content.strip()
        return None

    @staticmethod
    def _truncate_text(text: str, limit: int) -> str:
        cleaned = text.strip()
        if limit <= 0 or len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip()

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        usage = getattr(response, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                return usage.model_dump()
            if isinstance(usage, dict):
                return usage
        if isinstance(response, dict):
            maybe_usage = response.get("usage")
            if isinstance(maybe_usage, dict):
                return maybe_usage
        return None

    def _error(self, message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "timestamp": self._timestamp(),
        }

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")
