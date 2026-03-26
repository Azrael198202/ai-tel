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
        # Pass the recent turns from the current session so the reply can build on prior questions.
        conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        """Generate reply.
        
        Args:
            user_text: Text spoken by the user.
            system_prompt: Optional system instruction for the model.
            language_hint: Optional language hint for the current request.
            conversation_history: Recent conversation messages used as context.
        
        Returns:
            The dict[str, Any] value produced by this callable.
        """
        cleaned_text = user_text.strip()
        if not cleaned_text:
            return self._error("User text must not be empty.")

        knowledge_chunks = self._find_relevant_knowledge_chunks(cleaned_text)

        # When knowledge matches, let the model rewrite it into a natural reply instead of echoing the source text.
        api_key = self._get_api_key()
        if knowledge_chunks and api_key:
            try:
                OpenAI = self._load_openai_client_class()
                client = OpenAI(api_key=api_key)
                knowledge_reply = self._reply_from_knowledge_base(
                    client=client,
                    user_text=cleaned_text,
                    knowledge_chunks=knowledge_chunks,
                    system_prompt=system_prompt,
                    language_hint=language_hint,
                    conversation_history=conversation_history,
                )
                if knowledge_reply is not None:
                    return knowledge_reply
            except Exception:
                pass
        elif knowledge_chunks:
            knowledge_reply = self._fallback_knowledge_reply(knowledge_chunks, language_hint)
            if knowledge_reply is not None:
                return knowledge_reply

        if not api_key:
            return self._error("OPENAI_API_KEY is not set.")

        try:
            OpenAI = self._load_openai_client_class()
        except Exception as exc:
            return self._error(str(exc))

        client = OpenAI(api_key=api_key)
        messages = [{"role": "system", "content": self._build_system_prompt(system_prompt, language_hint)}]
        # Keep a short rolling transcript so the assistant can carry context across turns.
        messages.extend(self._sanitize_conversation_history(conversation_history))
        messages.append({"role": "user", "content": cleaned_text})

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

    def _find_relevant_knowledge_chunks(self, user_text: str) -> list[dict[str, str]]:
        # Keep only the top matches so the knowledge prompt stays focused and concise.
        """Find relevant knowledge chunks.
        
        Args:
            user_text: Text spoken by the user.
        
        Returns:
            The list[dict[str, str]] value produced by this callable.
        """
        documents = self._load_knowledge_documents()
        if not documents:
            return []

        chunks = self._chunk_documents(documents)
        if not chunks:
            return []

        terms = self._build_query_terms(user_text)
        ranked = []
        for chunk in chunks:
            score = self._score_chunk(chunk["content"], terms)
            if score > 0:
                ranked.append((score, chunk))

        if not ranked:
            return []

        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:2]]

    def _reply_from_knowledge_base(
        self,
        client,
        user_text: str,
        # These are the ranked knowledge snippets that the model should rewrite into a natural answer.
        knowledge_chunks: list[dict[str, str]],
        system_prompt: str | None,
        language_hint: str | None,
        conversation_history: list[dict[str, str]] | None,
    ) -> dict[str, Any] | None:
        """Generate from knowledge base.
        
        Args:
            client: OpenAI client instance used for API calls.
            user_text: Text spoken by the user.
            knowledge_chunks: Relevant knowledge chunks for reply generation.
            system_prompt: Optional system instruction for the model.
            language_hint: Optional language hint for the current request.
            conversation_history: Recent conversation messages used as context.
        
        Returns:
            The dict[str, Any] | None value produced by this callable.
        """
        knowledge_context = self._summarize_knowledge_chunks(knowledge_chunks)
        if not knowledge_context:
            return None

        messages = [
            {
                "role": "system",
                "content": self._build_knowledge_system_prompt(system_prompt, language_hint),
            }
        ]
        messages.extend(self._sanitize_conversation_history(conversation_history))
        messages.append(
            {
                "role": "user",
                "content": (
                    f"User question: {user_text}\n\n"
                    "Reference information:\n"
                    f"{knowledge_context}\n\n"
                    "Please answer naturally based on the reference information."
                ),
            }
        )

        response = client.chat.completions.create(model=self.model, messages=messages)
        text = self._extract_text(response)
        if not text:
            return None

        result = {
            "status": "success",
            "text": self._truncate_text(text, self.max_reply_characters),
            "model": self.model,
            "language_hint": language_hint,
            "source": "knowledge_base",
            "knowledge_files": [chunk["path"] for chunk in knowledge_chunks],
            "timestamp": self._timestamp(),
        }

        usage = self._extract_usage(response)
        if usage is not None:
            result["usage"] = usage

        return result

    def _fallback_knowledge_reply(self, knowledge_chunks: list[dict[str, str]], language_hint: str | None) -> dict[str, Any] | None:
        # Fall back to a local summary only when the API is unavailable.
        """Generate knowledge reply.
        
        Args:
            knowledge_chunks: Relevant knowledge chunks for reply generation.
            language_hint: Optional language hint for the current request.
        
        Returns:
            The dict[str, Any] | None value produced by this callable.
        """
        summary = self._summarize_knowledge_chunks(knowledge_chunks)
        if not summary:
            return None

        return {
            "status": "success",
            "text": self._truncate_text(summary, self.max_reply_characters),
            "model": "local-knowledge-base",
            "language_hint": language_hint,
            "source": "knowledge_base",
            "knowledge_files": [chunk["path"] for chunk in knowledge_chunks],
            "timestamp": self._timestamp(),
        }

    def _load_knowledge_documents(self) -> list[dict[str, str]]:
        """Load knowledge documents.
        
        Args:
            None.
        
        Returns:
            The list[dict[str, str]] value produced by this callable.
        """
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
        """Chunk documents.
        
        Args:
            documents: Knowledge-base documents to process.
        
        Returns:
            The list[dict[str, str]] value produced by this callable.
        """
        chunks: list[dict[str, str]] = []
        for document in documents:
            parts = [part.strip() for part in re.split(r"\n\s*\n", document["content"]) if part.strip()]
            if not parts:
                parts = [document["content"]]
            for part in parts:
                chunks.append({"path": document["path"], "content": part})
        return chunks

    def _sanitize_conversation_history(self, conversation_history: list[dict[str, str]] | None) -> list[dict[str, str]]:
        """Sanitize conversation history.
        
        Args:
            conversation_history: Recent conversation messages used as context.
        
        Returns:
            The list[dict[str, str]] value produced by this callable.
        """
        if not conversation_history:
            return []

        sanitized: list[dict[str, str]] = []
        for message in conversation_history:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role not in {"user", "assistant"} or not content:
                continue
            sanitized.append({"role": role, "content": content})
        return sanitized

    def _build_knowledge_system_prompt(self, system_prompt: str | None, language_hint: str | None) -> str:
        """Build knowledge system prompt.
        
        Args:
            system_prompt: Optional system instruction for the model.
            language_hint: Optional language hint for the current request.
        
        Returns:
            The str value produced by this callable.
        """
        base = (
            "You are a helpful voice assistant. Use the provided reference information, but do not read it verbatim. "
            "Answer in a natural, human, conversational way with clear logic. "
            "Keep the reply within 100 characters when possible. "
            "If the reference is incomplete, say so briefly instead of making things up. "
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

    def _build_system_prompt(self, system_prompt: str | None, language_hint: str | None) -> str:
        """Build system prompt.
        
        Args:
            system_prompt: Optional system instruction for the model.
            language_hint: Optional language hint for the current request.
        
        Returns:
            The str value produced by this callable.
        """
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
        """Language instruction.
        
        Args:
            language_hint: Optional language hint for the current request.
        
        Returns:
            The str | None value produced by this callable.
        """
        mapping = {
            "ja": "Prefer replying in Japanese unless the user clearly asks for another language.",
            "zh": "Prefer replying in Chinese unless the user clearly asks for another language.",
            "en": "Prefer replying in English unless the user clearly asks for another language.",
            "ko": "Prefer replying in Korean unless the user clearly asks for another language.",
            "fr": "Prefer replying in French unless the user clearly asks for another language.",
        }
        return mapping.get((language_hint or "").strip().lower())

    def _build_query_terms(self, text: str) -> list[str]:
        """Build query terms.
        
        Args:
            text: Input text handled by the current operation.
        
        Returns:
            The list[str] value produced by this callable.
        """
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
        """Score chunk.
        
        Args:
            chunk: Audio chunk used for level analysis.
            terms: Search terms derived from input text.
        
        Returns:
            The int value produced by this callable.
        """
        haystack = chunk.lower()
        score = 0
        for term in terms:
            if term.lower() in haystack:
                score += max(1, len(term) - 1)
        return score

    def _summarize_knowledge_chunks(self, chunks: list[dict[str, str]]) -> str:
        """Summarize knowledge chunks.
        
        Args:
            chunks: Knowledge-base chunks to summarize or inspect.
        
        Returns:
            The str value produced by this callable.
        """
        texts = []
        for chunk in chunks:
            content = re.sub(r"\s+", " ", chunk["content"]).strip()
            if content and content not in texts:
                texts.append(content)

        if not texts:
            return ""

        return " ".join(texts)

    def _get_api_key(self) -> str | None:
        """Get api key.
        
        Args:
            None.
        
        Returns:
            The str | None value produced by this callable.
        """
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
        """Load openai client class.
        
        Args:
            None.
        
        Returns:
            The result produced by this callable.
        """
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Install the project requirements to enable reply generation."
            ) from exc
        return OpenAI

    def _extract_text(self, response: Any) -> str | None:
        """Extract text.
        
        Args:
            response: Model response object to inspect.
        
        Returns:
            The str | None value produced by this callable.
        """
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
        """Truncate text.
        
        Args:
            text: Input text handled by the current operation.
            limit: Parameter `limit` used by this callable.
        
        Returns:
            The str value produced by this callable.
        """
        cleaned = text.strip()
        if limit <= 0 or len(cleaned) <= limit:
            return cleaned
        return cleaned[:limit].rstrip()

    def _extract_usage(self, response: Any) -> dict[str, Any] | None:
        """Extract usage.
        
        Args:
            response: Model response object to inspect.
        
        Returns:
            The dict[str, Any] | None value produced by this callable.
        """
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
        """Error.
        
        Args:
            message: Human-readable message text.
        
        Returns:
            The dict[str, Any] value produced by this callable.
        """
        return {
            "status": "error",
            "message": message,
            "timestamp": self._timestamp(),
        }

    @staticmethod
    def _timestamp() -> str:
        """Timestamp.
        
        Args:
            None.
        
        Returns:
            The str value produced by this callable.
        """
        return datetime.now().isoformat(timespec="seconds")
