from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_tel.openai_reply import OpenAITextResponder


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = {"total_tokens": 12}


class _FakeChatCompletionsApi:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeCompletionResponse("Hello, I can help with that.")


class _FakeChatApi:
    def __init__(self) -> None:
        self.completions = _FakeChatCompletionsApi()


class _FakeOpenAIClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.chat = _FakeChatApi()


def test_openai_text_responder_uses_knowledge_base(monkeypatch) -> None:
    responder = OpenAITextResponder()

    temp_root = Path("tests_tmp/reply_kb")
    kb_dir = temp_root / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "faq.txt").write_text(
        "Garbage sorting rules:\nBurnable trash is collected on Tuesday and Friday.",
        encoding="utf-8",
    )
    monkeypatch.chdir(temp_root)

    result = responder.generate_reply("When is burnable trash collected?", language_hint="en")

    assert result["status"] == "success"
    assert result["source"] == "knowledge_base"
    assert "Tuesday and Friday" in result["text"]


def test_openai_text_responder_requires_api_key(monkeypatch) -> None:
    responder = OpenAITextResponder()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    temp_root = Path("tests_tmp/reply_no_key")
    temp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(temp_root)

    result = responder.generate_reply("hello")

    assert result["status"] == "error"
    assert "openai_api_key" in result["message"].lower()


def test_openai_text_responder_generates_reply(monkeypatch) -> None:
    responder = OpenAITextResponder()
    fake_client = _FakeOpenAIClient(api_key="test-key")

    temp_root = Path("tests_tmp/reply_openai")
    temp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(temp_root)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(responder, "_load_openai_client_class", lambda: lambda api_key: fake_client)

    result = responder.generate_reply(
        user_text="Can you help me sort garbage today?",
        system_prompt="Keep answers concise.",
        language_hint="ja",
    )

    assert result["status"] == "success"
    assert result["text"] == "Hello, I can help with that."
    assert result["model"] == "gpt-4o-mini"
    assert result["source"] == "openai"
    assert result["usage"] == {"total_tokens": 12}
    messages = fake_client.chat.completions.last_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "Japanese" in messages[0]["content"]
    assert "Keep answers concise." in messages[0]["content"]
    assert messages[1]["content"] == "Can you help me sort garbage today?"
