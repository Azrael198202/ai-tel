from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_tel.openai_reply import OpenAITextResponder


class _FakeMessage:
    """Test double for Message.
    """
    def __init__(self, content: str) -> None:
        """Initialize the _FakeMessage instance.
        
        Args:
            content: Parameter `content` used by this callable.
        
        Returns:
            None.
        """
        self.content = content


class _FakeChoice:
    """Test double for Choice.
    """
    def __init__(self, content: str) -> None:
        """Initialize the _FakeChoice instance.
        
        Args:
            content: Parameter `content` used by this callable.
        
        Returns:
            None.
        """
        self.message = _FakeMessage(content)


class _FakeCompletionResponse:
    """Test double for CompletionResponse.
    """
    def __init__(self, content: str) -> None:
        """Initialize the _FakeCompletionResponse instance.
        
        Args:
            content: Parameter `content` used by this callable.
        
        Returns:
            None.
        """
        self.choices = [_FakeChoice(content)]
        self.usage = {"total_tokens": 12}


class _FakeChatCompletionsApi:
    """Test double for ChatCompletionsApi.
    """
    def __init__(self) -> None:
        """Initialize the _FakeChatCompletionsApi instance.
        
        Args:
            None.
        
        Returns:
            None.
        """
        self.last_kwargs = None

    def create(self, **kwargs):
        """Create.
        
        Args:
            kwargs: Additional keyword arguments passed through the helper.
        
        Returns:
            The result produced by this callable.
        """
        self.last_kwargs = kwargs
        return _FakeCompletionResponse("Hello, I can help with that.")


class _FakeChatApi:
    """Test double for ChatApi.
    """
    def __init__(self) -> None:
        """Initialize the _FakeChatApi instance.
        
        Args:
            None.
        
        Returns:
            None.
        """
        self.completions = _FakeChatCompletionsApi()


class _FakeOpenAIClient:
    """Test double for OpenAIClient.
    """
    def __init__(self, api_key: str) -> None:
        """Initialize the _FakeOpenAIClient instance.
        
        Args:
            api_key: OpenAI API key value.
        
        Returns:
            None.
        """
        self.api_key = api_key
        self.chat = _FakeChatApi()


def test_openai_text_responder_uses_knowledge_base(monkeypatch) -> None:
    """Test that openai text responder uses knowledge base.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    
    Returns:
        None.
    """
    responder = OpenAITextResponder()

    temp_root = Path("tests_tmp/reply_kb")
    kb_dir = temp_root / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "faq.txt").write_text(
        "Garbage sorting rules:\nBurnable trash is collected on Tuesday and Friday.",
        encoding="utf-8",
    )
    monkeypatch.chdir(temp_root)

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_client = _FakeOpenAIClient(api_key="test-key")
    monkeypatch.setattr(responder, "_load_openai_client_class", lambda: lambda api_key: fake_client)

    result = responder.generate_reply("When is burnable trash collected?", language_hint="en")

    assert result["status"] == "success"
    assert result["source"] == "knowledge_base"
    assert result["model"] == "gpt-4o-mini"
    knowledge_prompt = fake_client.chat.completions.last_kwargs["messages"][-1]["content"]
    assert "Reference information:" in knowledge_prompt
    assert "Tuesday and Friday" in knowledge_prompt


def test_openai_text_responder_falls_back_to_local_knowledge_without_api_key(monkeypatch) -> None:
    """Test that openai text responder falls back to local knowledge without api key.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    
    Returns:
        None.
    """
    responder = OpenAITextResponder()

    temp_root = Path("tests_tmp/reply_kb_local")
    kb_dir = temp_root / "knowledge_base"
    kb_dir.mkdir(parents=True, exist_ok=True)
    (kb_dir / "faq.txt").write_text(
        "Garbage sorting rules:\nBurnable trash is collected on Tuesday and Friday.",
        encoding="utf-8",
    )
    monkeypatch.chdir(temp_root)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    result = responder.generate_reply("When is burnable trash collected?", language_hint="en")

    assert result["status"] == "success"
    assert result["source"] == "knowledge_base"
    assert result["model"] == "local-knowledge-base"
    assert "Tuesday and Friday" in result["text"]


def test_openai_text_responder_requires_api_key(monkeypatch) -> None:
    """Test that openai text responder requires api key.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    
    Returns:
        None.
    """
    responder = OpenAITextResponder()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    temp_root = Path("tests_tmp/reply_no_key")
    temp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(temp_root)

    result = responder.generate_reply("hello")

    assert result["status"] == "error"
    assert "openai_api_key" in result["message"].lower()


def test_openai_text_responder_generates_reply(monkeypatch) -> None:
    """Test that openai text responder generates reply.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    
    Returns:
        None.
    """
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


def test_openai_text_responder_includes_conversation_history(monkeypatch) -> None:
    """Test that openai text responder includes conversation history.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    
    Returns:
        None.
    """
    responder = OpenAITextResponder()
    fake_client = _FakeOpenAIClient(api_key="test-key")

    temp_root = Path("tests_tmp/reply_history")
    temp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(temp_root)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(responder, "_load_openai_client_class", lambda: lambda api_key: fake_client)

    responder.generate_reply(
        user_text="What about tomorrow?",
        conversation_history=[
            {"role": "user", "content": "When is burnable trash collected?"},
            {"role": "assistant", "content": "Tuesday and Friday."},
            {"role": "system", "content": "ignore me"},
            {"role": "assistant", "content": "   "},
        ],
    )

    messages = fake_client.chat.completions.last_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1] == {"role": "user", "content": "When is burnable trash collected?"}
    assert messages[2] == {"role": "assistant", "content": "Tuesday and Friday."}
    assert messages[3] == {"role": "user", "content": "What about tomorrow?"}
