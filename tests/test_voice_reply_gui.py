from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_tel.voice_reply_gui import VoiceAssistantApp


class _FakeOutput:
    """Test double for Output.
    """
    def __init__(self) -> None:
        """Initialize the _FakeOutput instance.
        
        Args:
            None.
        
        Returns:
            None.
        """
        self.text = ""

    def configure(self, **kwargs) -> None:  # noqa: ARG002
        """Configure.
        
        Args:
            kwargs: Additional keyword arguments passed through the helper.
        
        Returns:
            None.
        """
        return None

    def delete(self, start: str, end: str) -> None:  # noqa: ARG002
        """Delete.
        
        Args:
            start: Start index or position.
            end: End index or position.
        
        Returns:
            None.
        """
        self.text = ""

    def insert(self, start: str, text: str) -> None:  # noqa: ARG002
        """Insert.
        
        Args:
            start: Start index or position.
            text: Input text handled by the current operation.
        
        Returns:
            None.
        """
        self.text += text

    def see(self, index: str) -> None:  # noqa: ARG002
        """See.
        
        Args:
            index: Index value used by the helper.
        
        Returns:
            None.
        """
        return None


def _make_app(tmp_path: Path) -> VoiceAssistantApp:
    """Make app.
    
    Args:
        tmp_path: Temporary directory used by the test.
    
    Returns:
        The VoiceAssistantApp value produced by this callable.
    """
    app = VoiceAssistantApp.__new__(VoiceAssistantApp)
    app.output = _FakeOutput()
    app.session_output_dir_name = "conversation_logs"
    app.session_started_at = datetime(2026, 3, 19, 14, 0, 0)
    app.session_stopped_at = None
    app.session_turns = []
    app.session_errors = []
    app.session_log_file_path = str(tmp_path / "conversation_logs" / "conversation_20260319_140000.txt")
    return app


def test_render_session_text_groups_turns_and_errors(tmp_path: Path) -> None:
    """Test that render session text groups turns and errors.
    
    Args:
        tmp_path: Temporary directory used by the test.
    
    Returns:
        None.
    """
    app = _make_app(tmp_path)
    app.session_turns = [
        {"user": "hello", "assistant": "hi there"},
        {"user": "second line", "assistant": "second answer"},
    ]
    app.session_errors = ["TTS failed"]
    app.session_stopped_at = datetime(2026, 3, 19, 14, 5, 0)

    rendered = app._render_session_text()

    assert "Conversation transcript" in rendered
    assert "Started: 2026-03-19 14:00:00" in rendered
    assert "Stopped: 2026-03-19 14:05:00" in rendered
    assert "Turn 1" in rendered
    assert "User: hello" in rendered
    assert "Assistant: hi there" in rendered
    assert "Turn 2" in rendered
    assert "Errors" in rendered
    assert "1. TTS failed" in rendered


def test_persist_session_log_updates_output_and_writes_file(tmp_path: Path) -> None:
    """Test that persist session log updates output and writes file.
    
    Args:
        tmp_path: Temporary directory used by the test.
    
    Returns:
        None.
    """
    app = _make_app(tmp_path)
    Path(app.session_log_file_path).parent.mkdir(parents=True, exist_ok=True)
    app.session_turns = [{"user": "hello", "assistant": "hi"}]

    app._persist_session_log()

    saved_text = Path(app.session_log_file_path).read_text(encoding="utf-8")
    assert "Turn 1" in saved_text
    assert "User: hello" in saved_text
    assert "Assistant: hi" in saved_text
    assert app.output.text == saved_text
