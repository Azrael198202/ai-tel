from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_tel.gui import SpeechTestApp


class _FakeOutput:
    def __init__(self) -> None:
        self.text = ""

    def configure(self, **kwargs) -> None:  # noqa: ARG002
        return None

    def delete(self, start: str, end: str) -> None:  # noqa: ARG002
        self.text = ""

    def insert(self, start: str, text: str) -> None:  # noqa: ARG002
        self.text += text

    def see(self, index: str) -> None:  # noqa: ARG002
        return None


def _make_app(tmp_path: Path) -> SpeechTestApp:
    app = SpeechTestApp.__new__(SpeechTestApp)
    app.output = _FakeOutput()
    app.transcript_output_dir_name = "transcription_logs"
    app.session_started_at = datetime(2026, 3, 19, 15, 0, 0)
    app.session_stopped_at = None
    app.session_segments = []
    app.session_errors = []
    app.session_log_file_path = str(tmp_path / "transcription_logs" / "transcript_20260319_150000.txt")
    return app


def test_render_session_text_groups_transcript_segments_and_errors(tmp_path: Path) -> None:
    """Test that rendered transcript text groups segments and errors clearly.

    Args:
        tmp_path: Temporary directory used by the test.

    Returns:
        None.
    """
    app = _make_app(tmp_path)
    app.session_segments = ["first sentence", "second sentence"]
    app.session_errors = ["microphone too quiet"]
    app.session_stopped_at = datetime(2026, 3, 19, 15, 5, 0)

    rendered = app._render_session_text()

    assert "Speech transcript" in rendered
    assert "Started: 2026-03-19 15:00:00" in rendered
    assert "Stopped: 2026-03-19 15:05:00" in rendered
    assert "Transcript" in rendered
    assert "1. first sentence" in rendered
    assert "2. second sentence" in rendered
    assert "Errors" in rendered
    assert "1. microphone too quiet" in rendered


def test_persist_session_log_updates_output_and_writes_file(tmp_path: Path) -> None:
    """Test that persisting the transcript refreshes the UI and saves the file.

    Args:
        tmp_path: Temporary directory used by the test.

    Returns:
        None.
    """
    app = _make_app(tmp_path)
    Path(app.session_log_file_path).parent.mkdir(parents=True, exist_ok=True)
    app.session_segments = ["hello world"]

    app._persist_session_log()

    saved_text = Path(app.session_log_file_path).read_text(encoding="utf-8")
    assert "1. hello world" in saved_text
    assert app.output.text == saved_text
