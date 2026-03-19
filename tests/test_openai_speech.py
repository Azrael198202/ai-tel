from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_tel.openai_speech import MicrophoneRecorder, OpenAISpeechRecognizer


class _FakeTranscription:
    def __init__(self, text: str) -> None:
        self.text = text
        self.usage = {"seconds": 4}


class _FakeTranscriptionsApi:
    def __init__(self) -> None:
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeTranscription("hello world")


class _FakeAudioApi:
    def __init__(self) -> None:
        self.transcriptions = _FakeTranscriptionsApi()


class _FakeOpenAIClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.audio = _FakeAudioApi()


def test_microphone_recorder_rejects_invalid_duration() -> None:
    recorder = MicrophoneRecorder()

    result = recorder.record_to_wav(duration=0)

    assert result["status"] == "error"
    assert "duration" in result["message"].lower()


def test_microphone_recorder_rejects_stop_before_start() -> None:
    recorder = MicrophoneRecorder()

    result = recorder.stop_recording_to_wav()

    assert result["status"] == "error"
    assert "started" in result["message"].lower()


def test_openai_transcriber_requires_api_key(monkeypatch) -> None:
    recognizer = OpenAISpeechRecognizer()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    temp_root = Path("tests_tmp/no_key")
    temp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(temp_root)

    audio_path = Path("sample.wav")
    audio_path.write_bytes(b"RIFF")

    result = recognizer.transcribe_audio_file(audio_path)

    assert result["status"] == "error"
    assert "openai_api_key" in result["message"].lower()


def test_openai_transcriber_reads_api_key_from_dotenv(monkeypatch) -> None:
    recognizer = OpenAISpeechRecognizer()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    temp_root = Path("tests_tmp/dotenv")
    temp_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(temp_root)
    Path(".env").write_text("OPENAI_API_KEY=test-dotenv-key\n", encoding="utf-8")

    assert recognizer._get_api_key() == "test-dotenv-key"


def test_openai_transcriber_normalizes_culture_to_language_hint() -> None:
    recognizer = OpenAISpeechRecognizer()

    assert recognizer._normalize_language_hint("ja-JP") == "ja"
    assert recognizer._normalize_language_hint("en") == "en"
    assert recognizer._normalize_language_hint(None) is None


def test_openai_transcriber_builds_default_language_prompt() -> None:
    recognizer = OpenAISpeechRecognizer()

    prompt = recognizer._build_prompt("ja", None)

    assert prompt is not None
    assert "Japanese" in prompt


def test_openai_transcriber_can_preserve_audio_file() -> None:
    recognizer = OpenAISpeechRecognizer()

    temp_root = Path("tests_tmp/preserve_audio")
    temp_root.mkdir(parents=True, exist_ok=True)
    source = temp_root / "input.wav"
    source.write_bytes(b"RIFFDATA")

    result = recognizer.preserve_audio_file(source, directory=temp_root / "saved")

    assert result["status"] == "success"
    saved = Path(result["file_path"])
    assert saved.exists()
    assert saved.read_bytes() == b"RIFFDATA"


def test_openai_transcriber_detects_quiet_audio() -> None:
    recognizer = OpenAISpeechRecognizer()

    assert recognizer.has_usable_audio({"peak_level": 0.0002, "rms_level": 0.0001}) is False
    assert recognizer.has_usable_audio({"peak_level": 0.02, "rms_level": 0.005}) is True


def test_openai_transcriber_uses_gpt_4o_transcribe(monkeypatch) -> None:
    recognizer = OpenAISpeechRecognizer()
    fake_client = _FakeOpenAIClient(api_key="test-key")

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(recognizer, "_load_openai_client_class", lambda: lambda api_key: fake_client)

    audio_path = Path("tests_tmp/fake.wav")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"RIFF")

    result = recognizer.transcribe_audio_file(audio_path, language="ja", prompt="The expected sentence is about daily garbage.")

    assert result["status"] == "success"
    assert result["text"] == "hello world"
    assert result["model"] == "gpt-4o-transcribe"
    assert result["language_hint"] == "ja"
    assert result["usage"] == {"seconds": 4}
    assert fake_client.audio.transcriptions.last_kwargs["model"] == "gpt-4o-transcribe"
    assert fake_client.audio.transcriptions.last_kwargs["language"] == "ja"
    prompt = fake_client.audio.transcriptions.last_kwargs["prompt"]
    assert "Japanese" in prompt
    assert "daily garbage" in prompt
