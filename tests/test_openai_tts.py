from pathlib import Path
import sys
import wave

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_tel.openai_tts import OpenAITTS


class _FakeSpeechResponse:
    """Test double for SpeechResponse.
    """
    def __enter__(self):
        """Enter .
        
        Args:
            None.
        
        Returns:
            The result produced by this callable.
        """
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit .
        
        Args:
            exc_type: Exception type used by the context manager.
            exc: Exception instance used by the context manager.
            tb: Traceback used by the context manager.
        
        Returns:
            The result produced by this callable.
        """
        return False

    def stream_to_file(self, path):
        """Stream to file.
        
        Args:
            path: Parameter `path` used by this callable.
        
        Returns:
            The result produced by this callable.
        """
        Path(path).write_bytes(b"RIFF" + b"0" * 64)


class _FakeSpeechApi:
    """Test double for SpeechApi.
    """
    def __init__(self) -> None:
        """Initialize the _FakeSpeechApi instance.
        
        Args:
            None.
        
        Returns:
            None.
        """
        self.last_kwargs = None
        self.with_streaming_response = self

    def create(self, **kwargs):
        """Create.
        
        Args:
            kwargs: Additional keyword arguments passed through the helper.
        
        Returns:
            The result produced by this callable.
        """
        self.last_kwargs = kwargs
        return _FakeSpeechResponse()


class _FakeAudioApi:
    """Test double for AudioApi.
    """
    def __init__(self) -> None:
        """Initialize the _FakeAudioApi instance.
        
        Args:
            None.
        
        Returns:
            None.
        """
        self.speech = _FakeSpeechApi()


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
        self.audio = _FakeAudioApi()


class _FakeSoundDevice:
    """Test double for SoundDevice.
    """
    def __init__(self) -> None:
        """Initialize the _FakeSoundDevice instance.
        
        Args:
            None.
        
        Returns:
            None.
        """
        self.play_called = False
        self.wait_called = False
        self.last_audio = None
        self.last_sample_rate = None

    def play(self, audio, sample_rate) -> None:
        """Play.
        
        Args:
            audio: Audio array or payload used for playback.
            sample_rate: Requested or detected audio sample rate.
        
        Returns:
            None.
        """
        self.play_called = True
        self.last_audio = audio
        self.last_sample_rate = sample_rate

    def wait(self) -> None:
        """Wait.
        
        Args:
            None.
        
        Returns:
            None.
        """
        self.wait_called = True


def test_resolve_voice_profile_returns_requested_profile() -> None:
    """Test that resolve voice profile returns requested profile.
    
    Args:
        None.
    
    Returns:
        None.
    """
    tts = OpenAITTS()

    profile = tts.resolve_voice_profile(gender="female", age_group="senior")

    assert profile.voice == "sage"
    assert profile.gender == "female"
    assert profile.age_group == "senior"


def test_resolve_voice_profile_falls_back_to_neutral_adult() -> None:
    """Test that resolve voice profile falls back to neutral adult.
    
    Args:
        None.
    
    Returns:
        None.
    """
    tts = OpenAITTS()

    profile = tts.resolve_voice_profile(gender="unknown", age_group="unknown")

    assert profile.voice == "alloy"
    assert profile.gender == "neutral"
    assert profile.age_group == "adult"


def test_validate_wav_file_rejects_incomplete_output() -> None:
    """Test that validate wav file rejects incomplete output.
    
    Args:
        None.
    
    Returns:
        None.
    """
    tts = OpenAITTS()
    test_dir = Path("tests_tmp/tts_validation")
    test_dir.mkdir(parents=True, exist_ok=True)
    bad_file = test_dir / "broken.wav"
    bad_file.write_bytes(b"RIFF")

    result = tts._validate_wav_file(bad_file)

    assert result is not None
    assert "incomplete" in result.lower()


def test_synthesize_to_wav_uses_gpt_4o_mini_tts(monkeypatch) -> None:
    """Test that synthesize to wav uses gpt 4o mini tts.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    
    Returns:
        None.
    """
    tts = OpenAITTS()
    fake_client = _FakeOpenAIClient(api_key="test-key")

    monkeypatch.setattr(tts, "_get_api_key", lambda: "test-key")
    monkeypatch.setattr(tts, "_load_openai_client_class", lambda: lambda api_key: fake_client)
    monkeypatch.setattr(tts, "_validate_wav_file", lambda path: None)

    result = tts.synthesize_to_wav("Hello world", gender="male", age_group="adult")

    assert result["status"] == "success"
    assert result["voice"] == "onyx"
    assert "generated_wav" in result["file_path"]
    assert Path(result["file_path"]).exists()
    assert fake_client.audio.speech.last_kwargs["model"] == "gpt-4o-mini-tts"
    assert fake_client.audio.speech.last_kwargs["voice"] == "onyx"
    assert fake_client.audio.speech.last_kwargs["response_format"] == "wav"


def test_synthesize_to_wav_rejects_empty_text() -> None:
    """Test that synthesize to wav rejects empty text.
    
    Args:
        None.
    
    Returns:
        None.
    """
    tts = OpenAITTS()

    result = tts.synthesize_to_wav("   ")

    assert result["status"] == "error"
    assert "empty" in result["message"].lower()


def test_play_wav_file_uses_sounddevice(monkeypatch) -> None:
    """Test that play wav file uses sounddevice.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    
    Returns:
        None.
    """
    tts = OpenAITTS()
    fake_sounddevice = _FakeSoundDevice()
    test_dir = Path("tests_tmp/tts_playback")
    test_dir.mkdir(parents=True, exist_ok=True)
    wav_path = test_dir / "playable.wav"

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)

    monkeypatch.setattr(tts, "_load_sounddevice_module", lambda: fake_sounddevice)

    backend = tts.play_wav_file(wav_path)

    assert backend == "sounddevice"
    assert fake_sounddevice.play_called is True
    assert fake_sounddevice.wait_called is True
    assert fake_sounddevice.last_sample_rate == 16000


def test_play_wav_file_falls_back_to_winsound_on_windows(monkeypatch) -> None:
    """Test that play wav file falls back to winsound on windows.
    
    Args:
        monkeypatch: Pytest monkeypatch fixture.
    
    Returns:
        None.
    """
    tts = OpenAITTS()
    test_dir = Path("tests_tmp/tts_playback")
    test_dir.mkdir(parents=True, exist_ok=True)
    wav_path = test_dir / "winsound_fallback.wav"

    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)

    played_with = {"backend": None}

    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(tts, "_load_sounddevice_module", lambda: (_ for _ in ()).throw(RuntimeError("sounddevice unavailable")))
    monkeypatch.setattr(tts, "_play_with_winsound", lambda path: played_with.__setitem__("backend", "winsound"))

    backend = tts.play_wav_file(wav_path)

    assert backend == "winsound"
    assert played_with["backend"] == "winsound"
