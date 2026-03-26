"""OpenAI text-to-speech helpers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
import wave

import numpy as np


@dataclass(frozen=True)
class VoiceProfile:
    """Provide the voice profile component.
    """
    gender: str
    age_group: str
    voice: str
    instructions: str


VOICE_PROFILE_MAP: dict[tuple[str, str], VoiceProfile] = {
    ("female", "child"): VoiceProfile("female", "child", "shimmer", "Speak with a bright, youthful, playful tone."),
    ("female", "young_adult"): VoiceProfile("female", "young_adult", "coral", "Speak with a clear, lively, youthful adult tone."),
    ("female", "adult"): VoiceProfile("female", "adult", "nova", "Speak with a calm, confident adult tone."),
    ("female", "senior"): VoiceProfile("female", "senior", "sage", "Speak with a warm, mature, senior tone."),
    ("male", "child"): VoiceProfile("male", "child", "ash", "Speak with a youthful, energetic tone."),
    ("male", "young_adult"): VoiceProfile("male", "young_adult", "echo", "Speak with a natural, modern young adult tone."),
    ("male", "adult"): VoiceProfile("male", "adult", "onyx", "Speak with a steady, confident adult tone."),
    ("male", "senior"): VoiceProfile("male", "senior", "fable", "Speak with a mature, gentle senior tone."),
    ("neutral", "child"): VoiceProfile("neutral", "child", "alloy", "Speak with a light, youthful, childlike tone."),
    ("neutral", "young_adult"): VoiceProfile("neutral", "young_adult", "ash", "Speak with a balanced, youthful tone."),
    ("neutral", "adult"): VoiceProfile("neutral", "adult", "alloy", "Speak with a balanced, neutral adult tone."),
    ("neutral", "senior"): VoiceProfile("neutral", "senior", "ballad", "Speak with a composed, mature, reflective tone."),
}


class OpenAITTS:
    """Generate spoken audio with OpenAI gpt-4o-mini-tts."""

    model = "gpt-4o-mini-tts"
    output_dir_name = "generated_wav"

    def synthesize_to_wav(self, text: str, gender: str = "neutral", age_group: str = "adult") -> dict[str, Any]:
        """Synthesize to wav.
        
        Args:
            text: Input text handled by the current operation.
            gender: Requested voice gender setting.
            age_group: Requested voice age group setting.
        
        Returns:
            The dict[str, Any] value produced by this callable.
        """
        cleaned_text = text.strip()
        if not cleaned_text:
            return self._error("Text must not be empty.")

        api_key = self._get_api_key()
        if not api_key:
            return self._error("OPENAI_API_KEY is not set.")

        try:
            OpenAI = self._load_openai_client_class()
        except Exception as exc:
            return self._error(str(exc))

        profile = self.resolve_voice_profile(gender=gender, age_group=age_group)
        client = OpenAI(api_key=api_key)
        output_dir = self._ensure_output_dir()
        final_path = self._build_output_path(output_dir=output_dir, profile=profile)
        temp_path = final_path.with_suffix(".part")

        try:
            with client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=profile.voice,
                input=cleaned_text,
                instructions=profile.instructions,
                response_format="wav",
            ) as response:
                response.stream_to_file(temp_path)

            validation_error = self._validate_wav_file(temp_path)
            if validation_error is not None:
                raise RuntimeError(validation_error)

            temp_path.replace(final_path)
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            try:
                final_path.unlink(missing_ok=True)
            except Exception:
                pass
            return self._error(f"Speech generation failed: {exc}")

        return {
            "status": "success",
            "file_path": str(final_path),
            "voice": profile.voice,
            "gender": profile.gender,
            "age_group": profile.age_group,
            "instructions": profile.instructions,
            "timestamp": self._timestamp(),
        }

    def speak_text(self, text: str, gender: str = "neutral", age_group: str = "adult") -> dict[str, Any]:
        """Generate and play text.
        
        Args:
            text: Input text handled by the current operation.
            gender: Requested voice gender setting.
            age_group: Requested voice age group setting.
        
        Returns:
            The dict[str, Any] value produced by this callable.
        """
        result = self.synthesize_to_wav(text=text, gender=gender, age_group=age_group)
        if result.get("status") != "success":
            return result

        file_path = Path(str(result["file_path"]))
        try:
            backend = self.play_wav_file(file_path)
        except Exception as exc:
            return self._error(f"Audio playback failed: {exc}")

        result["playback_backend"] = backend
        return result

    def resolve_voice_profile(self, gender: str, age_group: str) -> VoiceProfile:
        """Resolve voice profile.
        
        Args:
            gender: Requested voice gender setting.
            age_group: Requested voice age group setting.
        
        Returns:
            The VoiceProfile value produced by this callable.
        """
        key = (gender.strip().lower(), age_group.strip().lower())
        return VOICE_PROFILE_MAP.get(key, VOICE_PROFILE_MAP[("neutral", "adult")])

    def _ensure_output_dir(self) -> Path:
        """Ensure output dir.
        
        Args:
            None.
        
        Returns:
            The Path value produced by this callable.
        """
        output_dir = Path.cwd() / self.output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _build_output_path(self, output_dir: Path, profile: VoiceProfile) -> Path:
        """Build output path.
        
        Args:
            output_dir: Directory used for output files.
            profile: Voice profile used for speech synthesis.
        
        Returns:
            The Path value produced by this callable.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tts_{timestamp}_{profile.gender}_{profile.age_group}_{profile.voice}.wav"
        return output_dir / filename

    def _validate_wav_file(self, file_path: Path) -> str | None:
        """Validate wav file.
        
        Args:
            file_path: Path to the file being read, written, or played.
        
        Returns:
            The str | None value produced by this callable.
        """
        if not file_path.exists():
            return "The generated audio file was not created."
        if file_path.stat().st_size <= 44:
            return "The generated WAV file is incomplete."
        try:
            with wave.open(str(file_path), "rb") as wav_file:
                wav_file.getnchannels()
                wav_file.getframerate()
                wav_file.getsampwidth()
        except Exception:
            return "The generated WAV file is invalid."
        return None

    def play_wav_file(self, file_path: str | Path) -> str:
        """Play wav file.
        
        Args:
            file_path: Path to the file being read, written, or played.
        
        Returns:
            The str value produced by this callable.
        """
        path = Path(file_path)
        validation_error = self._validate_wav_file(path)
        if validation_error is not None:
            raise RuntimeError(validation_error)

        sounddevice_error = None
        try:
            self._play_with_sounddevice(path)
            return "sounddevice"
        except Exception as exc:
            sounddevice_error = exc

        winsound_error = None
        if sys.platform.startswith("win"):
            try:
                self._play_with_winsound(path)
                return "winsound"
            except Exception as exc:
                winsound_error = exc

        if winsound_error is not None:
            raise RuntimeError(f"sounddevice playback failed: {sounddevice_error}; winsound playback failed: {winsound_error}") from winsound_error
        if sounddevice_error is not None:
            raise sounddevice_error
        raise RuntimeError("No audio playback backend is available.")

    def _play_with_winsound(self, file_path: Path) -> None:
        """Play with winsound.
        
        Args:
            file_path: Path to the file being read, written, or played.
        
        Returns:
            None.
        """
        import winsound

        winsound.PlaySound(str(file_path), winsound.SND_FILENAME)

    def open_wav_with_system_player(self, file_path: str | Path) -> None:
        """Open wav with system player.
        
        Args:
            file_path: Path to the file being read, written, or played.
        
        Returns:
            None.
        """
        path = Path(file_path)
        validation_error = self._validate_wav_file(path)
        if validation_error is not None:
            raise RuntimeError(validation_error)
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return
        raise RuntimeError("System player launch is only implemented for Windows.")

    def _play_with_sounddevice(self, file_path: Path) -> None:
        """Play with sounddevice.
        
        Args:
            file_path: Path to the file being read, written, or played.
        
        Returns:
            None.
        """
        sounddevice = self._load_sounddevice_module()
        with wave.open(str(file_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_rate = wav_file.getframerate()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())

        dtype_map = {
            1: np.int8,
            2: np.int16,
            4: np.int32,
        }
        dtype = dtype_map.get(sample_width)
        if dtype is None:
            raise RuntimeError(f"Unsupported WAV sample width: {sample_width}")

        audio = np.frombuffer(frames, dtype=dtype)
        if channels > 1:
            audio = audio.reshape(-1, channels)

        sounddevice.play(audio, sample_rate)
        sounddevice.wait()

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
                "The openai package is not installed. Install the project requirements to enable text-to-speech."
            ) from exc
        return OpenAI

    def _load_sounddevice_module(self):
        """Load sounddevice module.
        
        Args:
            None.
        
        Returns:
            The result produced by this callable.
        """
        try:
            import sounddevice  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The sounddevice package is not installed. Install the project requirements to enable audio playback."
            ) from exc
        return sounddevice

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
