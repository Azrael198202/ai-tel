"""Speech capture and OpenAI transcription helpers."""

from __future__ import annotations

import os
import shutil
import tempfile
import wave
from datetime import datetime
from pathlib import Path
from typing import Any


class MicrophoneRecorder:
    """Record audio from the default microphone into a WAV file."""

    def __init__(self) -> None:
        self._stream = None
        self._frames = []
        self._sample_rate = 48000
        self._channels = 1
        self._dtype = "int16"

    def record_to_wav(
        self,
        duration: int = 8,
        sample_rate: int | None = None,
        channels: int = 1,
    ) -> dict[str, Any]:
        if duration <= 0:
            return self._error("Duration must be greater than zero.")
        if channels <= 0:
            return self._error("Channels must be greater than zero.")

        try:
            sounddevice, _ = self._load_audio_dependencies()
        except Exception as exc:
            return self._error(str(exc))

        resolved_sample_rate = self._resolve_sample_rate(sounddevice, sample_rate)
        if resolved_sample_rate <= 0:
            return self._error("Sample rate must be greater than zero.")

        frames = int(duration * resolved_sample_rate)

        try:
            recording = sounddevice.rec(
                frames,
                samplerate=resolved_sample_rate,
                channels=channels,
                dtype=self._dtype,
            )
            sounddevice.wait()
        except Exception as exc:
            return self._error(f"Microphone recording failed: {exc}")

        self._sample_rate = resolved_sample_rate
        return self._write_wav_file(recording, sample_rate=resolved_sample_rate, channels=channels)

    def start_recording(self, sample_rate: int | None = None, channels: int = 1) -> dict[str, Any]:
        if self._stream is not None:
            return self._error("Recording is already in progress.")
        if channels <= 0:
            return self._error("Channels must be greater than zero.")

        try:
            sounddevice, _ = self._load_audio_dependencies()
        except Exception as exc:
            return self._error(str(exc))

        self._frames = []
        self._sample_rate = self._resolve_sample_rate(sounddevice, sample_rate)
        if self._sample_rate <= 0:
            return self._error("Sample rate must be greater than zero.")
        self._channels = channels

        def callback(indata, frames, time, status) -> None:  # noqa: ARG001
            if status:
                return
            self._frames.append(indata.copy())

        try:
            self._stream = sounddevice.InputStream(
                samplerate=self._sample_rate,
                channels=channels,
                dtype=self._dtype,
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            self._frames = []
            return self._error(f"Microphone recording failed: {exc}")

        return {
            "status": "success",
            "message": "Recording started.",
            "sample_rate": self._sample_rate,
            "channels": channels,
            "timestamp": self._timestamp(),
        }

    def stop_recording_to_wav(self) -> dict[str, Any]:
        if self._stream is None:
            return self._error("Recording has not been started.")

        try:
            self._stream.stop()
            self._stream.close()
        except Exception as exc:
            self._stream = None
            self._frames = []
            return self._error(f"Failed to stop microphone recording: {exc}")

        self._stream = None

        if not self._frames:
            self._frames = []
            return self._error("No audio frames were captured.")

        try:
            _, numpy = self._load_audio_dependencies()
            recording = numpy.concatenate(self._frames, axis=0)
        except Exception as exc:
            self._frames = []
            return self._error(f"Failed to assemble recorded audio: {exc}")

        self._frames = []
        return self._write_wav_file(recording, sample_rate=self._sample_rate, channels=self._channels)

    def _write_wav_file(self, recording, sample_rate: int, channels: int) -> dict[str, Any]:
        prepared = self._prepare_recording(recording, sample_rate)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = Path(temp_file.name)
        temp_file.close()

        try:
            with wave.open(str(temp_path), "wb") as wav_file:
                wav_file.setnchannels(channels)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(prepared["audio"].tobytes())
        except Exception as exc:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass
            return self._error(f"Failed to save recorded audio: {exc}")

        return {
            "status": "success",
            "file_path": str(temp_path),
            "sample_rate": sample_rate,
            "channels": channels,
            "duration_seconds": prepared["duration_seconds"],
            "trimmed_seconds": prepared["trimmed_seconds"],
            "peak_level": prepared["peak_level"],
            "rms_level": prepared["rms_level"],
            "timestamp": self._timestamp(),
        }

    def _load_audio_dependencies(self):
        try:
            import numpy
            import sounddevice  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Audio recording dependencies are missing. Install the project requirements to enable microphone capture."
            ) from exc
        return sounddevice, numpy

    def _resolve_sample_rate(self, sounddevice, sample_rate: int | None) -> int:
        if sample_rate is not None:
            return int(sample_rate)

        try:
            device_info = sounddevice.query_devices(kind="input")
            default_rate = device_info.get("default_samplerate")
            if default_rate:
                return int(default_rate)
        except Exception:
            pass

        return 48000

    def _prepare_recording(self, recording, sample_rate: int) -> dict[str, Any]:
        _, numpy = self._load_audio_dependencies()
        audio = recording.astype(numpy.float32)

        if audio.ndim == 1:
            mono = audio
        else:
            mono = audio.mean(axis=1)

        if mono.size == 0:
            return {
                "audio": recording,
                "duration_seconds": 0.0,
                "trimmed_seconds": 0.0,
                "peak_level": 0.0,
                "rms_level": 0.0,
            }

        peak = float(numpy.max(numpy.abs(mono)))
        silence_threshold = max(peak * 0.02, 500.0)
        active = numpy.where(numpy.abs(mono) >= silence_threshold)[0]

        trimmed = audio
        trimmed_seconds = 0.0
        if active.size > 0:
            lead_padding = int(sample_rate * 0.1)
            tail_padding = int(sample_rate * 0.2)
            start = max(int(active[0]) - lead_padding, 0)
            end = min(int(active[-1]) + tail_padding + 1, audio.shape[0])
            trimmed = audio[start:end]
            trimmed_seconds = max((audio.shape[0] - trimmed.shape[0]) / float(sample_rate), 0.0)

        trimmed_peak = float(numpy.max(numpy.abs(trimmed))) if trimmed.size else 0.0
        target_peak = 26000.0
        if 0 < trimmed_peak < target_peak:
            gain = min(target_peak / trimmed_peak, 8.0)
            trimmed = trimmed * gain

        clipped = numpy.clip(trimmed, -32768, 32767).astype(numpy.int16)
        clipped_mono = clipped.astype(numpy.float32)
        if clipped.ndim > 1:
            clipped_mono = clipped_mono.mean(axis=1)

        return {
            "audio": clipped,
            "duration_seconds": round(clipped.shape[0] / float(sample_rate), 2),
            "trimmed_seconds": round(trimmed_seconds, 2),
            "peak_level": round(float(numpy.max(numpy.abs(clipped_mono))) / 32768.0, 4) if clipped.size else 0.0,
            "rms_level": round(float(numpy.sqrt(numpy.mean(numpy.square(clipped_mono)))) / 32768.0, 4) if clipped.size else 0.0,
        }

    def _error(self, message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "timestamp": self._timestamp(),
        }

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")


class OpenAISpeechRecognizer:
    """Transcribe recorded audio using OpenAI gpt-4o-transcribe."""

    model = "gpt-4o-transcribe"
    min_detectable_peak_level = 0.003
    min_detectable_rms_level = 0.001

    def __init__(self) -> None:
        self.recorder = MicrophoneRecorder()

    def listen_once(self, timeout: int = 8, culture: str | None = None, prompt: str | None = None) -> dict[str, Any]:
        recording = self.recorder.record_to_wav(duration=timeout)
        if recording.get("status") != "success":
            return recording

        audio_path = Path(str(recording["file_path"]))
        try:
            transcription = self.transcribe_audio_file(
                audio_path,
                language=self._normalize_language_hint(culture),
                prompt=prompt,
            )
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

        return transcription

    def transcribe_audio_file(
        self,
        file_path: str | Path,
        language: str | None = None,
        prompt: str | None = None,
    ) -> dict[str, Any]:
        api_key = self._get_api_key()
        if not api_key:
            return self._error("OPENAI_API_KEY is not set.")

        try:
            OpenAI = self._load_openai_client_class()
        except Exception as exc:
            return self._error(str(exc))

        client = OpenAI(api_key=api_key)
        path = Path(file_path)
        if not path.exists():
            return self._error(f"Audio file not found: {path}")

        kwargs: dict[str, Any] = {"model": self.model}
        if language:
            kwargs["language"] = language
        combined_prompt = self._build_prompt(language=language, prompt=prompt)
        if combined_prompt:
            kwargs["prompt"] = combined_prompt

        try:
            with path.open("rb") as audio_file:
                transcription = client.audio.transcriptions.create(file=audio_file, **kwargs)
        except Exception as exc:
            return self._error(f"OpenAI transcription failed: {exc}")

        text = self._extract_text(transcription)
        if not text:
            return self._error("OpenAI transcription returned no text.")

        result = {
            "status": "success",
            "text": text,
            "model": self.model,
            "language_hint": language,
            "timestamp": self._timestamp(),
        }

        usage = self._extract_usage(transcription)
        if usage is not None:
            result["usage"] = usage

        return result

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
                    value = value.strip().strip('"').strip("'")
                    if value:
                        os.environ["OPENAI_API_KEY"] = value
                        return value
        except Exception:
            return None

        return None

    def _load_openai_client_class(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The openai package is not installed. Install the project requirements to enable gpt-4o-transcribe."
            ) from exc
        return OpenAI

    def preserve_audio_file(
        self,
        file_path: str | Path,
        directory: str | Path | None = None,
        prefix: str = "stt_recording",
    ) -> dict[str, Any]:
        source = Path(file_path)
        if not source.exists():
            return self._error(f"Audio file not found: {source}")

        target_dir = Path(directory) if directory else Path.cwd() / "recorded_wav"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        target_path = target_dir / target_name

        try:
            shutil.copy2(source, target_path)
        except Exception as exc:
            return self._error(f"Failed to preserve audio file: {exc}")

        return {
            "status": "success",
            "file_path": str(target_path),
            "timestamp": self._timestamp(),
        }

    def has_usable_audio(self, recording: dict[str, Any]) -> bool:
        peak_level = recording.get("peak_level")
        rms_level = recording.get("rms_level")

        if not isinstance(peak_level, (int, float)) or not isinstance(rms_level, (int, float)):
            return True

        return not (
            peak_level < self.min_detectable_peak_level
            and rms_level < self.min_detectable_rms_level
        )

    def _extract_text(self, transcription: Any) -> str | None:
        if hasattr(transcription, "text"):
            return transcription.text
        if isinstance(transcription, dict):
            return transcription.get("text")
        return None

    def _extract_usage(self, transcription: Any) -> dict[str, Any] | None:
        usage = getattr(transcription, "usage", None)
        if usage is not None:
            if hasattr(usage, "model_dump"):
                return usage.model_dump()
            if isinstance(usage, dict):
                return usage
        if isinstance(transcription, dict):
            maybe_usage = transcription.get("usage")
            if isinstance(maybe_usage, dict):
                return maybe_usage
        return None

    def _normalize_language_hint(self, culture: str | None) -> str | None:
        if not culture:
            return None
        value = culture.strip()
        if not value:
            return None
        if "-" in value:
            return value.split("-", 1)[0].lower()
        return value.lower()

    def _build_prompt(self, language: str | None, prompt: str | None) -> str | None:
        base_prompt = self._default_prompt_for_language(language)
        custom_prompt = (prompt or "").strip()

        if base_prompt and custom_prompt:
            return f"{base_prompt}\nAdditional context: {custom_prompt}"
        if base_prompt:
            return base_prompt
        if custom_prompt:
            return custom_prompt
        return None

    def _default_prompt_for_language(self, language: str | None) -> str | None:
        prompts = {
            "ja": (
                "This audio is in Japanese. Transcribe it faithfully in natural Japanese script, "
                "keeping Japanese words intact and using punctuation when it is clear from the speech."
            ),
            "zh": (
                "This audio is in Chinese. Transcribe it faithfully in natural Chinese characters and "
                "preserve the original wording as spoken."
            ),
            "ko": (
                "This audio is in Korean. Transcribe it faithfully in natural Korean Hangul and "
                "preserve the original wording as spoken."
            ),
        }
        return prompts.get(language)

    def _error(self, message: str) -> dict[str, Any]:
        return {
            "status": "error",
            "message": message,
            "timestamp": self._timestamp(),
        }

    @staticmethod
    def _timestamp() -> str:
        return datetime.now().isoformat(timespec="seconds")
