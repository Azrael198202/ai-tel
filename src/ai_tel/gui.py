"""Simple desktop GUI for testing microphone transcription."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from .openai_speech import OpenAISpeechRecognizer


LANGUAGE_OPTIONS = [
    ("Japanese (ja-JP)", "ja-JP"),
    ("Chinese Simplified (zh-CN)", "zh-CN"),
    ("English (en-US)", "en-US"),
    ("Korean (ko-KR)", "ko-KR"),
    ("French (fr-FR)", "fr-FR"),
]


class SpeechTestApp:
    """Provide the speech-to-text test window."""

    transcript_output_dir_name = "transcription_logs"

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the SpeechTestApp instance.

        Args:
            root: Tk root window used by the UI.

        Returns:
            None.
        """
        self.root = root
        self.root.title("AI Speech Test")
        self.root.geometry("760x520")

        self.recognizer = OpenAISpeechRecognizer()
        self.recorder = self.recognizer.recorder
        self.is_recording = False
        self.worker_thread: threading.Thread | None = None

        self.language_map = {label: value for label, value in LANGUAGE_OPTIONS}
        self.language_var = tk.StringVar(value=LANGUAGE_OPTIONS[0][0])
        self.prompt_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self.session_started_at: datetime | None = None
        self.session_stopped_at: datetime | None = None
        self.session_segments: list[str] = []
        self.session_errors: list[str] = []
        self.session_log_file_path: str | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Create and layout all widgets for the transcription window.

        Args:
            None.

        Returns:
            None.
        """
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        nav = ttk.Frame(container)
        nav.pack(fill="x")
        ttk.Label(nav, text="Speech To Text", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(nav, text="Go To Text To Speech", command=self._open_tts_window).pack(side="right")

        title = ttk.Label(container, text="Speech Test", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w", pady=(10, 0))

        subtitle = ttk.Label(
            container,
            text="Click once to start listening. Each pause creates a new transcript line. Click again to stop and save the document.",
            wraplength=700,
        )
        subtitle.pack(anchor="w", pady=(4, 16))

        controls = ttk.Frame(container)
        controls.pack(fill="x")

        ttk.Label(controls, text="Language hint").grid(row=0, column=0, sticky="w")
        language_combo = ttk.Combobox(
            controls,
            textvariable=self.language_var,
            values=[label for label, _ in LANGUAGE_OPTIONS],
            state="readonly",
            width=24,
        )
        language_combo.grid(row=1, column=0, padx=(0, 12), sticky="w")

        ttk.Label(controls, text="Prompt hint").grid(row=0, column=1, sticky="w")
        prompt_entry = ttk.Entry(controls, textvariable=self.prompt_var, width=48)
        prompt_entry.grid(row=1, column=1, padx=(0, 12), sticky="ew")
        controls.columnconfigure(1, weight=1)

        self.record_button = ttk.Button(controls, text="Start Recording", command=self.toggle_recording)
        self.record_button.grid(row=1, column=2, sticky="e")

        status = ttk.Label(container, textvariable=self.status_var)
        status.pack(anchor="w", pady=(12, 8))

        self.output = ScrolledText(container, wrap="word", font=("Consolas", 10), height=22)
        self.output.pack(fill="both", expand=True)
        self.output.insert("1.0", "Transcript results will appear here.\n")
        self.output.configure(state="disabled")

    def toggle_recording(self) -> None:
        """Toggle between starting and stopping transcription mode.

        Args:
            None.

        Returns:
            None.
        """
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        """Start continuous listening and initialize the transcript document.

        Args:
            None.

        Returns:
            None.
        """
        if self.is_recording:
            return

        self._begin_session_log()
        self.is_recording = True
        self.record_button.configure(text="Stop Recording")
        self._set_status("Listening for speech...")
        self.worker_thread = threading.Thread(target=self._transcription_loop, daemon=True)
        self.worker_thread.start()

    def stop_recording(self) -> None:
        """Request the active transcription session to stop.

        Args:
            None.

        Returns:
            None.
        """
        self.is_recording = False
        self.record_button.configure(state="disabled")
        self._set_status("Stopping transcription mode...")

    def _transcription_loop(self) -> None:
        """Continuously capture speech segments until recording is stopped.

        Args:
            None.

        Returns:
            None.
        """
        while self.is_recording:
            self.root.after(0, lambda: self._set_status("Listening for speech..."))
            recording = self._record_until_pause()
            if not self.is_recording:
                break
            if not recording:
                continue

            result = self._process_segment(recording)
            self.root.after(0, lambda payload=result: self._finish_segment(payload))

        self.root.after(0, self._finish_recording_stop)

    def _record_until_pause(self) -> dict | None:
        """Capture microphone audio until a speech pause marks one segment.

        Args:
            None.

        Returns:
            A recording payload for one segment, or ``None`` when no segment is
            available because the session stopped first.
        """
        try:
            sounddevice, numpy = self.recorder._load_audio_dependencies()
        except Exception as exc:
            return self._error_result(str(exc))

        sample_rate = self.recorder._resolve_sample_rate(sounddevice, None)
        channels = 1
        block_duration = 0.1
        pre_roll_seconds = 0.4
        pause_seconds = 1.0
        minimum_speech_seconds = 0.35
        blocksize = max(1, int(sample_rate * block_duration))
        pre_roll_blocks = max(1, int(pre_roll_seconds / block_duration))
        required_silence_blocks = max(1, int(pause_seconds / block_duration))
        minimum_speech_blocks = max(1, int(minimum_speech_seconds / block_duration))
        peak_start = max(self.recognizer.min_detectable_peak_level * 4, 0.008)
        rms_start = max(self.recognizer.min_detectable_rms_level * 4, 0.002)
        peak_silence = self.recognizer.min_detectable_peak_level * 1.5
        rms_silence = self.recognizer.min_detectable_rms_level * 1.5

        chunk_queue: queue.Queue = queue.Queue()
        pre_roll: deque = deque(maxlen=pre_roll_blocks)
        collected: list = []
        started = False
        silence_blocks = 0
        speech_blocks = 0

        def callback(indata, frames, time, status) -> None:  # noqa: ARG001
            """Push each captured audio block into the local queue.

            Args:
                indata: Input audio buffer from the callback.
                frames: Number of audio frames in the callback.
                time: Timing metadata provided by the audio callback.
                status: Status value reported by the operation or callback.

            Returns:
                None.
            """
            if status:
                return
            chunk_queue.put(indata.copy())

        try:
            with sounddevice.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype=self.recorder._dtype,
                blocksize=blocksize,
                callback=callback,
            ):
                while self.is_recording:
                    try:
                        chunk = chunk_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue

                    levels = self._audio_levels(chunk, numpy)
                    if not started:
                        pre_roll.append(chunk)
                        if self._is_loud_enough(levels, peak_start, rms_start):
                            started = True
                            collected = list(pre_roll)
                            speech_blocks = 1
                            silence_blocks = 0
                            self.root.after(0, lambda: self._set_status("Speech detected. Waiting for pause to transcribe..."))
                        continue

                    collected.append(chunk)
                    if self._is_loud_enough(levels, peak_silence, rms_silence):
                        speech_blocks += 1
                        silence_blocks = 0
                    else:
                        silence_blocks += 1
                        if speech_blocks >= minimum_speech_blocks and silence_blocks >= required_silence_blocks:
                            break
        except Exception as exc:
            return self._error_result(f"Microphone recording failed: {exc}")

        if not self.is_recording:
            return None
        if not collected:
            return None

        try:
            recording = numpy.concatenate(collected, axis=0)
        except Exception as exc:
            return self._error_result(f"Failed to assemble recorded audio: {exc}")

        return self.recorder._write_wav_file(recording, sample_rate=sample_rate, channels=channels)

    def _process_segment(self, recording: dict) -> dict:
        """Transcribe one captured speech segment.

        Args:
            recording: Recording payload used for downstream processing.

        Returns:
            A segment result dictionary containing transcript and audio details.
        """
        if recording.get("status") != "success":
            return recording

        audio_path = Path(str(recording["file_path"]))
        preserved = self.recognizer.preserve_audio_file(audio_path, prefix="stt_segment")
        audio_payload = {
            "sample_rate": recording.get("sample_rate"),
            "duration_seconds": recording.get("duration_seconds"),
            "trimmed_seconds": recording.get("trimmed_seconds"),
            "peak_level": recording.get("peak_level"),
            "rms_level": recording.get("rms_level"),
        }
        if preserved.get("status") == "success":
            audio_payload["saved_file_path"] = preserved.get("file_path")
        else:
            audio_payload["save_error"] = preserved.get("message")

        if not self.recognizer.has_usable_audio(recording):
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass
            return {
                "status": "error",
                "message": "No voice was detected. Please check the microphone input volume or selected device.",
                "audio": audio_payload,
            }

        self.root.after(0, lambda: self._set_status("Transcribing speech..."))
        try:
            result = self.recognizer.transcribe_audio_file(
                audio_path,
                language=self.recognizer._normalize_language_hint(self._selected_culture()),
                prompt=self.prompt_var.get().strip() or None,
            )
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

        result["audio"] = audio_payload
        return result

    def _selected_culture(self) -> str:
        """Resolve the currently selected language hint value.

        Args:
            None.

        Returns:
            The selected culture code.
        """
        return self.language_map.get(self.language_var.get(), "ja-JP")

    def _finish_with_error(self, message: str) -> None:
        """Reset the UI after a fatal recording error.

        Args:
            message: Human-readable message text.

        Returns:
            None.
        """
        self.is_recording = False
        self.record_button.configure(text="Start Recording", state="normal")
        self._set_status(message)

    def _finish_segment(self, result: dict) -> None:
        """Append one finished transcript segment to the document view.

        Args:
            result: Result payload produced by a previous step.

        Returns:
            None.
        """
        if result.get("status") != "success":
            audio = result.get("audio") or {}
            saved_file_path = audio.get("saved_file_path")
            message = result.get("message", "Transcription failed.")
            if saved_file_path:
                message = f"{message} Saved WAV: {saved_file_path}"
            self._record_session_error(message)
            self._set_status(message)
            return

        transcript_text = str(result.get("text", "")).strip()
        audio = result.get("audio") or {}
        rms_level = audio.get("rms_level")
        saved_file_path = audio.get("saved_file_path")
        self._record_transcript_segment(transcript_text)
        if isinstance(rms_level, (int, float)) and rms_level < 0.015:
            self._set_status(
                f"Transcript appended. Audio level was very low, so recognition may be unreliable. Saved WAV: {saved_file_path or 'unavailable'}"
            )
        else:
            self._set_status(f"Transcript appended. Saved WAV: {saved_file_path or 'unavailable'}")

    def _finish_recording_stop(self) -> None:
        """Finalize the transcript session after listening stops.

        Args:
            None.

        Returns:
            None.
        """
        self.is_recording = False
        self.record_button.configure(text="Start Recording", state="normal")
        self.session_stopped_at = datetime.now()
        self._persist_session_log()
        saved_note = f" Saved transcript: {self.session_log_file_path}" if self.session_log_file_path else ""
        self._set_status(f"Transcription mode stopped.{saved_note}")

    def _open_tts_window(self) -> None:
        """Switch from the transcription window to the TTS window.

        Args:
            None.

        Returns:
            None.
        """
        if self.is_recording:
            self._set_status("Stop recording before switching windows.")
            return
        from .tts_gui import TTSTestApp

        self.root.destroy()
        new_root = tk.Tk()
        TTSTestApp(new_root)
        new_root.mainloop()

    def _set_status(self, message: str) -> None:
        """Update the status label shown in the window.

        Args:
            message: Human-readable message text.

        Returns:
            None.
        """
        self.status_var.set(message)

    def _begin_session_log(self) -> None:
        """Create a fresh transcript document for a new recording session.

        Args:
            None.

        Returns:
            None.
        """
        self.session_started_at = datetime.now()
        self.session_stopped_at = None
        self.session_segments = []
        self.session_errors = []
        output_dir = Path.cwd() / self.transcript_output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self.session_started_at.strftime("%Y%m%d_%H%M%S")
        self.session_log_file_path = str(output_dir / f"transcript_{timestamp}.txt")
        self._persist_session_log()

    def _record_transcript_segment(self, text: str) -> None:
        """Append one transcribed segment to the current document.

        Args:
            text: Input text handled by the current operation.

        Returns:
            None.
        """
        cleaned = text.strip()
        if not cleaned:
            return
        self.session_segments.append(cleaned)
        self._persist_session_log()

    def _record_session_error(self, message: str) -> None:
        """Append one session-level error message to the document.

        Args:
            message: Human-readable message text.

        Returns:
            None.
        """
        cleaned = message.strip()
        if not cleaned:
            return
        self.session_errors.append(cleaned)
        self._persist_session_log()

    def _persist_session_log(self) -> None:
        """Refresh the transcript view and save the document to disk.

        Args:
            None.

        Returns:
            None.
        """
        rendered = self._render_session_text()
        self._replace_output(rendered)
        if not self.session_log_file_path:
            return
        Path(self.session_log_file_path).write_text(rendered, encoding="utf-8")

    def _render_session_text(self) -> str:
        """Render the accumulated transcript as plain text.

        Args:
            None.

        Returns:
            A formatted transcript document for the current session.
        """
        lines = ["Speech transcript"]
        if self.session_started_at:
            lines.append(f"Started: {self.session_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.session_stopped_at:
            lines.append(f"Stopped: {self.session_stopped_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.session_log_file_path:
            lines.append(f"Saved to: {self.session_log_file_path}")
        lines.append("")

        if not self.session_segments and not self.session_errors:
            lines.append("Waiting for speech...")
            return "\n".join(lines).rstrip() + "\n"

        if self.session_segments:
            lines.append("Transcript")
            for index, segment in enumerate(self.session_segments, start=1):
                lines.append(f"{index}. {segment}")
            lines.append("")

        if self.session_errors:
            lines.append("Errors")
            for index, message in enumerate(self.session_errors, start=1):
                lines.append(f"{index}. {message}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _replace_output(self, text: str) -> None:
        """Replace the full contents of the output text area.

        Args:
            text: Full text that should be displayed in the output area.

        Returns:
            None.
        """
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.see("end")
        self.output.configure(state="disabled")

    def _audio_levels(self, chunk, numpy) -> tuple[float, float]:
        """Calculate normalized peak and RMS levels for an audio chunk.

        Args:
            chunk: Audio chunk used for level analysis.
            numpy: Imported NumPy module.

        Returns:
            A ``(peak, rms)`` tuple normalized to the 0-1 range.
        """
        audio = chunk.astype(numpy.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if audio.size == 0:
            return 0.0, 0.0
        peak = float(numpy.max(numpy.abs(audio))) / 32768.0
        rms = float(numpy.sqrt(numpy.mean(numpy.square(audio)))) / 32768.0
        return peak, rms

    @staticmethod
    def _is_loud_enough(levels: tuple[float, float], peak_threshold: float, rms_threshold: float) -> bool:
        """Check whether audio levels exceed either speech threshold.

        Args:
            levels: Peak and RMS audio levels.
            peak_threshold: Minimum peak level accepted as speech.
            rms_threshold: Minimum RMS level accepted as speech.

        Returns:
            ``True`` when the chunk should be treated as speech.
        """
        peak, rms = levels
        return peak >= peak_threshold or rms >= rms_threshold

    def _error_result(self, message: str) -> dict:
        """Create a standard error payload for the STT window flow.

        Args:
            message: Human-readable message text.

        Returns:
            A result dictionary with error details.
        """
        return {"status": "error", "message": message}


def main() -> None:
    """Launch the speech-to-text test application.

    Args:
        None.

    Returns:
        None.
    """
    root = tk.Tk()
    SpeechTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
