"""Voice assistant GUI: continuous speech-to-text, OpenAI reply, then text-to-speech."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

try:
    from .gui import LANGUAGE_OPTIONS
    from .openai_reply import OpenAITextResponder
    from .openai_speech import OpenAISpeechRecognizer
    from .openai_tts import OpenAITTS
    from .tts_gui import AGE_OPTIONS, GENDER_OPTIONS
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from ai_tel.gui import LANGUAGE_OPTIONS
    from ai_tel.openai_reply import OpenAITextResponder
    from ai_tel.openai_speech import OpenAISpeechRecognizer
    from ai_tel.openai_tts import OpenAITTS
    from ai_tel.tts_gui import AGE_OPTIONS, GENDER_OPTIONS


class VoiceAssistantApp:
    """Provide the voice assistant app component.
    """
    session_output_dir_name = "conversation_logs"

    def __init__(self, root: tk.Tk) -> None:
        """Initialize the voice assistant window and runtime state.

        Args:
            root: Tk root window used to host the application UI.

        Returns:
            None.
        """
        self.root = root
        self.root.title("AI Voice Assistant")
        self.root.geometry("860x620")

        self.recognizer = OpenAISpeechRecognizer()
        self.recorder = self.recognizer.recorder
        self.responder = OpenAITextResponder()
        self.tts = OpenAITTS()

        self.conversation_active = False
        self.worker_thread: threading.Thread | None = None
        self.language_map = {label: value for label, value in LANGUAGE_OPTIONS}
        self.gender_map = {label: value for label, value in GENDER_OPTIONS}
        self.age_map = {label: value for label, value in AGE_OPTIONS}

        self.language_var = tk.StringVar(value=LANGUAGE_OPTIONS[0][0])
        self.gender_var = tk.StringVar(value=GENDER_OPTIONS[2][0])
        self.age_var = tk.StringVar(value=AGE_OPTIONS[2][0])
        self.system_prompt_var = tk.StringVar(value="Be friendly and helpful.")
        self.status_var = tk.StringVar(value="Ready")
        self.last_speech_file_path: str | None = self._find_latest_reply_wav()
        # Keep roughly the last 6 turns (user + assistant pairs) for short-term conversation memory.
        self.conversation_history: deque[dict[str, str]] = deque(maxlen=12)
        self.session_started_at: datetime | None = None
        self.session_stopped_at: datetime | None = None
        self.session_turns: list[dict[str, str]] = []
        self.session_errors: list[str] = []
        self.session_log_file_path: str | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Create and layout all widgets used by the conversation window.

        Args:
            None.

        Returns:
            None.
        """
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        nav = ttk.Frame(container)
        nav.pack(fill="x")
        ttk.Label(nav, text="Voice Assistant", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(nav, text="Open Speech To Text", command=self._open_stt_window).pack(side="right")
        ttk.Button(nav, text="Open Text To Speech", command=self._open_tts_window).pack(side="right", padx=(0, 8))

        title = ttk.Label(container, text="Voice Assistant", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w", pady=(10, 0))

        subtitle = ttk.Label(
            container,
            text="Start conversation mode once, speak naturally, pause to get a reply, and press the button again to end the session.",
            wraplength=800,
        )
        subtitle.pack(anchor="w", pady=(4, 16))

        controls = ttk.Frame(container)
        controls.pack(fill="x")

        ttk.Label(controls, text="Language hint").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.language_var,
            values=[label for label, _ in LANGUAGE_OPTIONS],
            state="readonly",
            width=22,
        ).grid(row=1, column=0, padx=(0, 12), sticky="w")

        ttk.Label(controls, text="Voice gender").grid(row=0, column=1, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.gender_var,
            values=[label for label, _ in GENDER_OPTIONS],
            state="readonly",
            width=16,
        ).grid(row=1, column=1, padx=(0, 12), sticky="w")

        ttk.Label(controls, text="Voice age").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.age_var,
            values=[label for label, _ in AGE_OPTIONS],
            state="readonly",
            width=16,
        ).grid(row=1, column=2, padx=(0, 12), sticky="w")

        self.record_button = ttk.Button(controls, text="Start Conversation", command=self.toggle_conversation)
        self.record_button.grid(row=1, column=3, sticky="e")
        self.play_last_button = ttk.Button(controls, text="Play Last Reply", command=self.play_last_reply)
        self.play_last_button.grid(row=1, column=4, padx=(8, 0), sticky="e")
        self.open_last_button = ttk.Button(controls, text="Open Last WAV", command=self.open_last_reply)
        self.open_last_button.grid(row=1, column=5, padx=(8, 0), sticky="e")

        ttk.Label(container, text="Assistant instruction").pack(anchor="w", pady=(14, 6))
        ttk.Entry(container, textvariable=self.system_prompt_var, width=100).pack(fill="x")

        status = ttk.Label(container, textvariable=self.status_var)
        status.pack(anchor="w", pady=(12, 8))

        self.output = ScrolledText(container, wrap="word", font=("Consolas", 10), height=24)
        self.output.pack(fill="both", expand=True)
        self.output.insert("1.0", "Conversation results will appear here.\n")
        self.output.configure(state="disabled")

    def toggle_conversation(self) -> None:
        """Toggle between starting and stopping conversation mode.

        Args:
            None.

        Returns:
            None.
        """
        if self.conversation_active:
            self.stop_conversation()
        else:
            self.start_conversation()

    def start_conversation(self) -> None:
        """Start a new conversation session and launch the worker loop.

        Args:
            None.

        Returns:
            None.
        """
        if self.conversation_active:
            return

        self.conversation_history.clear()
        self._begin_session_log()
        self.conversation_active = True
        self.record_button.configure(text="Stop Conversation")
        self._set_status("Conversation mode started. Listening for speech...")
        self.worker_thread = threading.Thread(target=self._conversation_loop, daemon=True)
        self.worker_thread.start()

    def stop_conversation(self) -> None:
        """Request the active conversation session to stop.

        Args:
            None.

        Returns:
            None.
        """
        self.conversation_active = False
        self.record_button.configure(state="disabled")
        self._set_status("Stopping conversation mode...")

    def play_last_reply(self) -> None:
        """Replay the latest generated assistant WAV file.

        Args:
            None.

        Returns:
            None.
        """
        self.last_speech_file_path = self._find_latest_reply_wav() or self.last_speech_file_path
        if not self.last_speech_file_path:
            self._set_status("No reply audio is available yet.")
            return

        self.play_last_button.configure(state="disabled")
        self._set_status("Replaying the last reply...")
        threading.Thread(target=self._play_last_reply_worker, daemon=True).start()

    def _play_last_reply_worker(self) -> None:
        """Play the latest reply audio on a background thread.

        Args:
            None.

        Returns:
            None.
        """
        path = self.last_speech_file_path
        if not path:
            self.root.after(0, lambda: self._finish_manual_playback({"status": "error", "message": "No reply audio is available yet."}))
            return

        try:
            backend = self.tts.play_wav_file(path)
            result = {"status": "success", "file_path": path, "playback_backend": backend}
        except Exception as exc:
            result = {"status": "error", "message": f"Manual playback failed: {exc}", "file_path": path}
        self.root.after(0, lambda payload=result: self._finish_manual_playback(payload))

    def open_last_reply(self) -> None:
        """Open the latest assistant reply WAV with the system player.

        Args:
            None.

        Returns:
            None.
        """
        self.last_speech_file_path = self._find_latest_reply_wav() or self.last_speech_file_path
        if not self.last_speech_file_path:
            self._set_status("No reply audio is available yet.")
            return

        try:
            self.tts.open_wav_with_system_player(self.last_speech_file_path)
        except Exception as exc:
            self._set_status(f"Failed to open the last reply WAV: {exc}")
            return

        self._set_status(f"Opened last reply WAV: {self.last_speech_file_path}")

    def _finish_manual_playback(self, result: dict) -> None:
        """Apply UI updates after manual playback completes.

        Args:
            result: Playback result payload produced by the worker thread.

        Returns:
            None.
        """
        self.play_last_button.configure(state="normal")
        if result.get("status") != "success":
            self._set_status(result.get("message", "Manual playback failed."))
            return

        self._set_status(
            f"Last reply replayed with {result.get('playback_backend', 'unknown')} from {result.get('file_path', '')}"
        )

    def _conversation_loop(self) -> None:
        """Continuously record, transcribe, answer, and speak until stopped.

        Args:
            None.

        Returns:
            None.
        """
        while self.conversation_active:
            self.root.after(0, lambda: self._set_status("Listening for speech..."))
            recording = self._record_until_pause()
            if not self.conversation_active:
                break
            if not recording:
                continue

            result = self._process_turn(recording)
            self.root.after(0, lambda payload=result: self._finish_turn(payload))

        self.root.after(0, self._finish_conversation_stop)

    def _record_until_pause(self) -> dict | None:
        """Capture microphone audio until speech is followed by a pause.

        Args:
            None.

        Returns:
            A recording payload when capture succeeds, or ``None`` when the
            session is stopped before a usable clip is produced.
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
            """Push each captured audio block into the worker queue.

            Args:
                indata: Raw audio frame data from the input stream.
                frames: Number of frames in the current block.
                time: Stream timing metadata from sounddevice.
                status: Status flags reported by the audio backend.

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
                while self.conversation_active:
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
                            self.root.after(0, lambda: self._set_status("Speech detected. Waiting for pause to reply..."))
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

        if not self.conversation_active:
            return None
        if not collected:
            return None

        try:
            recording = numpy.concatenate(collected, axis=0)
        except Exception as exc:
            return self._error_result(f"Failed to assemble recorded audio: {exc}")

        return self.recorder._write_wav_file(recording, sample_rate=sample_rate, channels=channels)

    def _process_turn(self, recording: dict) -> dict:
        """Run STT, text reply generation, and TTS for one conversation turn.

        Args:
            recording: Audio capture payload returned by the recorder.

        Returns:
            A result dictionary containing transcript, reply, speech, and error
            information for the turn.
        """
        if recording.get("status") != "success":
            return recording

        audio_path = Path(str(recording["file_path"]))
        preserved = self.recognizer.preserve_audio_file(audio_path, prefix="voice_assistant_recording")

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

        language_hint = self.recognizer._normalize_language_hint(self._selected_culture())
        self.root.after(0, lambda: self._set_status("Transcribing speech..."))
        try:
            transcript = self.recognizer.transcribe_audio_file(audio_path, language=language_hint)
        finally:
            try:
                audio_path.unlink(missing_ok=True)
            except Exception:
                pass

        if transcript.get("status") != "success":
            transcript["audio"] = audio_payload
            return transcript

        self.root.after(0, lambda: self._set_status("Generating assistant reply..."))
        transcript_text = str(transcript.get("text", "")).strip()
        reply = self.responder.generate_reply(
            user_text=transcript_text,
            system_prompt=self.system_prompt_var.get().strip() or None,
            language_hint=language_hint,
            # Send the rolling history as a plain list because the responder only needs read access.
            conversation_history=list(self.conversation_history),
        )
        if reply.get("status") != "success":
            return {
                "status": "error",
                "message": reply.get("message", "Failed to generate assistant reply."),
                "transcript": transcript,
                "reply": reply,
                "audio": audio_payload,
            }

        reply_text = str(reply.get("text", "")).strip()
        # Save the completed turn after a successful reply so the next request can reuse it.
        if transcript_text and reply_text:
            self.conversation_history.append({"role": "user", "content": transcript_text})
            self.conversation_history.append({"role": "assistant", "content": reply_text})

        if not self.conversation_active:
            return {
                "status": "success",
                "transcript": transcript,
                "reply": reply,
                "audio": audio_payload,
                "speech": {"status": "skipped", "message": "Conversation stopped before playback."},
            }

        self.root.after(0, lambda: self._set_status("Playing assistant reply..."))
        speech = self.tts.speak_text(
            text=str(reply.get("text", "")),
            gender=self.gender_map.get(self.gender_var.get(), "neutral"),
            age_group=self.age_map.get(self.age_var.get(), "adult"),
        )
        result = {
            "status": speech.get("status", "success"),
            "transcript": transcript,
            "reply": reply,
            "speech": speech,
            "audio": audio_payload,
        }

        if speech.get("status") != "success":
            result["message"] = speech.get("message", "Failed to generate or play assistant speech.")
        return result

    def _finish_turn(self, result: dict) -> None:
        """Update UI and session log after one turn finishes.

        Args:
            result: Turn result payload returned by ``_process_turn``.

        Returns:
            None.
        """
        if result.get("status") != "success":
            audio = result.get("audio") or {}
            saved = audio.get("saved_file_path")
            message = result.get("message", "Voice assistant flow failed.")
            if saved:
                message = f"{message} Saved WAV: {saved}"
            transcript_text = str(result.get("transcript", {}).get("text", "")).strip()
            self._record_session_error(message=message, transcript_text=transcript_text)
            self._set_status(message)
            return

        transcript_text = str(result.get("transcript", {}).get("text", "")).strip()
        reply_text = str(result.get("reply", {}).get("text", "")).strip()
        self._record_session_turn(user_text=transcript_text, assistant_text=reply_text)
        speech_file_path = str(result.get("speech", {}).get("file_path", "")).strip()
        if speech_file_path:
            self.last_speech_file_path = speech_file_path
        backend = str(result.get("speech", {}).get("playback_backend", "unknown"))
        self._set_status(
            f"Reply played with {backend}. WAV: {self.last_speech_file_path or 'n/a'}"
        )

    def _finish_conversation_stop(self) -> None:
        """Finalize the active session after the worker loop exits.

        Args:
            None.

        Returns:
            None.
        """
        self.record_button.configure(text="Start Conversation", state="normal")
        self.conversation_active = False
        self.session_stopped_at = datetime.now()
        self._persist_session_log()
        saved_note = f" Saved transcript: {self.session_log_file_path}" if self.session_log_file_path else ""
        self._set_status(f"Conversation mode stopped.{saved_note}")

    def _find_latest_reply_wav(self) -> str | None:
        """Find the most recently generated assistant reply WAV file.

        Args:
            None.

        Returns:
            The latest WAV file path, or ``None`` when no reply audio exists.
        """
        output_dir = Path.cwd() / self.tts.output_dir_name
        if not output_dir.exists():
            return None

        candidates = sorted(output_dir.glob("tts_*.wav"), key=lambda item: item.stat().st_mtime, reverse=True)
        if not candidates:
            return None
        return str(candidates[0])

    def _selected_culture(self) -> str:
        """Resolve the currently selected language hint value.

        Args:
            None.

        Returns:
            A culture code such as ``ja-JP``.
        """
        return self.language_map.get(self.language_var.get(), "ja-JP")

    def _audio_levels(self, chunk, numpy) -> tuple[float, float]:
        """Calculate normalized peak and RMS levels for an audio chunk.

        Args:
            chunk: Input audio block captured from the microphone.
            numpy: Imported NumPy module used for numeric processing.

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
            levels: Tuple containing normalized peak and RMS audio levels.
            peak_threshold: Minimum accepted peak level.
            rms_threshold: Minimum accepted RMS level.

        Returns:
            ``True`` when the chunk is considered speech-like, else ``False``.
        """
        peak, rms = levels
        return peak >= peak_threshold or rms >= rms_threshold

    def _error_result(self, message: str) -> dict:
        """Create a standard error payload used by the conversation flow.

        Args:
            message: Human-readable error description.

        Returns:
            A result dictionary with ``status`` and ``message`` keys.
        """
        return {
            "status": "error",
            "message": message,
        }

    def _open_stt_window(self) -> None:
        """Switch from the conversation window to the speech-to-text window.

        Args:
            None.

        Returns:
            None.
        """
        from .gui import SpeechTestApp

        self.root.destroy()
        new_root = tk.Tk()
        SpeechTestApp(new_root)
        new_root.mainloop()

    def _open_tts_window(self) -> None:
        """Switch from the conversation window to the text-to-speech window.

        Args:
            None.

        Returns:
            None.
        """
        from .tts_gui import TTSTestApp

        self.root.destroy()
        new_root = tk.Tk()
        TTSTestApp(new_root)
        new_root.mainloop()

    def _set_status(self, message: str) -> None:
        """Update the status label shown in the window.

        Args:
            message: Status text to present to the user.

        Returns:
            None.
        """
        self.status_var.set(message)

    def _begin_session_log(self) -> None:
        """Create a fresh transcript file for a newly started session.

        Args:
            None.

        Returns:
            None.
        """
        self.session_started_at = datetime.now()
        self.session_stopped_at = None
        self.session_turns = []
        self.session_errors = []
        output_dir = Path.cwd() / self.session_output_dir_name
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = self.session_started_at.strftime("%Y%m%d_%H%M%S")
        self.session_log_file_path = str(output_dir / f"conversation_{timestamp}.txt")
        self._persist_session_log()

    def _record_session_turn(self, user_text: str, assistant_text: str) -> None:
        """Append one successful user/assistant exchange to the session log.

        Args:
            user_text: Text transcribed from the user's speech.
            assistant_text: Text reply generated by the assistant.

        Returns:
            None.
        """
        if not user_text and not assistant_text:
            return
        self.session_turns.append(
            {
                "user": user_text,
                "assistant": assistant_text,
            }
        )
        self._persist_session_log()

    def _record_session_error(self, message: str, transcript_text: str | None = None) -> None:
        """Append an error entry to the current session transcript.

        Args:
            message: Error message to record in the transcript.
            transcript_text: Optional user transcript associated with the error.

        Returns:
            None.
        """
        if transcript_text:
            self.session_turns.append(
                {
                    "user": transcript_text,
                    "assistant": "",
                }
            )
        if message:
            self.session_errors.append(message)
        self._persist_session_log()

    def _persist_session_log(self) -> None:
        """Refresh the transcript view and save it to disk when possible.

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
        """Render the current session state into readable transcript text.

        Args:
            None.

        Returns:
            A formatted plain-text transcript for the current session.
        """
        lines = ["Conversation transcript"]
        if self.session_started_at:
            lines.append(f"Started: {self.session_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.session_stopped_at:
            lines.append(f"Stopped: {self.session_stopped_at.strftime('%Y-%m-%d %H:%M:%S')}")
        if self.session_log_file_path:
            lines.append(f"Saved to: {self.session_log_file_path}")
        lines.append("")

        if not self.session_turns and not self.session_errors:
            lines.append("Waiting for conversation...")
            return "\n".join(lines).rstrip() + "\n"

        for index, turn in enumerate(self.session_turns, start=1):
            lines.append(f"Turn {index}")
            user_text = turn.get("user", "").strip() or "(no transcript)"
            assistant_text = turn.get("assistant", "").strip() or "(no assistant reply)"
            lines.append(f"User: {user_text}")
            lines.append(f"Assistant: {assistant_text}")
            lines.append("")

        if self.session_errors:
            lines.append("Errors")
            for index, message in enumerate(self.session_errors, start=1):
                lines.append(f"{index}. {message}")
            lines.append("")

        return "\n".join(lines).rstrip() + "\n"

    def _replace_output(self, text: str) -> None:
        """Replace the transcript text shown in the output widget.

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

    def _append_output(self, payload: dict) -> None:
        """Append a JSON payload to the output widget.

        Args:
            payload: Structured data to render as indented JSON text.

        Returns:
            None.
        """
        self.output.configure(state="normal")
        self.output.insert("end", json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n")
        self.output.see("end")
        self.output.configure(state="disabled")



def main() -> None:
    """Launch the voice assistant application.

    Args:
        None.

    Returns:
        None.
    """
    root = tk.Tk()
    VoiceAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
