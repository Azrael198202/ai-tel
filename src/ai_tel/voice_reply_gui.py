"""Voice assistant GUI: continuous speech-to-text, OpenAI reply, then text-to-speech."""

from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from collections import deque
from pathlib import Path
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from .gui import LANGUAGE_OPTIONS
from .openai_reply import OpenAITextResponder
from .openai_speech import OpenAISpeechRecognizer
from .openai_tts import OpenAITTS
from .tts_gui import AGE_OPTIONS, GENDER_OPTIONS


class VoiceAssistantApp:
    def __init__(self, root: tk.Tk) -> None:
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

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        nav = ttk.Frame(container)
        nav.pack(fill="x")
        ttk.Label(nav, text="Voice Assistant", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(nav, text="Open Speech To Text", command=self._open_stt_window).pack(side="right")
        ttk.Button(nav, text="Open Text To Speech", command=self._open_tts_window).pack(side="right", padx=(0, 8))

        title = ttk.Label(container, text="OpenAI Voice Assistant", font=("Segoe UI", 16, "bold"))
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

        ttk.Label(container, text="Assistant instruction").pack(anchor="w", pady=(14, 6))
        ttk.Entry(container, textvariable=self.system_prompt_var, width=100).pack(fill="x")

        status = ttk.Label(container, textvariable=self.status_var)
        status.pack(anchor="w", pady=(12, 8))

        self.output = ScrolledText(container, wrap="word", font=("Consolas", 10), height=24)
        self.output.pack(fill="both", expand=True)
        self.output.insert("1.0", "Conversation results will appear here.\n")
        self.output.configure(state="disabled")

    def toggle_conversation(self) -> None:
        if self.conversation_active:
            self.stop_conversation()
        else:
            self.start_conversation()

    def start_conversation(self) -> None:
        if self.conversation_active:
            return

        self.conversation_active = True
        self.record_button.configure(text="Stop Conversation")
        self._set_status("Conversation mode started. Listening for speech...")
        self.worker_thread = threading.Thread(target=self._conversation_loop, daemon=True)
        self.worker_thread.start()

    def stop_conversation(self) -> None:
        self.conversation_active = False
        self.record_button.configure(state="disabled")
        self._set_status("Stopping conversation mode...")

    def _conversation_loop(self) -> None:
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
        reply = self.responder.generate_reply(
            user_text=str(transcript.get("text", "")),
            system_prompt=self.system_prompt_var.get().strip() or None,
            language_hint=language_hint,
        )
        if reply.get("status") != "success":
            return {
                "status": "error",
                "message": reply.get("message", "Failed to generate assistant reply."),
                "transcript": transcript,
                "reply": reply,
                "audio": audio_payload,
            }

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
        if result.get("status") != "success":
            audio = result.get("audio") or {}
            saved = audio.get("saved_file_path")
            message = result.get("message", "Voice assistant flow failed.")
            if saved:
                message = f"{message} Saved WAV: {saved}"
            self._set_status(message)
            self._append_output({"error": result})
            return

        transcript_text = str(result.get("transcript", {}).get("text", "")).strip()
        reply_text = str(result.get("reply", {}).get("text", "")).strip()
        self._set_status(
            f"Reply played. Heard: {transcript_text[:40] or 'n/a'} | Replied: {reply_text[:40] or 'n/a'}"
        )
        self._append_output(result)

    def _finish_conversation_stop(self) -> None:
        self.record_button.configure(text="Start Conversation", state="normal")
        self.conversation_active = False
        self._set_status("Conversation mode stopped.")

    def _selected_culture(self) -> str:
        return self.language_map.get(self.language_var.get(), "ja-JP")

    def _audio_levels(self, chunk, numpy) -> tuple[float, float]:
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
        peak, rms = levels
        return peak >= peak_threshold or rms >= rms_threshold

    def _error_result(self, message: str) -> dict:
        return {
            "status": "error",
            "message": message,
        }

    def _open_stt_window(self) -> None:
        from .gui import SpeechTestApp

        self.root.destroy()
        new_root = tk.Tk()
        SpeechTestApp(new_root)
        new_root.mainloop()

    def _open_tts_window(self) -> None:
        from .tts_gui import TTSTestApp

        self.root.destroy()
        new_root = tk.Tk()
        TTSTestApp(new_root)
        new_root.mainloop()

    def _set_status(self, message: str) -> None:
        self.status_var.set(message)

    def _append_output(self, payload: dict) -> None:
        self.output.configure(state="normal")
        self.output.insert("end", json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n")
        self.output.see("end")
        self.output.configure(state="disabled")



def main() -> None:
    root = tk.Tk()
    VoiceAssistantApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
