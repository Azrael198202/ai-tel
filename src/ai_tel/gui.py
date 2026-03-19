"""Simple desktop GUI for testing microphone transcription."""

from __future__ import annotations

import json
import threading
import tkinter as tk
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
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AI Speech Test")
        self.root.geometry("760x520")

        self.recognizer = OpenAISpeechRecognizer()
        self.recorder = self.recognizer.recorder
        self.is_recording = False

        self.language_map = {label: value for label, value in LANGUAGE_OPTIONS}
        self.language_var = tk.StringVar(value=LANGUAGE_OPTIONS[0][0])
        self.prompt_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        nav = ttk.Frame(container)
        nav.pack(fill="x")
        ttk.Label(nav, text="Speech To Text", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(nav, text="Go To Text To Speech", command=self._open_tts_window).pack(side="right")

        title = ttk.Label(container, text="OpenAI Speech Test", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w", pady=(10, 0))

        subtitle = ttk.Label(container, text="Click once to start recording. Click again to stop and transcribe with gpt-4o-transcribe.")
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
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        result = self.recorder.start_recording()
        if result.get("status") != "success":
            self._set_status(result.get("message", "Failed to start recording."))
            return

        self.is_recording = True
        self.record_button.configure(text="Stop Recording")
        self._set_status("Recording... click again to stop.")

    def stop_recording(self) -> None:
        self.record_button.configure(state="disabled")
        self._set_status("Stopping recording and sending audio for transcription...")
        threading.Thread(target=self._stop_and_transcribe, daemon=True).start()

    def _stop_and_transcribe(self) -> None:
        recording = self.recorder.stop_recording_to_wav()
        if recording.get("status") != "success":
            self.root.after(0, lambda: self._finish_with_error(recording.get("message", "Failed to stop recording.")))
            return

        audio_path = Path(str(recording["file_path"]))
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

        self.root.after(0, lambda: self._finish_transcription(result))

    def _selected_culture(self) -> str:
        return self.language_map.get(self.language_var.get(), "ja-JP")

    def _finish_with_error(self, message: str) -> None:
        self.is_recording = False
        self.record_button.configure(text="Start Recording", state="normal")
        self._set_status(message)

    def _finish_transcription(self, result: dict) -> None:
        self.is_recording = False
        self.record_button.configure(text="Start Recording", state="normal")

        if result.get("status") != "success":
            self._set_status(result.get("message", "Transcription failed."))
            self._append_output({"error": result})
            return

        self._set_status("Transcription complete.")
        self._append_output(result)

    def _open_tts_window(self) -> None:
        if self.is_recording:
            self._set_status("Stop recording before switching windows.")
            return
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
    SpeechTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
