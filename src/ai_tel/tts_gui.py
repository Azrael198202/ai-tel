"""Simple desktop GUI for testing text-to-speech."""

from __future__ import annotations

import json
import threading
import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from .openai_tts import OpenAITTS

GENDER_OPTIONS = [
    ("Female", "female"),
    ("Male", "male"),
    ("Neutral", "neutral"),
]

AGE_OPTIONS = [
    ("Child", "child"),
    ("Young Adult", "young_adult"),
    ("Adult", "adult"),
    ("Senior", "senior"),
]


class TTSTestApp:
    """Provide the ttstest app component.
    """
    def __init__(self, root: tk.Tk) -> None:
        """Initialize the TTSTestApp instance.
        
        Args:
            root: Tk root window used by the UI.
        
        Returns:
            None.
        """
        self.root = root
        self.root.title("AI Text To Speech Test")
        self.root.geometry("780x560")

        self.tts = OpenAITTS()
        self.gender_map = {label: value for label, value in GENDER_OPTIONS}
        self.age_map = {label: value for label, value in AGE_OPTIONS}

        self.gender_var = tk.StringVar(value=GENDER_OPTIONS[2][0])
        self.age_var = tk.StringVar(value=AGE_OPTIONS[2][0])
        self.status_var = tk.StringVar(value="Ready")

        self._build_ui()

    def _build_ui(self) -> None:
        """Build ui.
        
        Args:
            None.
        
        Returns:
            None.
        """
        container = ttk.Frame(self.root, padding=16)
        container.pack(fill="both", expand=True)

        nav = ttk.Frame(container)
        nav.pack(fill="x")
        ttk.Label(nav, text="Text To Speech", font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Button(nav, text="Go To Speech To Text", command=self._open_stt_window).pack(side="right")

        title = ttk.Label(container, text="Text To Speech Test", font=("Segoe UI", 16, "bold"))
        title.pack(anchor="w", pady=(10, 0))

        subtitle = ttk.Label(container, text="Enter text, choose gender and age style, then play generated speech.")
        subtitle.pack(anchor="w", pady=(4, 16))

        controls = ttk.Frame(container)
        controls.pack(fill="x")

        ttk.Label(controls, text="Gender").grid(row=0, column=0, sticky="w")
        gender_combo = ttk.Combobox(
            controls,
            textvariable=self.gender_var,
            values=[label for label, _ in GENDER_OPTIONS],
            state="readonly",
            width=16,
        )
        gender_combo.grid(row=1, column=0, padx=(0, 12), sticky="w")

        ttk.Label(controls, text="Age").grid(row=0, column=1, sticky="w")
        age_combo = ttk.Combobox(
            controls,
            textvariable=self.age_var,
            values=[label for label, _ in AGE_OPTIONS],
            state="readonly",
            width=16,
        )
        age_combo.grid(row=1, column=1, padx=(0, 12), sticky="w")

        self.play_button = ttk.Button(controls, text="Generate And Play", command=self.generate_and_play)
        self.play_button.grid(row=1, column=2, sticky="e")

        ttk.Label(container, text="Text").pack(anchor="w", pady=(16, 6))

        self.input_text = ScrolledText(container, wrap="word", font=("Segoe UI", 10), height=10)
        self.input_text.pack(fill="x")
        self.input_text.insert("1.0", "こんにちは。これは音声合成のテストです。")

        status = ttk.Label(container, textvariable=self.status_var)
        status.pack(anchor="w", pady=(12, 8))

        self.output = ScrolledText(container, wrap="word", font=("Consolas", 10), height=14)
        self.output.pack(fill="both", expand=True)
        self.output.insert("1.0", "Speech generation details will appear here.\n")
        self.output.configure(state="disabled")

    def generate_and_play(self) -> None:
        """Generate and play.
        
        Args:
            None.
        
        Returns:
            None.
        """
        text = self.input_text.get("1.0", "end").strip()
        if not text:
            self._set_status("Please enter some text first.")
            return

        self.play_button.configure(state="disabled")
        self._set_status("Generating speech and playing audio...")
        threading.Thread(target=self._generate_and_play_worker, args=(text,), daemon=True).start()

    def _generate_and_play_worker(self, text: str) -> None:
        """Generate and play worker.
        
        Args:
            text: Input text handled by the current operation.
        
        Returns:
            None.
        """
        result = self.tts.speak_text(
            text=text,
            gender=self.gender_map.get(self.gender_var.get(), "neutral"),
            age_group=self.age_map.get(self.age_var.get(), "adult"),
        )
        self.root.after(0, lambda: self._finish(result))

    def _finish(self, result: dict) -> None:
        """Finalize.
        
        Args:
            result: Result payload produced by a previous step.
        
        Returns:
            None.
        """
        self.play_button.configure(state="normal")

        if result.get("status") != "success":
            self._set_status(result.get("message", "Speech generation failed."))
            self._append_output({"error": result})
            return

        self._set_status("Speech playback complete.")
        self._append_output(result)

    def _open_stt_window(self) -> None:
        """Open stt window.
        
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

    def _set_status(self, message: str) -> None:
        """Set status.
        
        Args:
            message: Human-readable message text.
        
        Returns:
            None.
        """
        self.status_var.set(message)

    def _append_output(self, payload: dict) -> None:
        """Append output.
        
        Args:
            payload: Structured payload to render or process.
        
        Returns:
            None.
        """
        self.output.configure(state="normal")
        self.output.insert("end", json.dumps(payload, ensure_ascii=False, indent=2) + "\n\n")
        self.output.see("end")
        self.output.configure(state="disabled")


def main() -> None:
    """Main.
    
    Args:
        None.
    
    Returns:
        None.
    """
    root = tk.Tk()
    TTSTestApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
