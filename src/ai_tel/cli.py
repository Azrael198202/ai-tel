"""Command-line interface for ai-tel."""

from __future__ import annotations

import argparse
import json
import sys

from .openai_speech import OpenAISpeechRecognizer
from .processor import AITextProcessor


LISTEN_PROCESS_CHOICES = ["transcript", "detect", "analyze", "generate"]


def build_parser() -> argparse.ArgumentParser:
    """Build parser.
    
    Args:
        None.
    
    Returns:
        The argparse.ArgumentParser value produced by this callable.
    """
    parser = argparse.ArgumentParser(description="AI text processing toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser("detect", help="Detect a text language")
    detect_parser.add_argument("text", help="Input text")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze a text")
    analyze_parser.add_argument("text", help="Input text")

    generate_parser = subparsers.add_parser("generate", help="Generate template text")
    generate_parser.add_argument("prompt", help="Generation prompt")
    generate_parser.add_argument("--language", default="english", choices=["english", "chinese", "japanese"], help="Generation language")
    generate_parser.add_argument("--length", type=int, default=120, help="Maximum output length")

    listen_parser = subparsers.add_parser("listen", help="Record microphone audio and transcribe it with OpenAI gpt-4o-transcribe before feeding it into the text pipeline")
    listen_parser.add_argument("--timeout", type=int, default=8, help="Recording duration in seconds")
    listen_parser.add_argument("--culture", default=None, help="Optional language hint such as en-US, ja-JP, or zh-CN")
    listen_parser.add_argument("--prompt", default=None, help="Optional transcription prompt to improve recognition")
    listen_parser.add_argument("--process", default="transcript", choices=LISTEN_PROCESS_CHOICES, help="How to process the recognized text")
    listen_parser.add_argument("--language", default="english", choices=["english", "chinese", "japanese"], help="Generation language when --process generate is used")
    listen_parser.add_argument("--length", type=int, default=120, help="Maximum output length when --process generate is used")

    return parser


def _configure_stdout() -> None:
    """Configure stdout.
    
    Args:
        None.
    
    Returns:
        None.
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")


def _process_transcript(processor: AITextProcessor, transcript: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    """Process transcript.
    
    Args:
        processor: Text processor instance used by the command flow.
        transcript: Transcript payload produced by speech recognition.
        args: Parsed command-line arguments.
    
    Returns:
        The dict[str, object] value produced by this callable.
    """
    text = str(transcript["text"])

    if args.process == "detect":
        output = processor.detect_language(text)
        return {
            "status": output.get("status", "success"),
            "transcript": transcript,
            "result": output,
        }

    if args.process == "analyze":
        output = processor.analyze_text(text)
        return {
            "status": output.get("status", "success"),
            "transcript": transcript,
            "result": output,
        }

    if args.process == "generate":
        output = processor.generate_text(text, language=args.language, length=args.length)
        return {
            "status": output.get("status", "success"),
            "transcript": transcript,
            "result": output,
        }

    return transcript


def main() -> None:
    """Main.
    
    Args:
        None.
    
    Returns:
        None.
    """
    _configure_stdout()
    parser = build_parser()
    args = parser.parse_args()
    processor = AITextProcessor()

    if args.command == "detect":
        result = processor.detect_language(args.text)
    elif args.command == "analyze":
        result = processor.analyze_text(args.text)
    elif args.command == "generate":
        result = processor.generate_text(args.prompt, language=args.language, length=args.length)
    else:
        recognizer = OpenAISpeechRecognizer()
        transcript = recognizer.listen_once(timeout=args.timeout, culture=args.culture, prompt=args.prompt)
        if transcript.get("status") != "success":
            result = transcript
        else:
            result = _process_transcript(processor, transcript, args)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
