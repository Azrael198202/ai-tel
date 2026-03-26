"""Backward-compatible entry point for the legacy script name."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_tel import AITextProcessor
from ai_tel.cli import main

__all__ = ["AITextProcessor", "main"]


if __name__ == "__main__":
    main()
