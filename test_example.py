"""Legacy compatibility runner for the new project structure."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from ai_tel import AITextProcessor


def main() -> None:
    """Main.
    
    Args:
        None.
    
    Returns:
        None.
    """
    processor = AITextProcessor()
    print(processor.detect_language("Hello from the compatibility script."))
    print(processor.analyze_text("This project has been reorganized into a package."))
    print(processor.generate_text("project structure", language="english", length=100))


if __name__ == "__main__":
    main()
