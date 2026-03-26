from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_tel.cli import build_parser


def test_listen_parser_defaults_to_transcript() -> None:
    """Test that listen parser defaults to transcript.
    
    Args:
        None.
    
    Returns:
        None.
    """
    parser = build_parser()

    args = parser.parse_args(["listen"])

    assert args.process == "transcript"
    assert args.timeout == 8
    assert args.prompt is None


def test_listen_parser_accepts_generate_pipeline_options() -> None:
    """Test that listen parser accepts generate pipeline options.
    
    Args:
        None.
    
    Returns:
        None.
    """
    parser = build_parser()

    args = parser.parse_args([
        "listen",
        "--process",
        "generate",
        "--culture",
        "ja-JP",
        "--prompt",
        "please prefer technical terms",
        "--language",
        "japanese",
        "--length",
        "90",
    ])

    assert args.process == "generate"
    assert args.culture == "ja-JP"
    assert args.prompt == "please prefer technical terms"
    assert args.language == "japanese"
    assert args.length == 90
