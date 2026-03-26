from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ai_tel import AITextProcessor


def test_detect_language_returns_success_for_english() -> None:
    """Test that detect language returns success for english.
    
    Args:
        None.
    
    Returns:
        None.
    """
    processor = AITextProcessor()

    result = processor.detect_language("Hello, this is a short test.")

    assert result["status"] == "success"
    assert result["language_code"] == "en"
    assert result["language_name"] == "English"


def test_detect_language_rejects_empty_text() -> None:
    """Test that detect language rejects empty text.
    
    Args:
        None.
    
    Returns:
        None.
    """
    processor = AITextProcessor()

    result = processor.detect_language("   ")

    assert result["status"] == "error"
    assert "empty" in result["message"].lower()


def test_analyze_text_returns_basic_metrics() -> None:
    """Test that analyze text returns basic metrics.
    
    Args:
        None.
    
    Returns:
        None.
    """
    processor = AITextProcessor()

    result = processor.analyze_text("This is a clear sentence. This is another one.")

    assert result["status"] == "success"
    assert result["sentences_count"] == 2
    assert result["words_count"] >= 8


def test_generate_text_honors_max_length() -> None:
    """Test that generate text honors max length.
    
    Args:
        None.
    
    Returns:
        None.
    """
    processor = AITextProcessor()

    result = processor.generate_text("machine learning", length=60)

    assert result["status"] == "success"
    assert len(result["generated_text"]) <= 60
    assert result["text_length"] <= 60


def test_history_tracks_successful_operations() -> None:
    """Test that history tracks successful operations.
    
    Args:
        None.
    
    Returns:
        None.
    """
    processor = AITextProcessor()

    processor.detect_language("Hello world")
    processor.generate_text("AI systems")

    history = processor.get_history()

    assert len(history) == 2
    assert all(item["status"] == "success" for item in history)
