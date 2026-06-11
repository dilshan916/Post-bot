import pytest
from unittest.mock import MagicMock, patch
from src.whisper_timestamps import TimestampExtractor, _build_sentence

@pytest.fixture
def whisper_config():
    return {
        "whisper": {
            "model_size": "medium",
            "language": "en",
            "min_vram_gb": 5.0,
            "vosk_model": "vosk-model-small-en-us-0.15",
            "vosk_model_path": "assets/vosk_model",
        },
        "pipeline": {
            "temp_dir": "temp",
            "log_dir": "output/logs",
            "log_level": "INFO",
        }
    }

def test_sentence_building():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.5, "confidence": 0.9},
        {"word": "world.", "start": 0.6, "end": 1.0, "confidence": 0.95}
    ]
    sentence = _build_sentence(words)
    assert sentence["text"] == "Hello world."
    assert sentence["start"] == 0.0
    assert sentence["end"] == 1.0
    assert len(sentence["words"]) == 2

def test_group_into_sentences():
    words = [
        {"word": "This", "start": 0.0, "end": 0.4, "confidence": 0.9},
        {"word": "is", "start": 0.5, "end": 0.8, "confidence": 0.9},
        {"word": "sentence", "start": 0.9, "end": 1.2, "confidence": 0.9},
        {"word": "one.", "start": 1.3, "end": 1.8, "confidence": 0.9},
        {"word": "And", "start": 1.9, "end": 2.2, "confidence": 0.9},
        {"word": "two", "start": 2.3, "end": 2.7, "confidence": 0.9}
    ]
    
    sentences = TimestampExtractor._group_into_sentences(words)
    assert len(sentences) == 2
    assert sentences[0]["text"] == "This is sentence one."
    assert sentences[0]["start"] == 0.0
    assert sentences[0]["end"] == 1.8
    
    assert sentences[1]["text"] == "And two"
    assert sentences[1]["start"] == 1.9
    assert sentences[1]["end"] == 2.7

@patch("torch.cuda.is_available")
@patch("torch.cuda.mem_get_info")
def test_vram_check(mock_mem, mock_avail):
    mock_avail.return_value = True
    # 6 GB free, 8 GB total
    mock_mem.return_value = (6 * (1024 ** 3), 8 * (1024 ** 3))
    
    extractor = TimestampExtractor({"whisper": {"min_vram_gb": 5.0}})
    vram = extractor._check_vram()
    assert vram == 6.0
