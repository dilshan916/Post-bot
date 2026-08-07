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
    
    # Use config dict
    extractor = TimestampExtractor({"whisper": {"min_vram_gb": 5.0}})
    vram = extractor._check_vram()
    assert vram == 6.0

def test_group_into_sentences_speaker_change():
    words = [
        {"word": "Hello", "start": 0.0, "end": 0.4, "confidence": 0.9, "speaker": "MALE"},
        {"word": "there", "start": 0.5, "end": 0.8, "confidence": 0.9, "speaker": "MALE"},
        {"word": "Hi", "start": 0.9, "end": 1.2, "confidence": 0.9, "speaker": "FEMALE"},
        {"word": "friend", "start": 1.3, "end": 1.8, "confidence": 0.9, "speaker": "FEMALE"}
    ]
    
    sentences = TimestampExtractor._group_into_sentences(words)
    assert len(sentences) == 2
    assert sentences[0]["text"] == "Hello there"
    assert sentences[0]["words"][0]["speaker"] == "MALE"
    assert sentences[1]["text"] == "Hi friend"
    assert sentences[1]["words"][0]["speaker"] == "FEMALE"

def test_whisper_extractor_init_device_config():
    config = {
        "whisper": {
            "model_size": "medium",
            "language": "en",
            "min_vram_gb": 5.0,
            "device": "cpu",
            "cpu_model_size": "tiny",
            "vosk_model": "vosk-model-small-en-us-0.15",
            "vosk_model_path": "assets/vosk_model",
        }
    }
    extractor = TimestampExtractor(config)
    assert extractor.device == "cpu"
    assert extractor.cpu_model_size == "tiny"

@patch("src.whisper_timestamps.TimestampExtractor._check_vram")
@patch("src.whisper_timestamps.TimestampExtractor._extract_with_whisper")
@patch("src.whisper_timestamps.TimestampExtractor._group_into_sentences")
@patch("src.whisper_timestamps.Path.exists")
def test_extract_whisper_device_selection(mock_exists, mock_group, mock_extract_whisper, mock_check_vram):
    from pathlib import Path
    mock_exists.return_value = True
    mock_check_vram.return_value = 2.0  # Low VRAM
    mock_extract_whisper.return_value = [{"word": "test", "start": 0.0, "end": 1.0, "confidence": 0.9}]
    mock_group.return_value = []

    # Case 1: auto device with low VRAM -> should select cpu device and cpu_model_size
    config = {
        "whisper": {
            "model_size": "medium",
            "language": "en",
            "min_vram_gb": 5.0,
            "device": "auto",
            "cpu_model_size": "base",
        }
    }
    extractor = TimestampExtractor(config)
    extractor.extract("dummy_audio.mp3")
    mock_extract_whisper.assert_called_with(
        Path("dummy_audio.mp3"),
        device="cpu",
        model_size="base"
    )

    # Case 2: auto device with high VRAM -> should select cuda device and regular model_size
    mock_check_vram.return_value = 8.0  # High VRAM
    extractor.extract("dummy_audio.mp3")
    mock_extract_whisper.assert_called_with(
        Path("dummy_audio.mp3"),
        device="cuda",
        model_size="medium"
    )
