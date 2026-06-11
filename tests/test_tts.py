import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from src.tts_engine import TTSEngine
from src.acoustic_cadence import AcousticCadenceProcessor

@pytest.fixture
def tts_config():
    return {
        "tts": {
            "voice": "en-US-ChristopherNeural",
        },
        "cadence": {
            "tail_phrase": "THE END.",
            "room_tone_duration_sec": 1.2,
            "room_tone_amplitude": 0.005,
        },
        "pipeline": {
            "temp_dir": "temp",
            "log_dir": "output/logs",
            "log_level": "INFO",
        }
    }

def test_preprocess_cadence():
    # Test em-dash replacement with ellipsis
    text = "First part — second part."
    processed = TTSEngine._preprocess_cadence(text)
    assert "..." in processed
    assert "—" not in processed

    # Test space normalisation after period
    text2 = "Sentence one.   Sentence two."
    processed2 = TTSEngine._preprocess_cadence(text2)
    assert "one. Sentence" in processed2

@patch("edge_tts.Communicate")
def test_synthesize_success(mock_communicate, tts_config):
    # Setup mock Communicate instance
    mock_instance = MagicMock()
    mock_instance.save = AsyncMock()
    mock_communicate.return_value = mock_instance
    
    # When save is called, write a dummy file to satisfy the file size checks
    def write_dummy_file(path_str):
        with open(path_str, "wb") as f:
            f.write(b"dummy audio data")
            
    mock_instance.save.side_effect = write_dummy_file
    
    engine = TTSEngine(tts_config)
    audio_path = engine.synthesize("Test script for Edge TTS.")
    
    assert audio_path.exists()
    assert audio_path.stat().st_size > 0
    assert audio_path.name.startswith("tts_")
    
    # Check that Communicate was called with the correct text and voice
    expected_processed_text = TTSEngine._preprocess_cadence("Test script for Edge TTS.")
    mock_communicate.assert_called_once_with(expected_processed_text, "en-US-ChristopherNeural")
    mock_instance.save.assert_called_once_with(str(audio_path))
    
    # Cleanup
    audio_path.unlink()

@patch("edge_tts.Communicate")
def test_synthesize_failure(mock_communicate, tts_config):
    mock_instance = MagicMock()
    mock_instance.save = AsyncMock(side_effect=Exception("Connection error"))
    mock_communicate.return_value = mock_instance
    
    engine = TTSEngine(tts_config)
    with pytest.raises(RuntimeError, match="Edge TTS synthesis failed"):
        engine.synthesize("Hello world")

def test_acoustic_cadence_vocal_decay_prep(tts_config):
    processor = AcousticCadenceProcessor(tts_config, MagicMock())
    script = "This is a story."
    prepared = processor.prepare_script_with_tail(script)
    assert prepared == "This is a story. THE END."
