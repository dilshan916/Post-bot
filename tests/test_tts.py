import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from src.tts_engine import TTSEngine
from src.acoustic_cadence import AcousticCadenceProcessor
from main import RedditDailyBot

@pytest.fixture
def tts_config():
    return {
        "tts": {
            "engine": "edge-tts",
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
        },
        "video": {
            "output_dir": "output",
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

@patch("edge_tts.Communicate")
def test_synthesize_edge_tts_invalid_voice_fallback(mock_communicate, tts_config):
    mock_instance = MagicMock()
    mock_instance.save = AsyncMock()
    mock_communicate.return_value = mock_instance
    
    def write_dummy(path_str):
        with open(path_str, "wb") as f:
            f.write(b"dummy")
    mock_instance.save.side_effect = write_dummy

    engine = TTSEngine(tts_config)
    # Pass an invalid voice name (doesn't contain "Neural" or "-")
    audio_path = engine.synthesize(
        script="Hello invalid voice",
        voice="invalid_voice_id"
    )
    
    # Verify Communicate was called with the fallback voice (en-US-ChristopherNeural)
    mock_communicate.assert_called_once_with(
        TTSEngine._preprocess_cadence("Hello invalid voice"),
        "en-US-ChristopherNeural"
    )
    
    # Cleanup
    audio_path.unlink()


class TestTTSVoicingIntegration:
    @patch("src.tts_engine.TTSEngine.synthesize")
    @patch("src.acoustic_cadence.AcousticCadenceProcessor.prepare_script_with_tail")
    def test_monologue_edge_tts_male(self, mock_prepare, mock_synthesize, tts_config):
        mock_prepare.return_value = "Test monologue script. THE END."
        class StopExecution(Exception):
            pass
        mock_synthesize.side_effect = StopExecution("stopped")

        cfg = dict(tts_config)
        cfg["pipeline"]["pipeline_mode"] = "monologue"

        bot = RedditDailyBot(cfg, MagicMock())
        part = {"part_number": 1, "total_parts": 1, "script_text": "Test monologue script."}
        story = {"narrator_gender": "MALE", "title": "Test story", "subreddit": "stories"}

        with pytest.raises(Exception, match="stopped"):
            bot._render_part(part, story)

        mock_synthesize.assert_called_once_with(
            "Test monologue script. THE END.",
            voice="en-US-ChristopherNeural"
        )

    @patch("src.tts_engine.TTSEngine.synthesize")
    @patch("src.acoustic_cadence.AcousticCadenceProcessor.prepare_script_with_tail")
    def test_monologue_edge_tts_female(self, mock_prepare, mock_synthesize, tts_config):
        mock_prepare.return_value = "Test monologue script. THE END."
        class StopExecution(Exception):
            pass
        mock_synthesize.side_effect = StopExecution("stopped")

        cfg = dict(tts_config)
        cfg["pipeline"]["pipeline_mode"] = "monologue"

        bot = RedditDailyBot(cfg, MagicMock())
        part = {"part_number": 1, "total_parts": 1, "script_text": "Test monologue script."}
        story = {"narrator_gender": "FEMALE", "title": "Test story", "subreddit": "stories"}

        with pytest.raises(Exception, match="stopped"):
            bot._render_part(part, story)

        mock_synthesize.assert_called_once_with(
            "Test monologue script. THE END.",
            voice="en-US-JennyNeural"
        )

    @patch("src.tts_engine.TTSEngine.synthesize")
    @patch("src.acoustic_cadence.AcousticCadenceProcessor.prepare_script_with_tail")
    @patch("pydub.AudioSegment.from_file")
    def test_thread_voice_assignment(self, mock_from_file, mock_prepare, mock_synthesize, tts_config):
        mock_prepare.return_value = "Comment 2. THE END."
        
        # Mock AudioSegment.from_file to return dummy segment
        mock_segment = MagicMock()
        mock_segment.__len__.return_value = 1000
        mock_from_file.return_value = mock_segment
        
        dummy_paths = [Path("temp/tts1.mp3"), Path("temp/tts2.mp3"), Path("temp/tts3.mp3")]
        mock_synthesize.side_effect = dummy_paths
        
        cfg = dict(tts_config)
        cfg["pipeline"]["pipeline_mode"] = "thread"
        cfg["thread"] = {
            "default_voice": "en-US-ChristopherNeural",
        }

        bot = RedditDailyBot(cfg, MagicMock())
        
        import json
        script_text = json.dumps([
            {"speaker": "NARRATOR", "voice": "en-US-ChristopherNeural", "text": "Question?"},
            {"speaker": "USER_1", "voice": "en-US-GuyNeural", "text": "Comment 1."},
            {"speaker": "USER_2", "voice": "en-US-JennyNeural", "text": "Comment 2."}
        ])
        
        part = {"part_number": 1, "total_parts": 1, "script_text": script_text}
        story = {"title": "Test thread", "subreddit": "AskReddit"}

        with patch("pydub.AudioSegment.export", side_effect=RuntimeError("export_called")):
            with pytest.raises(RuntimeError, match="export_called"):
                bot._render_part(part, story)

        assert mock_synthesize.call_count == 3
        calls = mock_synthesize.call_args_list
        
        assert calls[0][0][0] == "Question?"
        assert calls[0][1]["voice"] == "en-US-ChristopherNeural"
        
        assert calls[1][0][0] == "Comment 1."
        assert calls[1][1]["voice"] == "en-US-GuyNeural"
        
        assert calls[2][0][0] == "Comment 2. THE END."
        assert calls[2][1]["voice"] == "en-US-JennyNeural"

    @patch("src.tts_engine.TTSEngine.synthesize")
    @patch("src.acoustic_cadence.AcousticCadenceProcessor.prepare_script_with_tail")
    @patch("pydub.AudioSegment.from_file")
    def test_riddle_voice_assignment(self, mock_from_file, mock_prepare, mock_synthesize, tts_config):
        mock_segment = MagicMock()
        mock_segment.__len__.return_value = 1000
        mock_from_file.return_value = mock_segment
        
        dummy_paths = [Path("temp/tts1.mp3"), Path("temp/tts2.mp3"), Path("temp/tts3.mp3"), Path("temp/tts4.mp3"), Path("temp/tts5.mp3")]
        mock_synthesize.side_effect = dummy_paths
        
        cfg = dict(tts_config)
        cfg["pipeline"]["pipeline_mode"] = "riddle"
        cfg["conversational"] = {
            "male_voice": "en-US-GuyNeural",
            "female_voice": "en-US-MichelleNeural",
        }

        bot = RedditDailyBot(cfg, MagicMock())
        
        import json
        script_text = json.dumps([
            {"speaker": "MALE", "text": "Hook."},
            {"speaker": "FEMALE", "text": "Question?"},
            {"speaker": "MALE", "text": "Clues."},
            {"speaker": "FEMALE", "text": "Guess."},
            {"speaker": "MALE", "text": "CTA."}
        ])
        
        part = {"part_number": 1, "total_parts": 1, "script_text": script_text}
        story = {"title": "Test riddle", "subreddit": "Riddles"}

        with patch("pydub.AudioSegment.export", side_effect=RuntimeError("export_called")):
            with pytest.raises(RuntimeError, match="export_called"):
                bot._render_part(part, story)

        assert mock_synthesize.call_count == 5
        calls = mock_synthesize.call_args_list
        
        # MALE uses en-US-GuyNeural
        assert calls[0][1]["voice"] == "en-US-GuyNeural"
        
        # FEMALE uses en-US-MichelleNeural
        assert calls[1][1]["voice"] == "en-US-MichelleNeural"
        
        # MALE uses en-US-GuyNeural
        assert calls[2][1]["voice"] == "en-US-GuyNeural"
        
        # FEMALE uses en-US-MichelleNeural
        assert calls[3][1]["voice"] == "en-US-MichelleNeural"
        
        # MALE uses en-US-GuyNeural
        assert calls[4][1]["voice"] == "en-US-GuyNeural"

    def test_monologue_kokoro_male_voice(self, tts_config):
        cfg = dict(tts_config)
        cfg["tts"]["engine"] = "kokoro"
        cfg["tts"]["voice"] = "am_adam"
        cfg["tts"]["male_voice"] = "am_adam"
        cfg["tts"]["female_voice"] = "af_heart"
        cfg["pipeline"]["pipeline_mode"] = "monologue"

        bot = RedditDailyBot(cfg, MagicMock())
        part = {"part_number": 1, "total_parts": 1, "script_text": "Test monologue script."}
        story = {"narrator_gender": "MALE", "title": "Test story", "subreddit": "stories"}

        with patch("src.tts_engine.TTSEngine.synthesize", side_effect=RuntimeError("stopped")) as mock_synthesize:
            with patch("src.acoustic_cadence.AcousticCadenceProcessor.prepare_script_with_tail", return_value="Test monologue script. THE END."):
                with pytest.raises(RuntimeError, match="stopped"):
                    bot._render_part(part, story)

                mock_synthesize.assert_called_once_with(
                    "Test monologue script. THE END.",
                    voice="am_adam"
                )

    @patch("src.tts_engine.TTSEngine.synthesize")
    @patch("src.acoustic_cadence.AcousticCadenceProcessor.prepare_script_with_tail")
    @patch("pydub.AudioSegment.from_file")
    def test_conversational_kokoro_distinct_voices(self, mock_from_file, mock_prepare, mock_synthesize, tts_config):
        mock_segment = MagicMock()
        mock_segment.__len__.return_value = 1000
        mock_from_file.return_value = mock_segment
        
        dummy_paths = [Path("temp/tts1.mp3"), Path("temp/tts2.mp3"), Path("temp/tts3.mp3")]
        mock_synthesize.side_effect = dummy_paths
        
        cfg = dict(tts_config)
        cfg["tts"]["engine"] = "kokoro"
        cfg["tts"]["male_voice"] = "am_adam"
        cfg["tts"]["female_voice"] = "af_bella"
        cfg["tts"]["old_female_voice"] = "af_nicole"
        cfg["pipeline"]["pipeline_mode"] = "conversational"

        bot = RedditDailyBot(cfg, MagicMock())
        
        import json
        script_text = json.dumps({
            "caption": "Test caption #test",
            "script": [
                {"speaker": "FEMALE", "text": "Line 1"},
                {"speaker": "MALE", "text": "Line 2"},
                {"speaker": "OLD_FEMALE", "text": "Line 3"}
            ]
        })
        
        part = {"part_number": 1, "total_parts": 1, "script_text": script_text}
        story = {"title": "Test story", "subreddit": "stories"}

        with patch("pydub.AudioSegment.export", side_effect=RuntimeError("export_called")):
            with pytest.raises(RuntimeError, match="export_called"):
                bot._render_part(part, story)

        assert mock_synthesize.call_count == 3
        calls = mock_synthesize.call_args_list
        
        # FEMALE uses af_bella
        assert calls[0][1]["voice"] == "af_bella"
        
        # MALE uses am_adam
        assert calls[1][1]["voice"] == "am_adam"
        
        # OLD_FEMALE uses af_nicole
        assert calls[2][1]["voice"] == "af_nicole"


def test_kokoro_fallback_assets_missing(tts_config):
    cfg = dict(tts_config)
    cfg["tts"]["engine"] = "kokoro"
    cfg["tts"]["kokoro"] = {
        "model_path": "non_existent/model.onnx",
        "voices_path": "non_existent/voices.json"
    }
    
    mock_log = MagicMock()
    engine = TTSEngine(cfg, logger=mock_log)
    
    assert engine._engine_type == "edge-tts"
    # Check that warning was logged
    warning_logged = any(
        "Kokoro assets missing" in str(call) or "kokoro-onnx package missing" in str(call)
        for call in mock_log.warning.call_args_list
    )
    assert warning_logged

