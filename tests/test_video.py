import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
from src.video_compositor import VideoCompositor
from src.video_processor import HashDestructionPipeline
from src.subtitle_renderer import SubtitleRenderer

@pytest.fixture
def video_config():
    return {
        "video": {
            "output_dir": "output",
            "resolution": {"width": 1080, "height": 1920},
            "fps": 30,
            "codec": "libx264",
            "audio_codec": "aac",
            "crf": 18,
            "preset": "medium",
            "pixel_format": "yuv420p",
            "gameplay_dir": "assets/gameplay",
            "filters": {
                "scale_width": 1200,
                "scale_height": 2130,
                "crop_width": 1080,
                "crop_height": 1920,
                "hue_shift": 0.5,
                "saturation_mult": 1.02,
                "brightness_shift": 0.01,
                "noise_strength": 2,
                "noise_flags": "t",
                "audio_tempo": 1.01,
            }
        },
        "splitter": {
            "watermark_font_size": 28,
            "watermark_color": "#FFFFFF",
            "watermark_opacity": 0.7,
            "watermark_position": "top-right",
        },
        "subtitles": {
            "font_family": "Arial",
            "font_size": 42,
            "font_bold": True,
            "passive_color": "#FFFFFF",
            "passive_opacity": 0.40,
            "passive_stroke_color": "#000000",
            "passive_stroke_width": 2,
            "active_color": "#FFE500",
            "active_opacity": 1.0,
            "active_stroke_color": "#000000",
            "active_stroke_width": 3,
            "vertical_position": 0.72,
            "max_chars_per_line": 35,
        },
        "pipeline": {
            "temp_dir": "temp",
            "log_dir": "output/logs",
            "log_level": "INFO",
        }
    }

def test_generate_output_filename(video_config):
    compositor = VideoCompositor(video_config, MagicMock())
    
    # Single-part
    filename = compositor._generate_output_filename(
        part_info=None,
        subreddit="AmItheAsshole",
        title="AITA for telling my friend to leave?",
    )
    assert "AmItheAsshole" in filename
    assert "AITA_for_telling" in filename
    assert filename.endswith(".mp4")
    assert "part" not in filename

    # Multi-part
    part_info = {"part_number": 1, "total_parts": 2}
    filename_part = compositor._generate_output_filename(
        part_info=part_info,
        subreddit="AmItheAsshole",
        title="AITA for telling my friend to leave?",
    )
    assert "AmItheAsshole" in filename_part
    assert "part1" in filename_part

def test_hash_destruction_filters(video_config):
    pipeline = HashDestructionPipeline(video_config, MagicMock())
    vf = pipeline._build_filter_chain()
    af = pipeline._build_audio_filter()
    
    assert "mpdecimate" in vf
    assert "scale=1200:2130" in vf
    assert "crop=1080:1920" in vf
    assert "hue=h=0.5:s=1.02:b=0.01" in vf
    assert "noise=alls=2:allf=t" in vf
    
    assert af == "atempo=1.01"

def test_subtitle_text_wrap(video_config):
    words = [
        {"word": "This"}, {"word": "is"}, {"word": "a"}, {"word": "fairly"},
        {"word": "long"}, {"word": "sentence"}, {"word": "that"}, {"word": "should"},
        {"word": "wrap"}, {"word": "onto"}, {"word": "multiple"}, {"word": "lines"},
        {"word": "because"}, {"word": "of"}, {"word": "character"}, {"word": "limitations"}
    ]
    wrapped = SubtitleRenderer._wrap_sentence_text(words, max_chars=20)
    lines = wrapped.split("\n")
    assert len(lines) > 1
    for line in lines:
        assert len(line) <= 20

def test_create_speaker_stickers(video_config):
    from unittest.mock import MagicMock, patch
    from PIL import Image
    
    # Add stickers config settings to video_config
    video_config["conversational"] = {
        "stickers_enabled": True,
        "male_sticker_path": "assets/stickers/male_neutral.png",
        "female_sticker_path": "assets/stickers/female_neutral.png",
        "sticker_size": 250,
        "sticker_y_position": 0.35,
    }
    
    compositor = VideoCompositor(video_config, MagicMock())
    
    # 1. Test fallback when paths don't exist
    with patch("src.video_compositor.resolve_path") as mock_resolve:
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        mock_resolve.return_value = mock_path
        
        clips = compositor._create_speaker_stickers(
            dialogue_timings=[{"start": 0.0, "end": 2.0, "speaker": "MALE", "text": "Hello"}],
            frame_size=(1080, 1920)
        )
        assert clips == []
        
    # 2. Test clip generation when paths exist
    dummy_img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    
    with patch("src.video_compositor.resolve_path") as mock_resolve, \
         patch("PIL.Image.open") as mock_open:
        
        mock_path = MagicMock()
        mock_path.exists.return_value = True
        mock_resolve.return_value = mock_path
        mock_open.return_value = dummy_img
        
        dialogue_timings = [
            {"start": 0.0, "end": 2.0, "speaker": "MALE", "emotion": "angry", "text": "I am not angry!"},
            {"start": 2.2, "end": 5.0, "speaker": "FEMALE", "emotion": "surprised", "text": "What do you mean?"},
        ]
        
        clips = compositor._create_speaker_stickers(
            dialogue_timings=dialogue_timings,
            frame_size=(1080, 1920)
        )
        
        assert len(clips) == 2
        assert clips[0].start == 0.0
        assert clips[0].duration == 2.0
        assert clips[1].start == 2.2
        assert clips[1].duration == 2.8

def test_detect_emotion(video_config):
    from unittest.mock import MagicMock
    compositor = VideoCompositor(video_config, MagicMock())
    
    assert compositor._detect_emotion("I am so angry and furious!") == "angry"
    assert compositor._detect_emotion("I am so sorry, I feel so sad.") == "crying"
    assert compositor._detect_emotion("What? This is insane!") == "surprised"
    assert compositor._detect_emotion("I am so worried and stressed.") == "stressed"
    assert compositor._detect_emotion("I love you sweetheart!") == "lovestruck"
    assert compositor._detect_emotion("This is happy, great fun!") == "happy"
    assert compositor._detect_emotion("Who are you?") == "thinking"
    assert compositor._detect_emotion("Hello world.") == "talking"


