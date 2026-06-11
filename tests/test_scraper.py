import pytest
from unittest.mock import MagicMock, patch
from src.script_rewriter import ScriptRewriter
from src.smart_splitter import SmartSplitter
from src.reddit_scraper import RedditScraper

@pytest.fixture
def dummy_config():
    return {
        "reddit": {
            "client_id": "dummy",
            "client_secret": "dummy",
            "user_agent": "dummy",
            "subreddits": ["AmItheAsshole"],
            "sort": "hot",
            "time_filter": "day",
            "post_limit": 5,
            "min_upvotes": 100,
            "min_word_count": 10,
            "max_word_count": 500,
        },
        "llm": {
            "provider": "",
        },
        "splitter": {
            "max_duration_sec": 60,
            "words_per_minute": 150,
            "min_part_duration_sec": 15,
            "watermark_text": "Part {n} of {total}",
            "watermark_position": "top-right",
            "watermark_font_size": 28,
            "watermark_color": "#FFFFFF",
            "watermark_opacity": 0.7,
        },
        "pipeline": {
            "temp_dir": "temp",
            "log_dir": "output/logs",
            "log_level": "INFO",
            "scraped_history": "data/scraped_history.json",
        },
    }

def test_script_rewriter_regex_cleanup(dummy_config):
    from src.script_rewriter import _INTRO_VARIATIONS
    rewriter = ScriptRewriter(dummy_config)
    post = {
        "title": "AITA for telling my friend to leave?",
        "body": "So WIBTA? I (25F) had a friend over.\nEDIT: they left.\nTL;DR: friend left because throwaway.\nSorry for the formatting, I am on mobile.",
    }
    cleaned = rewriter.rewrite(post)
    
    # Check that intro is prepended
    assert any(intro in cleaned for intro in _INTRO_VARIATIONS)
    
    # Check that Reddit artifacts like EDIT, TL;DR, formatting apology, etc. are stripped
    assert "EDIT:" not in cleaned
    assert "TL;DR:" not in cleaned
    assert "mobile" not in cleaned
    
    # Check that abbreviations are expanded for TTS
    assert "Would I be the jerk" in cleaned
    assert "WIBTA" not in cleaned

def test_smart_splitter_no_split(dummy_config):
    splitter = SmartSplitter(dummy_config, MagicMock())
    short_script = "This is a short script. It has very few words. It should not be split."
    parts = splitter.split(short_script)
    assert len(parts) == 1
    assert parts[0]["part_number"] == 1
    assert parts[0]["total_parts"] == 1
    assert parts[0]["script_text"] == short_script
    assert parts[0]["watermark_config"] is None

def test_smart_splitter_with_split(dummy_config):
    # Set max duration to 10 seconds (~25 words at 150 WPM)
    dummy_config["splitter"]["max_duration_sec"] = 10
    dummy_config["splitter"]["min_part_duration_sec"] = 2
    splitter = SmartSplitter(dummy_config, MagicMock())
    
    long_script = (
        "This is the first sentence of our story. It is quite simple. "
        "Here is the second sentence which has more words to increase length. "
        "And now we write the third sentence which will definitely cross the duration threshold. "
        "Finally, this is the fourth sentence of the script."
    )
    
    parts = splitter.split(long_script)
    assert len(parts) >= 2
    assert parts[0]["part_number"] == 1
    assert parts[0]["total_parts"] == len(parts)
    assert parts[0]["watermark_config"] is not None
    assert parts[0]["watermark_config"]["text"] == f"Part 1 of {len(parts)}"

def test_reddit_scraper_init(dummy_config):
    scraper = RedditScraper(dummy_config)
    assert scraper._history_path.name == "scraped_history.json"

@patch("requests.get")
def test_reddit_scraper_scrape_stories(mock_get, dummy_config):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
        <entry>
            <id>t3_post1</id>
            <title>Test Post Title</title>
            <author><name>/u/test_user</name></author>
            <link rel="alternate" href="https://www.reddit.com/r/AmItheAsshole/comments/post1/"/>
            <updated>2026-05-18T08:45:54+00:00</updated>
            <content type="html">&lt;div class="md"&gt;&lt;p&gt;This is a dummy self text post that is long enough to pass the filters and has enough characters. It needs to have at least 10 words as configured in dummy_config. So this sentence will pass.&lt;/p&gt;&lt;/div&gt;</content>
        </entry>
    </feed>
    """
    mock_get.return_value = mock_response

    scraper = RedditScraper(dummy_config)
    stories = scraper.scrape_stories()
    
    assert len(stories) == 1
    assert stories[0]["id"] == "post1"
    assert stories[0]["title"] == "Test Post Title"
    assert stories[0]["subreddit"] == "AmItheAsshole"
    assert stories[0]["score"] == 200

@patch("requests.post")
def test_script_rewriter_groq_success(mock_post, dummy_config):
    # Set config to use groq
    dummy_config["groq"] = {
        "api_key": "test_groq_api_key_1234",
        "model": "llama-3.3-70b-versatile"
    }
    
    # Setup mock return value for model generation
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": "This is a rewritten script by Groq."
            }
        }]
    }
    mock_post.return_value = mock_response
    
    rewriter = ScriptRewriter(dummy_config)
    post = {
        "title": "AITA for telling my friend to leave?",
        "body": "WIBTA? Had a friend over.",
    }
    cleaned = rewriter.rewrite(post)
    
    # Verify requests.post was called once
    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert kwargs["headers"]["Authorization"] == "Bearer test_groq_api_key_1234"
    assert kwargs["json"]["model"] == "llama-3.3-70b-versatile"
    
    # Verify text is from Groq output
    assert "rewritten script by Groq" in cleaned

@patch("time.sleep")
@patch("requests.post")
def test_script_rewriter_groq_fallback_on_error(mock_post, mock_sleep, dummy_config):
    dummy_config["groq"] = {
        "api_key": "test_groq_api_key_1234",
        "model": "llama-3.3-70b-versatile"
    }
    
    mock_post.side_effect = Exception("API connection error")
    
    rewriter = ScriptRewriter(dummy_config)
    post = {
        "title": "AITA for telling my friend to leave?",
        "body": "So WIBTA?\nEDIT: they left.",
    }
    cleaned = rewriter.rewrite(post)
    
    # Verify that despite exception, it returned regex cleaned text (e.g. WIBTA expanded, EDIT stripped)
    assert "Would I be the jerk" in cleaned
    assert "EDIT:" not in cleaned

@patch("requests.post")
def test_script_rewriter_groq_fallback_on_placeholder_key(mock_post, dummy_config):
    dummy_config["groq"] = {
        "api_key": "YOUR_GROQ_API_KEY_HERE", # placeholder
        "model": "llama-3.3-70b-versatile"
    }
    
    rewriter = ScriptRewriter(dummy_config)
    post = {
        "title": "AITA for telling my friend to leave?",
        "body": "So WIBTA?",
    }
    cleaned = rewriter.rewrite(post)
    
    # Verify API client was not created
    mock_post.assert_not_called()
    assert "Would I be the jerk" in cleaned


# ── Screenshot Manager Tests ──────────────────────────────────────

def test_screenshot_html_generation(dummy_config):
    """Verify the HTML card contains the subreddit, author, and title."""
    from src.screenshot_manager import ScreenshotManager

    dummy_config["screenshot"] = {"theme": "dark"}
    logger = MagicMock()
    manager = ScreenshotManager(dummy_config, logger)

    story = {
        "subreddit": "relationship_advice",
        "title": "My partner won't stop eating my leftovers",
        "author": "throwaway1234",
        "score": 4200,
    }

    html = manager._build_html(
        subreddit=story["subreddit"],
        title=story["title"],
        author=story["author"],
        score=story["score"],
        theme=manager._DARK_THEME,
    )

    # Verify key elements are present in the HTML
    assert "r/relationship_advice" in html
    assert "u/throwaway1234" in html
    assert "My partner won&#x27;t stop eating my leftovers" in html
    assert "4.2k" in html  # score formatting
    assert "#1A1A1B" in html  # dark theme card bg
    assert "IBM Plex Sans" in html  # font family


def test_screenshot_hook_clip_creation(dummy_config, tmp_path):
    """Verify hook clip has correct duration and position."""
    from src.screenshot_manager import ScreenshotManager
    from PIL import Image

    dummy_config["screenshot"] = {
        "display_duration_sec": 3.5,
        "fade_duration_sec": 0.8,
        "card_width_pct": 0.88,
    }
    logger = MagicMock()
    manager = ScreenshotManager(dummy_config, logger)

    # Create a small dummy PNG for testing
    dummy_img = Image.new("RGBA", (920, 200), (26, 26, 27, 255))
    png_path = tmp_path / "hook_screenshot.png"
    dummy_img.save(str(png_path))

    clip = manager.create_hook_clip(
        screenshot_path=png_path,
        frame_size=(1080, 1920),
    )

    assert clip.duration == 3.5
    assert clip.start == 0.0
    # Clip should be positioned within frame bounds
    pos = clip.pos(0)
    assert 0 <= pos[0] < 1080  # x within frame
    assert 0 <= pos[1] < 1920  # y within frame

    clip.close()


def test_reddit_scraper_keyword_rejection(dummy_config):
    """Verify that the scraper rejects posts containing meta keywords."""
    scraper = RedditScraper(dummy_config)
    
    # Base submission data that passes other filters
    base_sub = {
        "id": "test_kw",
        "title": "A normal story about my life",
        "selftext": "This is a detailed post about what happened to me yesterday at the supermarket. It contains enough words to pass.",
        "subreddit": "AmItheAsshole",
        "score": 1000,
        "is_self": True,
    }
    
    # 1. Normal post should pass
    assert scraper._passes_filters(base_sub) is True
    
    # 2. Post with "survey" in title should be rejected
    survey_title_sub = base_sub.copy()
    survey_title_sub["title"] = "Please fill out this survey"
    assert scraper._passes_filters(survey_title_sub) is False
    
    # 3. Post with "cornell" in body should be rejected
    cornell_body_sub = base_sub.copy()
    cornell_body_sub["selftext"] = "We are conducting research at Cornell university."
    assert scraper._passes_filters(cornell_body_sub) is False


@patch("time.sleep")
@patch("requests.post")
def test_script_rewriter_groq_retry_exponential_backoff(mock_post, mock_sleep, dummy_config):
    """Verify that the Groq LLM rewriter retries with exponential backoff on transient errors and succeeds."""
    dummy_config["groq"] = {
        "api_key": "test_api_key_123",
        "model": "llama-3.3-70b-versatile"
    }
    
    # Setup mock to fail on the first attempt and succeed on the second
    mock_response_fail = MagicMock()
    mock_response_fail.status_code = 503
    mock_response_fail.text = "Service Unavailable"
    
    mock_response_ok = MagicMock()
    mock_response_ok.status_code = 200
    mock_response_ok.json.return_value = {
        "choices": [{
            "message": {
                "content": "Success after retry."
            }
        }]
    }
    
    mock_post.side_effect = [
        Exception("503 Service Unavailable"),
        mock_response_ok
    ]
    
    rewriter = ScriptRewriter(dummy_config)
    post = {
        "title": "AITA for telling my friend to leave?",
        "body": "Had a friend over.",
    }
    cleaned = rewriter.rewrite(post)
    
    # Assert requests.post was called twice
    assert mock_post.call_count == 2
    
    # Assert sleep was called with 15 seconds (the first backoff duration)
    mock_sleep.assert_called_once_with(15)
    
    # Assert output contains the mock success response
    assert "Success after retry" in cleaned


@patch("time.sleep")
@patch("requests.post")
def test_script_rewriter_groq_retry_all_fail_fallback(mock_post, mock_sleep, dummy_config):
    """Verify that if all Groq API attempts fail, it falls back to regex."""
    dummy_config["groq"] = {
        "api_key": "test_api_key_123",
        "model": "llama-3.3-70b-versatile"
    }
    
    mock_post.side_effect = Exception("503 Service Unavailable")
    
    rewriter = ScriptRewriter(dummy_config)
    post = {
        "title": "AITA for telling my friend to leave?",
        "body": "So WIBTA? Had a friend over.\nEDIT: they left.",
    }
    cleaned = rewriter.rewrite(post)
    
    # Assert requests.post was called 4 times (1 initial + 3 retries)
    assert mock_post.call_count == 4
    
    # Assert sleep was called 3 times with backoffs [15, 30, 60]
    assert mock_sleep.call_count == 3
    mock_sleep.assert_any_call(15)
    mock_sleep.assert_any_call(30)
    mock_sleep.assert_any_call(60)
    
    # Assert fallback to regex worked (abbreviation expanded, EDIT stripped)
    assert "Would I be the jerk" in cleaned
    assert "EDIT:" not in cleaned


@patch("google.genai.Client")
def test_reddit_scraper_intelligent_llm_filter_success(mock_genai_client, dummy_config):
    """Verify that the Gemini selector correctly chooses a winner from candidate batch."""
    dummy_config["llm"]["provider"] = "gemini"
    dummy_config["llm"]["api_key"] = "test_api_key_123"
    dummy_config["llm"]["model"] = "gemini-2.5-flash"

    mock_client_instance = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"id": "post_winner", "gender": "FEMALE", "virality_reason": "High drama and conflict with wedding themes."}'
    mock_client_instance.models.generate_content.return_value = mock_response
    mock_genai_client.return_value = mock_client_instance

    scraper = RedditScraper(dummy_config)
    candidates = [
        {"id": "post_loser", "title": "Boring post", "body": "Short body", "subreddit": "AmItheAsshole", "score": 100},
        {"id": "post_winner", "title": "Dramatic post", "body": "Long dramatic body", "subreddit": "AmItheAsshole", "score": 200},
    ]
    
    result = scraper._intelligent_llm_filter(candidates)
    
    assert result is not None
    assert result["id"] == "post_winner"
    assert result["gender"] == "FEMALE"
    assert result["virality_reason"] == "High drama and conflict with wedding themes."
