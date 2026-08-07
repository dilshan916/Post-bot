"""
RedditDaily-Bot — Screenshot Manager
=======================================
Generates hyper-realistic Reddit post screenshots by rendering a
pixel-perfect local HTML card (dark-mode) via Playwright and capturing
an element screenshot.

The resulting PNG is used as a "hook" overlay during the first few
seconds of each Reel, matching the viral TikTok/Reels format where a
Reddit post card flashes on screen to establish the story context.

This avoids navigating to reddit.com (which blocks headless browsers)
by rendering all UI locally with accurate typography and styling.
"""

from __future__ import annotations

import hashlib
import random
from html import escape as html_escape
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from moviepy.editor import ImageClip, vfx
from PIL import Image

from src.utils import BotLogger, resolve_path


class ScreenshotManager:
    """Renders Reddit-style post cards and captures screenshots.

    Attributes:
        config: Full application configuration dictionary.
        logger: BotLogger instance for structured logging.
        screenshot_cfg: ``screenshot`` sub-config (theme, durations).
        temp_dir: Resolved path for temporary file output.
    """

    # Reddit dark-mode colour tokens
    _DARK_THEME = {
        "page_bg": "#030303",
        "card_bg": "#1A1A1B",
        "card_border": "#343536",
        "title_color": "#D7DADC",
        "meta_color": "#818384",
        "icon_color": "#D7DADC",
        "upvote_color": "#FF4500",
        "score_color": "#D7DADC",
        "divider_color": "#343536",
        "subreddit_color": "#D7DADC",
        "author_color": "#818384",
    }

    # Reddit light-mode colour tokens
    _LIGHT_THEME = {
        "page_bg": "#DAE0E6",
        "card_bg": "#FFFFFF",
        "card_border": "#CCCCCC",
        "title_color": "#222222",
        "meta_color": "#787C7E",
        "icon_color": "#878A8C",
        "upvote_color": "#FF4500",
        "score_color": "#1A1A1B",
        "divider_color": "#EDEFF1",
        "subreddit_color": "#1C1C1C",
        "author_color": "#787C7E",
    }

    # Subreddit avatar colours (deterministic based on name hash)
    _AVATAR_PALETTE = [
        "#FF4500", "#0079D3", "#46D160", "#FF585B",
        "#FFB000", "#7193FF", "#00A6A5", "#FF66AC",
        "#9B59B6", "#46A508", "#EA0027", "#25B79F",
    ]

    def __init__(self, config: Dict[str, Any], logger: BotLogger) -> None:
        """Initialise the screenshot manager.

        Args:
            config: Parsed application config (expects ``screenshot`` and
                ``pipeline`` sections).
            logger: Shared BotLogger instance.
        """
        self.config = config
        self.logger = logger
        self.screenshot_cfg: Dict[str, Any] = config.get("screenshot", {})
        self.temp_dir: Path = resolve_path(
            config["pipeline"]["temp_dir"], create=True
        )

    # ---- public API --------------------------------------------------------

    def capture(self, story: Dict[str, Any]) -> Path:
        """Render a Reddit post card and capture a screenshot.

        Builds a local HTML string mimicking the Reddit dark/light-mode
        post card, renders it in a headless Chromium browser via
        Playwright, and saves an element screenshot as a PNG.

        Args:
            story: Post dictionary with keys ``subreddit``, ``title``,
                ``author``, ``score``, ``url``.

        Returns:
            Path to the saved ``.png`` screenshot file.

        Raises:
            RuntimeError: If Playwright fails to capture the screenshot.
        """
        theme_name = self.screenshot_cfg.get("theme", "dark")
        theme = (
            self._DARK_THEME if theme_name == "dark" else self._LIGHT_THEME
        )

        subreddit = story.get("subreddit", "reddit")
        title = story.get("title", "Untitled")
        author = story.get("author", "[deleted]")
        score = story.get("score", 0)

        html_content = self._build_html(
            subreddit=subreddit,
            title=title,
            author=author,
            score=score,
            theme=theme,
        )

        output_path = self.temp_dir / "hook_screenshot.png"

        self.logger.info(
            "Capturing Reddit screenshot hook (theme=%s, subreddit=r/%s)",
            theme_name,
            subreddit,
        )

        try:
            self._render_and_capture(html_content, output_path)
        except Exception as exc:
            self.logger.error("Screenshot capture failed: %s", exc)
            raise RuntimeError(
                f"Failed to capture Reddit screenshot: {exc}"
            ) from exc

        file_size_kb = output_path.stat().st_size / 1024
        self.logger.info(
            "Screenshot saved: %s (%.1f KB)",
            output_path.name,
            file_size_kb,
        )
        return output_path

    def capture_post_card(self, story: Dict[str, Any], index: int) -> Path:
        """Render a Reddit post card and capture a screenshot with a unique index.

        Args:
            story: Post dictionary with keys ``subreddit``, ``title``,
                ``author``, ``score``, ``url``.
            index: The index of the post card (0, 1, 2).

        Returns:
            Path to the saved PNG screenshot file.
        """
        theme_name = self.screenshot_cfg.get("theme", "dark")
        theme = (
            self._DARK_THEME if theme_name == "dark" else self._LIGHT_THEME
        )

        subreddit = story.get("subreddit", "reddit")
        title = story.get("title", "Untitled")
        author = story.get("author", "[deleted]")
        score = story.get("score", 0)

        html_content = self._build_html(
            subreddit=subreddit,
            title=title,
            author=author,
            score=score,
            theme=theme,
        )

        output_path = self.temp_dir / f"shower_{index}.png"

        self.logger.info(
            "Capturing Reddit post card screenshot (theme=%s, subreddit=r/%s, index=%d)",
            theme_name,
            subreddit,
            index,
        )

        try:
            self._render_and_capture(html_content, output_path)
        except Exception as exc:
            self.logger.error("Post card screenshot capture failed: %s", exc)
            raise RuntimeError(
                f"Failed to capture Reddit post card screenshot: {exc}"
            ) from exc

        file_size_kb = output_path.stat().st_size / 1024
        self.logger.info(
            "Post card screenshot saved: %s (%.1f KB)",
            output_path.name,
            file_size_kb,
        )
        return output_path

    def capture_comment(self, comment: Dict[str, Any], index: int) -> Path:
        """Render a Reddit comment card and capture a screenshot.

        Args:
            comment: Dictionary with comment data (keys: 'author', 'body' or 'text', 'score').
            index: The index of the comment to write.

        Returns:
            Path to the saved PNG screenshot file.
        """
        theme_name = self.screenshot_cfg.get("theme", "dark")
        theme = (
            self._DARK_THEME if theme_name == "dark" else self._LIGHT_THEME
        )

        author = comment.get("author", "reddit_user")
        text = comment.get("text", comment.get("body", ""))
        score = comment.get("score", random.randint(100, 2500))

        html_content = self._build_comment_html(
            author=author,
            text=text,
            score=score,
            theme=theme,
        )

        output_path = self.temp_dir / f"comment_{index}.png"

        self.logger.info(
            "Capturing Reddit comment screenshot (theme=%s, author=u/%s, index=%d)",
            theme_name,
            author,
            index,
        )

        try:
            self._render_and_capture(html_content, output_path)
        except Exception as exc:
            self.logger.error("Comment screenshot capture failed: %s", exc)
            raise RuntimeError(
                f"Failed to capture Reddit comment screenshot: {exc}"
            ) from exc

        file_size_kb = output_path.stat().st_size / 1024
        self.logger.info(
            "Comment screenshot saved: %s (%.1f KB)",
            output_path.name,
            file_size_kb,
        )
        return output_path

    def create_hook_clip(
        self,
        screenshot_path: Path,
        frame_size: Tuple[int, int],
        duration: Optional[float] = None,
        fade_duration: Optional[float] = None,
        ken_burns: Optional[bool] = None,
    ) -> ImageClip:
        """Create a MoviePy ImageClip from the screenshot for compositing.

        The screenshot is resized to fit a configurable percentage of
        the frame width, centred, and given a smooth fade-out effect.

        Args:
            screenshot_path: Path to the ``.png`` screenshot.
            frame_size: ``(width, height)`` of the output video frame.
            duration: How long the card is visible. Falls back to
                ``screenshot.display_duration_sec`` (default 3.5s).
            fade_duration: Fade-out duration. Falls back to
                ``screenshot.fade_duration_sec`` (default 0.8s).
            ken_burns: Optional flag to enable/disable Ken Burns zoom.
                If None, falls back to config.

        Returns:
            A positioned, timed ``ImageClip`` with fade-out, ready for
            compositing in the layer stack.
        """
        display_sec: float = duration or self.screenshot_cfg.get(
            "display_duration_sec", 3.5
        )
        fade_sec: float = fade_duration if fade_duration is not None else self.screenshot_cfg.get(
            "fade_duration_sec", 0.8
        )
        card_width_pct: float = self.screenshot_cfg.get(
            "card_width_pct", 0.88
        )

        frame_w, frame_h = frame_size

        # Load screenshot with PIL to get dimensions and handle alpha
        img = Image.open(str(screenshot_path)).convert("RGBA")
        img_w, img_h = img.size

        # Calculate target width based on frame percentage, scaled by 1.25x
        target_w = int(frame_w * card_width_pct * 1.25)
        scale_factor = target_w / img_w
        target_h = int(img_h * scale_factor)

        # Resize with high-quality resampling
        img_resized = img.resize(
            (target_w, target_h), Image.Resampling.LANCZOS
        )

        # Separate RGB and alpha channels for MoviePy
        img_np = np.array(img_resized)
        rgb_np = img_np[:, :, :3]
        alpha_np = img_np[:, :, 3] / 255.0

        # Build ImageClip with mask
        clip = ImageClip(rgb_np)
        mask_clip = ImageClip(alpha_np, ismask=True)
        clip = clip.set_mask(mask_clip)

        # Centre the card in the frame (slightly above centre for
        # visual balance — 38% from top)
        if ken_burns is None:
            ken_burns_enabled = self.screenshot_cfg.get("ken_burns", True)
        else:
            ken_burns_enabled = ken_burns
        
        vertical_position = self.screenshot_cfg.get("vertical_position", 0.25)

        if ken_burns_enabled:
            # Linear zoom from 1.0 to 1.05 over the display duration
            def ken_burns_scale(t):
                return 1.0 + 0.05 * (t / display_sec)
                
            clip = clip.fx(vfx.resize, ken_burns_scale)
            
            # Position centered dynamically as the width/height changes over time
            def center_pos(t):
                scale = ken_burns_scale(t)
                curr_w = target_w * scale
                curr_h = target_h * scale
                x = (frame_w - curr_w) / 2
                y = int(frame_h * vertical_position) - (curr_h / 2)
                return (x, y)
                
            pos_x, pos_y = 0, 0 # placeholder for logging
            clip = (
                clip
                .set_position(center_pos)
                .set_start(0.0)
                .set_duration(display_sec)
            )
            if fade_sec > 0.0:
                clip = clip.fx(vfx.fadeout, fade_sec)
        else:
            pos_x = (frame_w - target_w) // 2
            pos_y = int(frame_h * vertical_position) - (target_h // 2)
            clip = (
                clip
                .set_position((pos_x, pos_y))
                .set_start(0.0)
                .set_duration(display_sec)
            )
            if fade_sec > 0.0:
                clip = clip.fx(vfx.fadeout, fade_sec)

        self.logger.info(
            "Hook clip created: %dx%d (Ken Burns=%s), "
            "duration=%.1fs, fade=%.1fs",
            target_w, target_h, ken_burns_enabled, display_sec, fade_sec,
        )
        return clip

    def create_comment_clip(
        self,
        screenshot_path: Path,
        frame_size: Tuple[int, int],
        start: float,
        duration: float,
        fade_duration: float = 0.3,
    ) -> ImageClip:
        """Create a MoviePy ImageClip from a comment screenshot for timing/overlay.

        Args:
            screenshot_path: Path to the comment PNG.
            frame_size: (width, height) of the video frame.
            start: Start time in seconds.
            duration: Duration in seconds.
            fade_duration: Fade in/out duration in seconds.

        Returns:
            ImageClip timed, positioned, and animated.
        """
        card_width_pct: float = self.screenshot_cfg.get(
            "card_width_pct", 0.88
        )
        frame_w, frame_h = frame_size

        # Load screenshot with PIL to get dimensions and handle alpha
        img = Image.open(str(screenshot_path)).convert("RGBA")
        img_w, img_h = img.size

        # Calculate target width based on frame percentage, scaled by 1.25x
        target_w = int(frame_w * card_width_pct * 1.25)
        scale_factor = target_w / img_w
        target_h = int(img_h * scale_factor)

        # Resize with high-quality resampling
        img_resized = img.resize(
            (target_w, target_h), Image.Resampling.LANCZOS
        )

        # Separate RGB and alpha channels for MoviePy
        img_np = np.array(img_resized)
        rgb_np = img_np[:, :, :3]
        alpha_np = img_np[:, :, 3] / 255.0

        # Build ImageClip with mask
        clip = ImageClip(rgb_np)
        mask_clip = ImageClip(alpha_np, ismask=True)
        clip = clip.set_mask(mask_clip)

        # Center the comment card slightly above the subtitles (e.g. 30% from top)
        vertical_position = self.screenshot_cfg.get("comment_vertical_position", 0.30)
        pos_x = (frame_w - target_w) // 2
        pos_y = int(frame_h * vertical_position) - (target_h // 2)

        clip = (
            clip
            .set_position((pos_x, pos_y))
            .set_start(start)
            .set_duration(duration)
            .fx(vfx.fadein, fade_duration)
            .fx(vfx.fadeout, fade_duration)
        )

        self.logger.info(
            "Comment clip created: %dx%d, start=%.1fs, duration=%.1fs",
            target_w, target_h, start, duration
        )
        return clip

    # ---- HTML builder ------------------------------------------------------

    def _get_avatar_color(self, subreddit: str) -> str:
        """Return a deterministic colour for the subreddit avatar.

        Uses an MD5 hash of the subreddit name to pick a colour from
        the palette, ensuring the same subreddit always gets the same
        colour.
        """
        digest = hashlib.md5(subreddit.lower().encode()).hexdigest()
        index = int(digest[:8], 16) % len(self._AVATAR_PALETTE)
        return self._AVATAR_PALETTE[index]

    def _format_score(self, score: int) -> str:
        """Format a score integer for display (e.g. 1.2k, 15.4k)."""
        if score >= 10_000:
            return f"{score / 1000:.1f}k"
        elif score >= 1_000:
            return f"{score / 1000:.1f}k"
        return str(score)

    def _build_html(
        self,
        subreddit: str,
        title: str,
        author: str,
        score: int,
        theme: Dict[str, str],
    ) -> str:
        """Build the complete HTML string for a Reddit post card.

        The card mimics Reddit's dark/light mode UI with accurate
        spacing, typography, and iconography. Supports a custom warning-badge
        hook mode if the author is 'system' or subreddit is 'Warning'.

        Args:
            subreddit: Subreddit name (without ``r/`` prefix).
            title: Post title text.
            author: Post author username.
            score: Post score/upvotes.
            theme: Colour token dictionary.

        Returns:
            A self-contained HTML string.
        """
        if author.lower() == "system" or subreddit.lower() == "warning":
            safe_title = html_escape(title)
            return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1080, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@700&display=swap" rel="stylesheet">
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }}
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
  }}
  body {{
    display: flex;
    justify-content: center;
    align-items: flex-start;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .card {{
    background: #1A1A1B;
    border: 3px solid #FF4500;
    border-radius: 12px;
    width: 600px;
    padding: 30px 24px;
    box-sizing: border-box;
    text-align: center;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
  }}
  .warning-badge {{
    color: #FF4500;
    font-size: 24px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
  }}
  .post-title {{
    font-size: 26px;
    font-weight: 700;
    color: #FFFFFF;
    line-height: 1.45;
    word-wrap: break-word;
    letter-spacing: -0.02em;
  }}
</style>
</head>
<body>
  <div class="card" id="reddit-card">
    <div class="warning-badge">
      <span>⚠️</span> WARNING <span>⚠️</span>
    </div>
    <h3 class="post-title">{safe_title}</h3>
  </div>
</body>
</html>"""

        avatar_color = self._get_avatar_color(subreddit)
        avatar_letter = subreddit[0].upper() if subreddit else "R"
        score_display = self._format_score(score)
        comment_count = random.randint(42, 1200)
        hours_ago = random.randint(2, 18)

        # Escape user-supplied strings for HTML safety
        safe_title = html_escape(title)
        safe_subreddit = html_escape(subreddit)
        safe_author = html_escape(author)

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1080, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }}
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
  }}
  body {{
    display: flex;
    justify-content: center;
    align-items: flex-start;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}

  .card {{
    background: {theme['card_bg']};
    border: 1px solid {theme['card_border']};
    border-radius: 8px;
    width: 600px;
    padding: 20px;
    box-sizing: border-box;
    overflow: hidden;
  }}

  /* ── Vote sidebar + Content area ── */
  .card-inner {{
    display: flex;
    flex-direction: row;
  }}

  .vote-sidebar {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0 10px 0 0;
    min-width: 40px;
    gap: 4px;
  }}

  .vote-btn {{
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  .vote-btn svg {{
    width: 22px;
    height: 22px;
  }}

  .vote-score {{
    font-size: 14px;
    font-weight: 700;
    color: {theme['score_color']};
    line-height: 1.2;
  }}

  .content-area {{
    flex: 1;
    padding: 0;
  }}

  /* ── Header (subreddit + author + time) ── */
  .post-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
  }}

  .avatar {{
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: {avatar_color};
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 700;
    color: #FFFFFF;
    flex-shrink: 0;
  }}

  .header-text {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
    font-size: 13px;
    line-height: 1.4;
  }}

  .subreddit-name {{
    color: {theme['subreddit_color']};
    font-weight: 700;
    text-decoration: none;
  }}

  .header-dot {{
    color: {theme['meta_color']};
    font-size: 11px;
  }}

  .posted-by {{
    color: {theme['author_color']};
    font-weight: 400;
  }}

  .time-ago {{
    color: {theme['meta_color']};
    font-weight: 400;
  }}

  /* ── Title ── */
  .post-title {{
    font-size: 21px;
    font-weight: 600;
    color: {theme['title_color']};
    line-height: 1.35;
    margin-bottom: 12px;
    word-wrap: break-word;
    letter-spacing: -0.01em;
  }}

  /* ── Footer (comments, share, etc.) ── */
  .post-footer {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 0 6px 0;
  }}

  .footer-btn {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 700;
    color: {theme['icon_color']};
    cursor: pointer;
    transition: background 0.15s;
  }}

  .footer-btn svg {{
    width: 20px;
    height: 20px;
    fill: {theme['icon_color']};
  }}
</style>
</head>
<body>
  <div class="card" id="reddit-card">
    <div class="card-inner">
      <!-- Vote sidebar -->
      <div class="vote-sidebar">
        <div class="vote-btn">
          <svg viewBox="0 0 24 24" fill="{theme['upvote_color']}">
            <path d="M12 4 3 15h6v5h6v-5h6z"/>
          </svg>
        </div>
        <span class="vote-score">{score_display}</span>
        <div class="vote-btn">
          <svg viewBox="0 0 24 24" fill="{theme['icon_color']}" opacity="0.5">
            <path d="M12 20l9-11h-6V4H9v5H3z"/>
          </svg>
        </div>
      </div>

      <!-- Content -->
      <div class="content-area">
        <div class="post-header">
          <div class="avatar">{avatar_letter}</div>
          <div class="header-text">
            <span class="subreddit-name">r/{safe_subreddit}</span>
            <span class="header-dot">•</span>
            <span class="posted-by">Posted by u/{safe_author}</span>
            <span class="header-dot">•</span>
            <span class="time-ago">{hours_ago}h ago</span>
          </div>
        </div>

        <h3 class="post-title">{safe_title}</h3>

        <div class="post-footer">
          <div class="footer-btn">
            <svg viewBox="0 0 24 24">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            {comment_count} Comments
          </div>
          <div class="footer-btn">
            <svg viewBox="0 0 24 24">
              <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/>
              <polyline points="16 6 12 2 8 6"/>
              <line x1="12" y1="2" x2="12" y2="15"/>
            </svg>
            Share
          </div>
          <div class="footer-btn">
            <svg viewBox="0 0 24 24">
              <path d="M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z"/>
            </svg>
            Save
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>"""

    def _build_comment_html(
        self,
        author: str,
        text: str,
        score: int,
        theme: Dict[str, str],
    ) -> str:
        """Build the complete HTML string for a Reddit comment card.

        The card mimics Reddit's comment UI in dark/light mode.

        Args:
            author: Comment author username.
            text: Comment text.
            score: Comment score/upvotes.
            theme: Colour token dictionary.

        Returns:
            A self-contained HTML string.
        """
        avatar_color = self._get_avatar_color(author)
        avatar_letter = author[0].upper() if author else "U"
        score_display = self._format_score(score)
        hours_ago = random.randint(1, 10)

        # Escape user-supplied strings for HTML safety
        safe_author = html_escape(author)
        safe_text = html_escape(text).replace("\n", "<br>")

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1080, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }}
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
  }}
  body {{
    display: flex;
    justify-content: center;
    align-items: flex-start;
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}

  .card {{
    background: {theme['card_bg']};
    border: 1px solid {theme['card_border']};
    border-radius: 8px;
    width: 600px;
    padding: 20px;
    box-sizing: border-box;
    overflow: hidden;
  }}

  /* ── Vote sidebar + Content area ── */
  .card-inner {{
    display: flex;
    flex-direction: row;
  }}

  .vote-sidebar {{
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 0 10px 0 0;
    min-width: 40px;
    gap: 4px;
  }}

  .vote-btn {{
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
  }}

  .vote-btn svg {{
    width: 22px;
    height: 22px;
  }}

  .vote-score {{
    font-size: 14px;
    font-weight: 700;
    color: {theme['score_color']};
    line-height: 1.2;
  }}

  .content-area {{
    flex: 1;
    padding: 0;
  }}

  /* ── Header ── */
  .post-header {{
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
  }}

  .avatar {{
    width: 26px;
    height: 26px;
    border-radius: 50%;
    background: {avatar_color};
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 700;
    color: #FFFFFF;
    flex-shrink: 0;
  }}

  .header-text {{
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px;
    font-size: 13px;
    line-height: 1.4;
  }}

  .author-name {{
    color: {theme['subreddit_color']};
    font-weight: 700;
    text-decoration: none;
  }}

  .header-dot {{
    color: {theme['meta_color']};
    font-size: 11px;
  }}

  .time-ago {{
    color: {theme['meta_color']};
    font-weight: 400;
  }}

  /* ── Comment Body ── */
  .comment-body {{
    font-size: 15px;
    font-weight: 400;
    color: {theme['title_color']};
    line-height: 1.45;
    margin-bottom: 12px;
    word-wrap: break-word;
  }}

  /* ── Footer ── */
  .post-footer {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 4px 0 6px 0;
  }}

  .footer-btn {{
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 13px;
    font-weight: 700;
    color: {theme['icon_color']};
    cursor: pointer;
    transition: background 0.15s;
  }}

  .footer-btn svg {{
    width: 20px;
    height: 20px;
    fill: {theme['icon_color']};
  }}
</style>
</head>
<body>
  <div class="card" id="reddit-card">
    <div class="card-inner">
      <!-- Vote sidebar -->
      <div class="vote-sidebar">
        <div class="vote-btn">
          <svg viewBox="0 0 24 24" fill="{theme['upvote_color']}">
            <path d="M12 4 3 15h6v5h6v-5h6z"/>
          </svg>
        </div>
        <span class="vote-score">{score_display}</span>
        <div class="vote-btn">
          <svg viewBox="0 0 24 24" fill="{theme['icon_color']}" opacity="0.5">
            <path d="M12 20l9-11h-6V4H9v5H3z"/>
          </svg>
        </div>
      </div>

      <!-- Content area -->
      <div class="content-area">
        <div class="post-header">
          <div class="avatar">{avatar_letter}</div>
          <div class="header-text">
            <span class="author-name">u/{safe_author}</span>
            <span class="header-dot">•</span>
            <span class="time-ago">{hours_ago}h ago</span>
          </div>
        </div>

        <div class="comment-body">{safe_text}</div>

        <div class="post-footer">
          <div class="footer-btn">
            <svg viewBox="0 0 24 24">
              <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/>
            </svg>
            Reply
          </div>
          <div class="footer-btn">
            <svg viewBox="0 0 24 24">
              <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/>
              <polyline points="16 6 12 2 8 6"/>
              <line x1="12" y1="2" x2="12" y2="15"/>
            </svg>
            Share
          </div>
        </div>
      </div>
    </div>
  </div>
</body>
</html>"""

    def capture_poll_card(self, story: Dict[str, Any], index: int, state: str) -> Path:
        """Render a Would You Rather card state and capture a screenshot.

        Args:
            story: Dictionary containing option_a, option_b, percentage_a.
            index: The index of the question.
            state: The state of the countdown ('question', '3', '2', '1', 'reveal').

        Returns:
            Path to the saved PNG screenshot file.
        """
        theme_name = self.screenshot_cfg.get("theme", "dark")
        theme = (
            self._DARK_THEME if theme_name == "dark" else self._LIGHT_THEME
        )

        option_a = story.get("option_a", "Option A")
        option_b = story.get("option_b", "Option B")
        percentage_a = story.get("percentage_a", 50)

        html_content = self._build_poll_html(
            option_a=option_a,
            option_b=option_b,
            percentage_a=percentage_a,
            state=state,
            theme=theme,
        )

        output_path = self.temp_dir / f"wyr_{index}_{state}.png"

        self.logger.info(
            "Capturing WYR poll card screenshot (state=%s, index=%d)",
            state,
            index,
        )

        try:
            self._render_and_capture(html_content, output_path)
        except Exception as exc:
            self.logger.error("WYR poll card screenshot capture failed: %s", exc)
            raise RuntimeError(
                f"Failed to capture WYR poll card screenshot: {exc}"
            ) from exc

        return output_path

    def capture_outro_card(self, text: str) -> Path:
        """Render a beautiful Outro card and capture a screenshot.

        Args:
            text: The outro text to display (e.g., "Follow for more").

        Returns:
            Path to the saved PNG screenshot file.
        """
        safe_text = html_escape(text)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1080, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }}
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
  }}
  body {{
    display: flex;
    justify-content: center;
    align-items: flex-start;
    font-family: 'Outfit', sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  .card {{
    background: linear-gradient(180deg, #16161a, #0d0d0f);
    border: 3px solid #FFE500;
    box-shadow: 0 0 25px rgba(255, 229, 0, 0.4), inset 0 0 15px rgba(255, 229, 0, 0.15);
    border-radius: 20px;
    width: 600px;
    padding: 40px 24px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 24px;
    overflow: hidden;
  }}
  .icon-container {{
    width: 90px;
    height: 90px;
    background: linear-gradient(135deg, #FFE500, #FFB000);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 0 15px rgba(255, 229, 0, 0.5);
  }}
  .plus-icon {{
    color: #030303;
    font-size: 54px;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 4px;
  }}
  .outro-title {{
    color: #ffffff;
    font-size: 36px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    text-align: center;
    line-height: 1.3;
    text-shadow: 0 2px 10px rgba(0, 0, 0, 0.5);
  }}
  .outro-subtitle {{
    color: #FFE500;
    font-size: 20px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    text-align: center;
    text-shadow: 0 0 8px rgba(255, 229, 0, 0.3);
  }}
</style>
</head>
<body>
  <div class="card">
    <div class="icon-container">
      <span class="plus-icon">+</span>
    </div>
    <div class="outro-title">
      {safe_text}
    </div>
    <div class="outro-subtitle">
      New content daily
    </div>
  </div>
</body>
</html>"""

        output_path = self.temp_dir / "outro_screenshot.png"

        self.logger.info("Capturing outro card screenshot: text='%s'", text)

        try:
            self._render_and_capture(html_content, output_path)
        except Exception as exc:
            self.logger.error("Outro card screenshot capture failed: %s", exc)
            raise RuntimeError(
                f"Failed to capture Outro card screenshot: {exc}"
            ) from exc

        return output_path

    def _build_poll_html(
        self,
        option_a: str,
        option_b: str,
        percentage_a: int,
        state: str,
        theme: Dict[str, str],
    ) -> str:
        """Build the complete HTML string for a Would You Rather card."""
        safe_option_a = html_escape(option_a)
        safe_option_b = html_escape(option_b)
        percentage_b = 100 - percentage_a

        is_reveal = (state == "reveal")
        is_winner_a = (percentage_a >= 50)

        class_a = ""
        class_b = ""
        if is_reveal:
            class_a = "winner" if is_winner_a else "loser"
            class_b = "loser" if is_winner_a else "winner"

        progress_pct = 100
        seconds_left = "5"
        if state.isdigit():
            seconds_left = state
            val = int(state)
            progress_pct = int((val / 5.0) * 100)
        elif state == "question":
            progress_pct = 100
            seconds_left = "READY"

        if is_reveal:
            bottom_markup = f"""
            <div class="results-bar-container">
                <div class="results-bar">
                    <div class="bar-fill fill-a" style="width: {percentage_a}%;">
                        <span class="bar-label">{percentage_a}%</span>
                    </div>
                    <div class="bar-fill fill-b" style="width: {percentage_b}%;">
                        <span class="bar-label">{percentage_b}%</span>
                    </div>
                </div>
            </div>
            """
        else:
            fill_style = f"width: {progress_pct}%;"
            if state == "question":
                fill_style = "width: 100%; background: #343536;"
            bottom_markup = f"""
            <div class="timer-container">
                <div class="progress-bar">
                    <div class="progress-fill" style="{fill_style}"></div>
                    <span class="timer-text">{seconds_left}</span>
                </div>
            </div>
            """

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=1080, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  * {{
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }}
  html, body {{
    margin: 0;
    padding: 0;
    background: transparent;
  }}
  body {{
    display: flex;
    justify-content: center;
    align-items: flex-start;
    font-family: 'Outfit', sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  
  .card {{
    background: linear-gradient(180deg, #16161a, #0d0d0f);
    border: 3px solid #FFE500;
    box-shadow: 0 0 20px rgba(255, 229, 0, 0.3), inset 0 0 15px rgba(255, 229, 0, 0.1);
    border-radius: 16px;
    width: 600px;
    padding: 24px;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 20px;
    overflow: hidden;
  }}
  
  .header {{
    text-align: center;
    color: #FFE500;
    font-size: 32px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    text-shadow: 0 0 10px rgba(255, 229, 0, 0.5);
  }}
  
  .options-stack {{
    display: flex;
    flex-direction: column;
    position: relative;
    gap: 16px;
  }}
  
  .option {{
    border-radius: 12px;
    padding: 20px 24px;
    min-height: 100px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: #ffffff;
    font-size: 22px;
    font-weight: 700;
    transition: all 0.3s ease;
    border: 2px solid transparent;
  }}
  
  .option-a {{
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    box-shadow: 0 6px 15px rgba(30, 60, 114, 0.3);
  }}
  
  .option-b {{
    background: linear-gradient(135deg, #b20a2c, #eb3c5a);
    box-shadow: 0 6px 15px rgba(178, 10, 44, 0.3);
  }}
  
  .option.winner {{
    border: 3px solid #46D160;
    box-shadow: 0 0 15px rgba(70, 209, 96, 0.6);
  }}
  
  .option.loser {{
    opacity: 0.45;
    filter: grayscale(40%);
  }}
  
  .option-text {{
    max-width: 80%;
    line-height: 1.3;
  }}
  
  .percentage {{
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -0.02em;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
  }}
  
  .or-circle {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    background: #0d0d0f;
    border: 3px solid #FFE500;
    color: #FFE500;
    width: 48px;
    height: 48px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: 800;
    z-index: 10;
    box-shadow: 0 0 10px rgba(255, 229, 0, 0.3);
  }}
  
  .timer-container {{
    width: 100%;
  }}
  
  .progress-bar {{
    background: #1C1C1E;
    height: 36px;
    border-radius: 18px;
    overflow: hidden;
    position: relative;
    border: 1px solid #343536;
  }}
  
  .progress-fill {{
    background: linear-gradient(90deg, #FFE500, #FFB000);
    height: 100%;
    transition: width 0.3s linear;
  }}
  
  .timer-text {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
  }}
  
  .results-bar-container {{
    width: 100%;
  }}
  
  .results-bar {{
    display: flex;
    height: 36px;
    border-radius: 18px;
    overflow: hidden;
    border: 1px solid #343536;
    background: #1C1C1E;
  }}
  
  .bar-fill {{
    display: flex;
    align-items: center;
    justify-content: center;
    color: #ffffff;
    font-weight: 700;
    font-size: 14px;
    transition: width 0.5s ease-out;
  }}
  
  .fill-a {{
    background: #1e3c72;
  }}
  
  .fill-b {{
    background: #b20a2c;
  }}
  
  .bar-label {{
    text-shadow: 0 1px 2px rgba(0,0,0,0.5);
  }}
</style>
</head>
<body>
  <div class="card" id="reddit-card">
    <div class="header">Would You Rather?</div>
    
    <div class="options-stack">
      <div class="option option-a {class_a}">
        <span class="option-text">{safe_option_a}</span>
        {f'<span class="percentage">{percentage_a}%</span>' if is_reveal else ''}
      </div>
      
      <div class="or-circle">OR</div>
      
      <div class="option option-b {class_b}">
        <span class="option-text">{safe_option_b}</span>
        {f'<span class="percentage">{percentage_b}%</span>' if is_reveal else ''}
      </div>
    </div>
    
    {bottom_markup}
  </div>
</body>
</html>"""

    # ---- Playwright renderer -----------------------------------------------

    def _render_and_capture(
        self, html_content: str, output_path: Path
    ) -> None:
        """Launch headless Chromium, render HTML, and capture screenshot.

        Uses Playwright's synchronous API to set page content from a
        raw HTML string (no network navigation to reddit.com), waits
        for fonts to load, then takes an element screenshot of the
        card container.

        Args:
            html_content: Complete HTML string to render.
            output_path: Where to save the PNG screenshot.
        """
        from playwright.sync_api import sync_playwright

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page(
                viewport={"width": 1080, "height": 1920},
                device_scale_factor=2,  # 2x for Retina-quality
            )

            # Render local HTML (no network navigation)
            page.set_content(html_content)

            # Wait for Google Fonts to load
            try:
                page.wait_for_load_state("networkidle", timeout=8000)
            except Exception:
                self.logger.warning(
                    "Font loading timed out — proceeding with fallback fonts"
                )

            # Capture element screenshot of the card
            card = page.locator("#reddit-card")
            card.screenshot(path=str(output_path), type="png", omit_background=True)

            browser.close()

        self.logger.debug("Playwright screenshot captured: %s", output_path)
