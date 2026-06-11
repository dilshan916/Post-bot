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

    def create_hook_clip(
        self,
        screenshot_path: Path,
        frame_size: Tuple[int, int],
        duration: Optional[float] = None,
        fade_duration: Optional[float] = None,
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

        Returns:
            A positioned, timed ``ImageClip`` with fade-out, ready for
            compositing in the layer stack.
        """
        display_sec: float = duration or self.screenshot_cfg.get(
            "display_duration_sec", 3.5
        )
        fade_sec: float = fade_duration or self.screenshot_cfg.get(
            "fade_duration_sec", 0.8
        )
        card_width_pct: float = self.screenshot_cfg.get(
            "card_width_pct", 0.88
        )

        frame_w, frame_h = frame_size

        # Load screenshot with PIL to get dimensions and handle alpha
        img = Image.open(str(screenshot_path)).convert("RGBA")
        img_w, img_h = img.size

        # Calculate target width based on frame percentage
        target_w = int(frame_w * card_width_pct)
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
        pos_x = (frame_w - target_w) // 2
        pos_y = int(frame_h * 0.38) - (target_h // 2)

        clip = (
            clip
            .set_position((pos_x, pos_y))
            .set_start(0.0)
            .set_duration(display_sec)
            .fx(vfx.fadeout, fade_sec)
        )

        self.logger.info(
            "Hook clip created: %dx%d at (%d,%d), "
            "duration=%.1fs, fade=%.1fs",
            target_w, target_h, pos_x, pos_y, display_sec, fade_sec,
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
        spacing, typography, and iconography.

        Args:
            subreddit: Subreddit name (without ``r/`` prefix).
            title: Post title text.
            author: Post author username.
            score: Post score/upvotes.
            theme: Colour token dictionary.

        Returns:
            A self-contained HTML string.
        """
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
