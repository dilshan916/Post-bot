"""
RedditDaily-Bot — Cover Image Generator
========================================
Generates high-contrast, viral 1080x1920 standalone Reel cover images
specifically optimized for the 'Reddit Daily' Facebook Page.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from PIL import Image, ImageDraw, ImageFont
from src.utils import BotLogger, resolve_path, PROJECT_ROOT

class CoverGenerator:
    """Generates 1080x1920 yellow/black high-contrast cover thumbnails."""

    def __init__(self, config: Dict[str, Any], logger: Optional[BotLogger] = None):
        self.config = config
        self.log = logger or BotLogger("CoverGenerator")
        
        # Load fonts
        font_dir = PROJECT_ROOT / "assets" / "fonts"
        black_font_p = font_dir / "ArialBlack.ttf"
        bold_font_p = font_dir / "ArialBold.ttf"
        
        if black_font_p.exists():
            self.font_huge = ImageFont.truetype(str(black_font_p), 76)
            self.font_badge = ImageFont.truetype(str(black_font_p), 40)
        else:
            self.font_huge = ImageFont.load_default()
            self.font_badge = ImageFont.load_default()
            
        if bold_font_p.exists():
            self.font_bold = ImageFont.truetype(str(bold_font_p), 55)
            self.font_sub = ImageFont.truetype(str(bold_font_p), 38)
        else:
            self.font_bold = ImageFont.load_default()
            self.font_sub = ImageFont.load_default()

    def _draw_hazard_icon(self, draw: ImageDraw.ImageDraw, cx: int, cy: int):
        """Draws a crisp, hazard triangle icon."""
        radius = 20
        p1 = (cx, cy - radius)
        p2 = (cx - radius, cy + radius)
        p3 = (cx + radius, cy + radius)
        draw.polygon([p1, p2, p3], fill=(0, 0, 0))
        draw.line([(cx, cy - 8), (cx, cy + 4)], fill=(255, 229, 0), width=6)
        draw.ellipse([(cx - 3, cy + 9), (cx + 3, cy + 15)], fill=(255, 229, 0))

    def generate_cover(
        self,
        title: str,
        subreddit: str = "RedditStories",
        sticker_path: Optional[Path] = None,
        output_path: Optional[Path] = None,
    ) -> Path:
        """Renders a 1080x1920 standalone cover image.

        Args:
            title: Raw post title or script text.
            subreddit: Subreddit origin name.
            sticker_path: Optional path to character emotion sticker PNG.
            output_path: Target .jpg output path.

        Returns:
            Path to generated cover image.
        """
        output_path = output_path or (PROJECT_ROOT / "output" / "cover_latest.jpg")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Base 1080x1920 canvas
        img = Image.new("RGBA", (1080, 1920), (255, 229, 0, 255))
        draw = ImageDraw.Draw(img)

        # Main dark central block
        draw.rectangle([0, 280, 1080, 1620], fill=(12, 12, 12, 255))

        # Top Yellow Header Strip
        draw.rectangle([0, 280, 1080, 400], fill=(255, 229, 0, 255))
        self._draw_hazard_icon(draw, 100, 340)
        draw.text((140, 315), "UNFORGIVABLE BETRAYAL", fill=(0, 0, 0), font=self.font_badge)
        self._draw_hazard_icon(draw, 980, 340)

        # Process headline text into 3 clean lines
        clean_title = title.upper().replace("\n", " ").strip()
        words = clean_title.split()
        
        line1 = " ".join(words[:4]) if len(words) >= 4 else clean_title
        line2 = " ".join(words[4:7]) if len(words) >= 7 else "SHOCKING TWIST!"
        line3 = " ".join(words[7:11]) if len(words) >= 11 else "LISTEN TILL THE END!"

        # Draw Headline lines
        draw.text((90, 460), line1[:22], fill=(255, 255, 255), font=self.font_huge)
        
        # Red highlight box for Line 2
        draw.rounded_rectangle([80, 580, 990, 710], radius=15, fill=(225, 29, 72))
        draw.text((100, 595), line2[:18], fill=(255, 255, 255), font=self.font_huge)
        
        # Yellow text for Line 3
        draw.text((90, 740), line3[:22], fill=(255, 229, 0), font=self.font_huge)

        # Bottom Call-To-Action Banner
        draw.rounded_rectangle([90, 960, 990, 1070], radius=20, fill=(255, 229, 0, 255))
        draw.text((140, 995), "FULL STORY • LISTEN TILL THE END", fill=(0, 0, 0), font=self.font_sub)

        # Character Sticker overlay if available
        if sticker_path and sticker_path.exists():
            try:
                stk = Image.open(sticker_path).convert("RGBA").resize((460, 460))
                img.paste(stk, (310, 1120), stk)
            except Exception as e:
                self.log.warning(f"Could not overlay sticker on cover: {e}")

        # Save final RGB JPG image
        img.convert("RGB").save(output_path, "JPEG", quality=95)
        self.log.info(f"Generated standalone cover image: {output_path}")
        return output_path
