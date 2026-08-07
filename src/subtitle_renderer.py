"""
RedditDaily-Bot — Subtitle Renderer (Double-Pass Dynamic Captions)
===================================================================
Renders word-by-word highlighted subtitles using a two-pass technique:

1. **Passive pass** – the full sentence is displayed in muted white.
2. **Active pass**  – each word lights up in neon yellow exactly when
   it is spoken, overlaid on top of its passive counterpart.

Public API
----------
    renderer = SubtitleRenderer(config)
    clips    = renderer.render(timestamp_data, frame_size)
"""

from __future__ import annotations

import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils import BotLogger, resolve_path, PROJECT_ROOT


class SubtitleRenderer:
    """Create MoviePy TextClip overlays for double-pass dynamic captions.

    Args:
        config: Parsed configuration dictionary (from ``load_config``).
        logger: Optional pre-configured ``BotLogger``.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[BotLogger] = None,
    ) -> None:
        self.config = config
        sub_cfg = config.get("subtitles", {})
        pipeline_cfg = config.get("pipeline", {})

        # Font settings - default to 65 or 95 (for max_words=1) for readable vertical format
        self.font_family: str = sub_cfg.get("font_family", "Arial Black")
        self.max_words: int = int(sub_cfg.get("max_words", 1))
        
        # Override for shower thoughts mode: force single word and large font size
        pipeline_mode = pipeline_cfg.get("pipeline_mode", "monologue")
        if pipeline_mode == "shower":
            self.max_words = 1
            default_font_size = 95
        else:
            default_font_size = 95 if self.max_words == 1 else 65
            
        self.font_size: int = int(sub_cfg.get("font_size", default_font_size))
        self.font_bold: bool = bool(sub_cfg.get("font_bold", True))

        # Passive pass
        self.passive_color: str = sub_cfg.get("passive_color", "#FFFFFF")
        self.passive_opacity: float = float(
            sub_cfg.get("passive_opacity", 0.40)
        )
        self.passive_stroke_color: str = sub_cfg.get(
            "passive_stroke_color", "#000000"
        )
        self.passive_stroke_width: int = int(
            sub_cfg.get("passive_stroke_width", 5)
        )

        # Active pass
        self.active_color: str = sub_cfg.get("active_color", "#FFE500")
        self.active_opacity: float = float(
            sub_cfg.get("active_opacity", 1.0)
        )
        self.active_stroke_color: str = sub_cfg.get(
            "active_stroke_color", "#000000"
        )
        self.active_stroke_width: int = int(
            sub_cfg.get("active_stroke_width", 6)
        )

        # Layout
        self.vertical_position: float = float(
            sub_cfg.get("vertical_position", 0.72)
        )
        self.max_chars_per_line: int = int(
            sub_cfg.get("max_chars_per_line", 35)
        )

        self.logger = logger or BotLogger(
            name="SubtitleRenderer",
            log_dir=pipeline_cfg.get("log_dir", "output/logs"),
            level=pipeline_cfg.get("log_level", "INFO"),
        )

    # ------------------------------------------------------------------
    # Font resolution
    # ------------------------------------------------------------------
    def _resolve_font(self) -> str:
        """Resolve the font family to a usable font file path or name.

        If ``font_family`` points to a file in ``assets/fonts/`` it is
        returned as an absolute path string. Otherwise, on Windows, it resolves
        the system font name to the actual .ttf file path in C:\\Windows\\Fonts.

        Returns:
            Absolute font path or system font name.
        """
        font_name = self.font_family.strip()

        # 1. Check if the config value is already an absolute/relative path
        font_path = Path(font_name)
        if font_path.exists():
            return str(font_path.resolve())

        # 2. Check assets/fonts/
        font_dir = PROJECT_ROOT / "assets" / "fonts"
        if font_dir.exists():
            for ext in (".ttf", ".otf", ".TTF", ".OTF"):
                candidate = font_dir / f"{font_name}{ext}"
                if candidate.exists():
                    return str(candidate)
                if self.font_bold:
                    for bold_suffix in ("bd", "b", " Bold", "-Bold", "Bold"):
                        candidate = font_dir / f"{font_name}{bold_suffix}{ext}"
                        if candidate.exists():
                            return str(candidate)

        # 3. Try to resolve system font file name on Windows to a real file path
        if os.name == "nt":
            win_fonts_dir = Path(os.environ.get("WINDIR", "C:\\Windows")) / "Fonts"
            if win_fonts_dir.exists():
                candidates = []
                name_clean = font_name.lower().replace(" ", "")
                
                if name_clean == "arial":
                    candidates = ["arialbd.ttf", "arial.ttf"] if self.font_bold else ["arial.ttf"]
                elif name_clean in ("arialblack", "ariblk"):
                    candidates = ["ariblk.ttf"]
                elif name_clean == "calibri":
                    candidates = ["calibrib.ttf", "calibri.ttf"] if self.font_bold else ["calibri.ttf"]
                elif name_clean == "segoeui":
                    candidates = ["segoeuib.ttf", "segoeui.ttf"] if self.font_bold else ["segoeui.ttf"]
                elif name_clean == "tahoma":
                    candidates = ["tahomabd.ttf", "tahoma.ttf"] if self.font_bold else ["tahoma.ttf"]
                elif name_clean == "verdana":
                    candidates = ["verdanab.ttf", "verdana.ttf"] if self.font_bold else ["verdana.ttf"]
                elif name_clean == "impact":
                    candidates = ["impact.ttf"]
                else:
                    # Generic lookup for unspecified fonts
                    exts = [".ttf", ".otf"]
                    for ext in exts:
                        if self.font_bold:
                            for bold_suffix in ("bd", "b", "bold", "-bold"):
                                candidates.append(f"{name_clean}{bold_suffix}{ext}")
                        candidates.append(f"{name_clean}{ext}")

                # Search the Windows Fonts directory case-insensitively
                for cand in candidates:
                    for f in win_fonts_dir.glob("*"):
                        if f.name.lower() == cand.lower():
                            self.logger.info("Resolved system font to file path: %s", f)
                            return str(f.resolve())

        return font_name

    def _get_pil_font(
        self, font_ref: str, font_size: int
    ) -> "PIL.ImageFont.FreeTypeFont | PIL.ImageFont.ImageFont":
        """Load a PIL font object for text measurement.

        Args:
            font_ref: System font name or path to a .ttf/.otf file.
            font_size: Desired point size.

        Returns:
            A PIL font object.
        """
        from PIL import ImageFont  # type: ignore[import-untyped]
        from pathlib import Path

        if os.path.isfile(font_ref):
            try:
                return ImageFont.truetype(font_ref, font_size)
            except (OSError, IOError) as exc:
                self.logger.warning(
                    "Could not load font file '%s': %s. Falling back to default.",
                    font_ref,
                    exc,
                )
                return ImageFont.load_default()

        # Try searching project assets/fonts directory
        project_root = Path(__file__).resolve().parent.parent
        asset_fonts = [
            project_root / "assets" / "fonts" / f"{font_ref}.ttf",
            project_root / "assets" / "fonts" / f"{font_ref.replace(' ', '')}.ttf",
            project_root / "assets" / "fonts" / "ArialBlack.ttf",
            project_root / "assets" / "fonts" / "ArialBold.ttf",
            project_root / "assets" / "fonts" / "Impact.ttf",
        ]
        for font_path in asset_fonts:
            if font_path.exists():
                try:
                    return ImageFont.truetype(str(font_path), font_size)
                except (OSError, IOError):
                    pass

        # Try standard system loading
        try:
            return ImageFont.truetype(font_ref, font_size)
        except (OSError, IOError):
            pass

        self.logger.warning(
            "Font '%s' not found. Using PIL default font.", font_ref
        )
        return ImageFont.load_default()

    # ------------------------------------------------------------------
    # Text measurement
    # ------------------------------------------------------------------
    def _measure_text(
        self, text: str, font: str, font_size: int
    ) -> Tuple[int, int]:
        """Measure the pixel dimensions of *text* when rendered.

        Uses PIL to compute an accurate bounding box. Special logic handles
        the space character to prevent default PIL empty box collapses (0px).

        Args:
            text: The string to measure.
            font: Font name or path.
            font_size: Point size.

        Returns:
            ``(width, height)`` in pixels.
        """
        from PIL import Image, ImageDraw  # type: ignore[import-untyped]

        pil_font = self._get_pil_font(font, font_size)

        img = Image.new("RGB", (1, 1))
        draw = ImageDraw.Draw(img)
        
        if text == " ":
            # Special space measurement formula: width("A B") - width("A") - width("B")
            # This yields the true font-specific space width instead of 0 or 1.
            w_ab = draw.textbbox((0, 0), "A B", font=pil_font)
            w_a = draw.textbbox((0, 0), "A", font=pil_font)
            w_b = draw.textbbox((0, 0), "B", font=pil_font)
            width = (w_ab[2] - w_ab[0]) - (w_a[2] - w_a[0]) - (w_b[2] - w_b[0])
            height = max(w_ab[3] - w_ab[1], w_a[3] - w_a[1])
            return max(width, 1), max(height, 1)

        bbox = draw.textbbox((0, 0), text, font=pil_font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        return max(width, 1), max(height, 1)

    # ------------------------------------------------------------------
    # Word position calculation
    # ------------------------------------------------------------------
    def _wrap_words_by_pixel_width(
        self,
        words: List[Dict[str, Any]],
        font: str,
        font_size: int,
        max_pixel_width: int,
    ) -> List[List[Dict[str, Any]]]:
        """Wrap words into lines dynamically based on their measured pixel width and character length."""
        lines: List[List[Dict[str, Any]]] = [[]]
        
        for w in words:
            if not lines[-1]:
                # First word in the line
                lines[-1].append(w)
            else:
                # Test if adding this word to the current line exceeds max_pixel_width
                current_words = lines[-1] + [w]
                line_text = " ".join([wd["word"] for wd in current_words]).upper()
                width, _ = self._measure_text(line_text, font, font_size)
                
                # Check character limit from config as a fallback check
                chars_len = len(" ".join([wd["word"] for wd in current_words]))
                
                if width > max_pixel_width or chars_len > self.max_chars_per_line:
                    lines.append([w])
                else:
                    lines[-1].append(w)
                    
        return lines

    def _calculate_word_positions(
        self,
        sentence_words: List[Dict[str, Any]],
        font: str,
        font_size: int,
        frame_width: int,
        max_chars_per_line: int,
    ) -> List[Dict[str, Any]]:
        """Calculate the pixel position of every word inside a sentence.

        The sentence is wrapped dynamically based on pixel width and
        each word receives ``{word, x, y, width, height, line_index}``.

        Args:
            sentence_words: Word dicts from timestamp data.
            font: Font name or path.
            font_size: Point size.
            frame_width: Video frame width in pixels.
            max_chars_per_line: Maximum characters before line-wrap.

        Returns:
            List of position dicts, one per word, in the same order.
        """
        # ------ Build wrapped lines ------
        max_pixel_width = int(frame_width * 0.82)
        lines = self._wrap_words_by_pixel_width(
            sentence_words, font, font_size, max_pixel_width
        )

        # Use font metrics for line height consistency
        pil_font = self._get_pil_font(font, font_size)
        ascent, descent = pil_font.getmetrics()
        font_line_height = ascent + descent
        line_spacing = int(font_line_height * 0.15)

        # Pre-calculate line bboxes to get exact drawing Y offsets
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(img)

        line_y_offsets = []
        current_y = 0

        for line_words in lines:
            line_str = " ".join([w["word"] for w in line_words]).upper()
            bbox = draw.textbbox((0, 0), line_str, font=pil_font)
            line_y_offsets.append(current_y - bbox[1])
            current_y += font_line_height + line_spacing

        positions: List[Dict[str, Any]] = []

        for line_idx, line_words in enumerate(lines):
            line_str = " ".join([w["word"] for w in line_words]).upper()
            tw, _ = self._measure_text(line_str, font, font_size)

            # Center the line horizontally relative to the frame
            line_x_offset = (frame_width - tw) // 2

            for i, w in enumerate(line_words):
                # Measure prefix text width before this word
                if i == 0:
                    prefix_w = 0
                else:
                    prefix_text = " ".join([wd["word"] for wd in line_words[:i]]).upper() + " "
                    prefix_w, _ = self._measure_text(prefix_text, font, font_size)

                ww, wh = self._measure_text(w["word"].upper(), font, font_size)
                positions.append(
                    {
                        "word": w["word"],
                        "x": line_x_offset + prefix_w,
                        "y": line_y_offsets[line_idx],
                        "width": ww,
                        "height": wh,
                        "line_index": line_idx,
                    }
                )

        return positions

    # ------------------------------------------------------------------
    # PIL Image helper for Text rendering
    # ------------------------------------------------------------------
    def _create_pil_text_clip(
        self,
        text: str,
        font_size: int,
        color: str,
        stroke_color: Optional[str] = None,
        stroke_width: int = 0,
        size: Optional[Tuple[int, Optional[int]]] = None,
        align: str = "center",
        duration: float = 1.0,
        start: float = 0.0,
        opacity: float = 1.0,
    ) -> Tuple[Any, Tuple[int, int], int, int]:
        from PIL import Image, ImageDraw  # type: ignore[import-untyped]
        from moviepy.editor import ImageClip  # type: ignore[import-untyped]
        import numpy as np

        text = text.upper()
        font_ref = self._resolve_font()
        pil_font = self._get_pil_font(font_ref, font_size)

        # Set up a dummy draw context to measure lines
        img = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(img)

        lines = text.split("\n")
        line_data = []
        max_line_w = 0

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=pil_font)
            lw = bbox[2] - bbox[0]
            lh = bbox[3] - bbox[1]
            line_data.append((line, bbox, lw, lh))
            if lw > max_line_w:
                max_line_w = lw

        ascent, descent = pil_font.getmetrics()
        font_line_height = ascent + descent
        line_spacing = int(font_line_height * 0.15)

        line_y_offsets = []
        current_y = 0
        for idx, (line, bbox, lw, lh) in enumerate(line_data):
            line_y_offsets.append(current_y - bbox[1])
            current_y += font_line_height + line_spacing

        th = current_y - line_spacing

        # Determine canvas size
        if size is not None:
            w = int(size[0])
            h = int(size[1] if size[1] is not None else (th + stroke_width * 2 + 10))
        else:
            w = int(max_line_w + stroke_width * 2 + 10)
            h = int(th + stroke_width * 2 + 10)

        w, h = max(w, 1), max(h, 1)

        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        y_start = stroke_width + 5

        # Render line by line
        for idx, (line, bbox, lw, lh) in enumerate(line_data):
            # Center horizontally relative to canvas
            line_x = (w - lw) // 2 - bbox[0]
            line_y = y_start + line_y_offsets[idx]

            # Draw outline stroke
            if stroke_width > 0 and stroke_color:
                draw.text(
                    (line_x, line_y),
                    line,
                    font=pil_font,
                    fill=stroke_color,
                    stroke_width=stroke_width,
                    stroke_fill=stroke_color,
                )
            # Draw inner fill
            draw.text(
                (line_x, line_y),
                line,
                font=pil_font,
                fill=color,
            )

        img_np = np.array(canvas)
        color_np = img_np[:, :, :3]
        alpha_np = img_np[:, :, 3] / 255.0

        clip = ImageClip(color_np)
        mask = ImageClip(alpha_np, ismask=True)
        clip = clip.set_mask(mask)

        clip = clip.set_opacity(opacity)
        clip = clip.set_start(start)
        clip = clip.set_duration(duration)

        return clip, (w, h), 0, y_start

    # ------------------------------------------------------------------
    # Active Word Canvas Helper
    # ------------------------------------------------------------------
    def _create_active_word_clip(
        self,
        word_text: str,
        font_ref: str,
        font_size: int,
        color: str,
        stroke_color: Optional[str] = None,
        stroke_width: int = 0,
        size: Tuple[int, int] = (972, 100),
        draw_pos: Tuple[int, int] = (0, 0),
        duration: float = 1.0,
        start: float = 0.0,
        opacity: float = 1.0,
    ) -> Any:
        from PIL import Image, ImageDraw  # type: ignore[import-untyped]
        from moviepy.editor import ImageClip  # type: ignore[import-untyped]
        import numpy as np

        word_text = word_text.upper()
        pil_font = self._get_pil_font(font_ref, font_size)

        w, h = size
        w, h = max(w, 1), max(h, 1)

        # Create transparent canvas matching the exact size of the passive sentence
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # Render ONLY the active word (No drop shadow as per user request)
        if stroke_width > 0 and stroke_color:
            draw.text(
                draw_pos,
                word_text,
                font=pil_font,
                fill=stroke_color,
                stroke_width=stroke_width,
                stroke_fill=stroke_color,
            )
        draw.text(
            draw_pos,
            word_text,
            font=pil_font,
            fill=color,
        )

        img_np = np.array(canvas)
        color_np = img_np[:, :, :3]
        alpha_np = img_np[:, :, 3] / 255.0

        clip = ImageClip(color_np)
        mask = ImageClip(alpha_np, ismask=True)
        clip = clip.set_mask(mask)

        clip = clip.set_opacity(opacity)
        clip = clip.set_start(start)
        clip = clip.set_duration(duration)

        return clip

    # ------------------------------------------------------------------
    # Passive sentence clip
    # ------------------------------------------------------------------
    def _render_passive_sentence(
        self,
        sentence: Dict[str, Any],
        frame_size: Tuple[int, int],
        font_size: int,
    ) -> Tuple[Any, Tuple[int, int], int, int]:
        """Render the full sentence/chunk as a muted passive ImageClip.

        Args:
            sentence: Sentence dict with ``text``, ``start``, ``end``.
            frame_size: ``(width, height)`` of the video frame.
            font_size: Desired point size of the font.

        Returns:
            Tuple of (clip, canvas_size, tx, ty).
        """
        frame_width, frame_height = frame_size

        font_ref = self._resolve_font()
        max_pixel_width = int(frame_width * 0.82)
        lines = self._wrap_words_by_pixel_width(
            sentence["words"], font_ref, font_size, max_pixel_width
        )
        wrapped = "\n".join([" ".join([w["word"] for w in line]) for line in lines])

        duration = sentence["end"] - sentence["start"]
        if duration <= 0:
            duration = 0.1

        canvas_width = int(frame_width * 0.90)

        clip, canvas_size, tx, ty = self._create_pil_text_clip(
            text=wrapped,
            font_size=font_size,
            color=self.passive_color,
            stroke_color=self.passive_stroke_color,
            stroke_width=self.passive_stroke_width,
            size=(canvas_width, None),
            align="center",
            duration=duration,
            start=sentence["start"],
            opacity=self.passive_opacity,
        )

        y_pos = int(frame_height * self.vertical_position)
        clip = clip.set_position(("center", y_pos))

        return clip, canvas_size, tx, ty

    # ------------------------------------------------------------------
    # Active word clips
    # ------------------------------------------------------------------
    def _render_active_words(
        self,
        sentence: Dict[str, Any],
        positions: List[Dict[str, Any]],
        frame_size: Tuple[int, int],
        canvas_size: Tuple[int, int],
        tx: int,
        ty: int,
        font_size: int,
    ) -> List[Any]:
        """Render individual active-highlight clips for each word.

        Each active word is rendered on a transparent canvas matching the exact
        passive canvas size and centered at the exact same coordinates. This
        ensures pixel-perfect sub-pixel overlay alignment.

        Args:
            sentence: Sentence dict.
            positions: Word positions from ``_calculate_word_positions``.
            frame_size: ``(width, height)`` of the video frame.
            canvas_size: ``(w, h)`` of the passive sentence canvas.
            tx: Horizontal draw origin of the passive text block inside the canvas.
            ty: Vertical draw origin of the passive text block inside the canvas.
            font_size: Point size of the active font.

        Returns:
            List of MoviePy ``ImageClip`` objects.
        """
        frame_width, frame_height = frame_size
        y_base = int(frame_height * self.vertical_position)

        clips: List[Any] = []
        sent_words = sentence.get("words", [])
        font_ref = self._resolve_font()

        zoom_enabled = self.config.get("subtitles", {}).get("word_zoom_animation", True)

        for i, w in enumerate(sent_words):
            if i >= len(positions):
                continue

            pos = positions[i]
            word_text = w["word"].upper()
            word_start = w["start"]
            word_end = w["end"]
            duration = word_end - word_start
            if duration <= 0:
                duration = 0.05

            # Measure word_bbox to adjust the origin offset
            from PIL import Image, ImageDraw
            img_temp = Image.new("RGBA", (1, 1))
            draw_temp = ImageDraw.Draw(img_temp)
            pil_font = self._get_pil_font(font_ref, font_size)
            word_bbox = draw_temp.textbbox((0, 0), word_text, font=pil_font)
            
            # Determine active word highlight color based on speaker if available
            speaker = w.get("speaker")
            if speaker:
                speaker_clean = str(speaker).strip().upper()
                if speaker_clean == "MALE":
                    word_color = "#00FF00"  # Green text highlight
                elif speaker_clean == "FEMALE":
                    word_color = "#FFE500"  # Yellow text highlight
                else:
                    word_color = self.active_color
            else:
                word_color = self.active_color
            
            if zoom_enabled:
                from moviepy.editor import vfx
                # Calculate small cropped canvas size for the word
                pad = self.active_stroke_width * 2 + 10
                w_w = pos["width"] + pad
                w_h = pos["height"] + pad
                
                # Draw position relative to the small canvas
                draw_x = self.active_stroke_width + 5 - word_bbox[0]
                draw_y = self.active_stroke_width + 5 - word_bbox[1]
                
                word_clip = self._create_active_word_clip(
                    word_text=word_text,
                    font_ref=font_ref,
                    font_size=font_size,
                    color=word_color,
                    stroke_color=self.active_stroke_color,
                    stroke_width=self.active_stroke_width,
                    size=(w_w, w_h),
                    draw_pos=(draw_x, draw_y),
                    duration=duration,
                    start=word_start,
                    opacity=self.active_opacity,
                )
                
                # Dynamic shrink-to-fit pop animation (zoom_factor down to 1.0x in 0.12s)
                zoom_factor = self.config.get("subtitles", {}).get("word_zoom_factor", 1.05)
                zoom_diff = zoom_factor - 1.0

                def zoom_effect(t):
                    if t < 0.12:
                        return 1.0 + zoom_diff * (1.0 - t / 0.12)
                    return 1.0
                
                word_clip = word_clip.fx(vfx.resize, zoom_effect)
                
                # Compensate for the internal drawing offset to align the active text pixel-perfectly with passive text
                pos_x = int(pos["x"] - (self.active_stroke_width + 5) + word_bbox[0])
                pos_y = int(y_base + ty + pos["y"] - (self.active_stroke_width + 5) + word_bbox[1])
                
                # Center-anchored zoom positioning function to keep scaling focused on the word's center
                def make_zoom_pos(px, py, width_w, height_h):
                    def zoom_pos(t):
                        scale = zoom_effect(t)
                        x = px + (width_w / 2.0) * (1.0 - scale)
                        y = py + (height_h / 2.0) * (1.0 - scale)
                        return (x, y)
                    return zoom_pos

                word_clip = word_clip.set_position(make_zoom_pos(pos_x, pos_y, w_w, w_h))
            else:
                # Absolute drawing coordinate inside the matching canvas
                draw_x = int(pos["x"] - (frame_width - canvas_size[0]) // 2 - word_bbox[0])
                draw_y = int(ty + pos["y"])

                word_clip = self._create_active_word_clip(
                    word_text=word_text,
                    font_ref=font_ref,
                    font_size=font_size,
                    color=word_color,
                    stroke_color=self.active_stroke_color,
                    stroke_width=self.active_stroke_width,
                    size=canvas_size,
                    draw_pos=(draw_x, draw_y),
                    duration=duration,
                    start=word_start,
                    opacity=self.active_opacity,
                )

                # Center-align on the exact same frame coordinate as the passive clip
                word_clip = word_clip.set_position(("center", y_base))
                
            clips.append(word_clip)

        return clips

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_sentence_text(
        words: List[Dict[str, Any]], max_chars: int
    ) -> str:
        """Wrap sentence words into lines respecting *max_chars*.

        Args:
            words: List of word dicts (only ``word`` key used).
            max_chars: Maximum characters per line.

        Returns:
            Multi-line string with ``\\n`` separators.
        """
        lines: List[str] = []
        current_line: List[str] = []
        current_len = 0

        for w in words:
            token = w["word"]
            token_len = len(token)

            if current_line and current_len + 1 + token_len > max_chars:
                lines.append(" ".join(current_line))
                current_line = []
                current_len = 0

            current_line.append(token)
            current_len += token_len + (1 if current_len > 0 else 0)

        if current_line:
            lines.append(" ".join(current_line))

        return "\n".join(lines)

    def _split_sentence_into_chunks(
        self, sentence: Dict[str, Any], max_words: int = 4
    ) -> List[Dict[str, Any]]:
        """Split a long sentence into shorter word blocks (3-5 words).

        This keeps captions short, punchy, and dynamic like TikTok/Reels.

        Args:
            sentence: Sentence dict containing word mappings.
            max_words: Target maximum words per sequence on screen (default 4).

        Returns:
            List of virtual sentence dicts.
        """
        words = sentence.get("words", [])
        if not words:
            return []
            
        chunks = []
        n = len(words)
        
        if n <= 5 and max_words > 1:
            # Keep as a single sequence if it's 5 words or fewer
            chunks.append(words)
        else:
            # Simple chunking with target size of max_words
            for i in range(0, n, max_words):
                chunks.append(words[i:i + max_words])
                
        virtual_sentences = []
        for chunk in chunks:
            if not chunk:
                continue
            chunk_text = " ".join([w["word"] for w in chunk])
            virtual_sentences.append({
                "text": chunk_text,
                "start": chunk[0]["start"],
                "end": chunk[-1]["end"],
                "words": chunk
            })
            
        return virtual_sentences

    # ------------------------------------------------------------------
    # Main public interface
    # ------------------------------------------------------------------
    def _get_scaled_font_size(self, text: str, font_ref: str, max_width: int) -> int:
        from PIL import Image, ImageDraw
        img = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(img)
        
        current_size = self.font_size
        while current_size > 30:
            pil_font = self._get_pil_font(font_ref, current_size)
            lines = text.split("\n")
            max_line_w = 0
            for line in lines:
                bbox = draw.textbbox((0, 0), line.upper(), font=pil_font)
                line_w = bbox[2] - bbox[0]
                if line_w > max_line_w:
                    max_line_w = line_w
            
            if max_line_w > max_width - 10:
                current_size -= 4
            else:
                break
        return max(current_size, 30)

    def render(
        self,
        timestamp_data: Dict[str, Any],
        frame_size: Tuple[int, int] = (1080, 1920),
        skip_until: float = 0.0,
    ) -> List[Any]:
        """Produce double-pass subtitle clips for every sentence.

        Args:
            timestamp_data: Output of
                ``TimestampExtractor.extract()``, containing at least
                a ``sentences`` key.
            frame_size: ``(width, height)`` of the target video frame.
                Defaults to 1080 × 1920 (vertical short-form).
            skip_until: Time threshold (seconds). Subtitle chunks starting
                before this time will be omitted to avoid overlapping with
                the Reddit title card screenshot.

        Returns:
            Flat list of MoviePy clip objects (``TextClip`` instances)
            ready to be passed to ``CompositeVideoClip``.
        """
        sentences = timestamp_data.get("sentences", [])
        if not sentences:
            self.logger.warning(
                "No sentences found in timestamp data — "
                "returning empty clip list."
            )
            return []

        font_ref = self._resolve_font()
        frame_width, frame_height = frame_size

        # Split sentences into short blocks of 3-5 words
        short_sentences = []
        for sentence in sentences:
            virtual_sentences = self._split_sentence_into_chunks(sentence, max_words=self.max_words)
            short_sentences.extend(virtual_sentences)

        if skip_until > 0.0:
            filtered_short_sentences = []
            for s in short_sentences:
                if s["start"] < skip_until:
                    continue
                filtered_short_sentences.append(s)
            self.logger.info(
                f"skip_until={skip_until}s: Omitted {len(short_sentences) - len(filtered_short_sentences)} "
                f"subtitle chunks that fall within the title card duration."
            )
            short_sentences = filtered_short_sentences

        all_clips: List[Any] = []
        total_passive = 0
        total_active = 0

        for idx, sentence in enumerate(short_sentences):
            sent_words = sentence.get("words", [])
            if not sent_words:
                continue
            
            # Keep font size constant across the entire video
            scaled_font_size = self.font_size

            # Calculate pixel positions for this sentence chunk using scaled_font_size
            positions = self._calculate_word_positions(
                sent_words,
                font_ref,
                scaled_font_size,
                frame_width,
                self.max_chars_per_line,
            )

            # Pass 1: passive (full sentence chunk)
            try:
                passive_clip, canvas_size, tx, ty = self._render_passive_sentence(
                    sentence, frame_size, scaled_font_size
                )
                all_clips.append(passive_clip)
                total_passive += 1
            except Exception as exc:
                self.logger.error(
                    "Failed to render passive clip for sentence chunk %d: %s",
                    idx,
                    exc,
                )
                continue

            # Pass 2: active (per-word highlights inside shared canvas width/height)
            try:
                active_clips = self._render_active_words(
                    sentence, positions, frame_size, canvas_size, tx, ty, scaled_font_size
                )
                all_clips.extend(active_clips)
                total_active += len(active_clips)
            except Exception as exc:
                self.logger.error(
                    "Failed to render active clips for sentence chunk %d: %s",
                    idx,
                    exc,
                )

        self.logger.info(
            "Subtitle rendering complete — %d sentence chunks, "
            "%d passive clips, %d active word clips, %d total.",
            len(short_sentences),
            total_passive,
            total_active,
            len(all_clips),
        )
        return all_clips
