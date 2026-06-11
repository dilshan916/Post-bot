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
        self.font_family: str = sub_cfg.get("font_family", "Arial")
        self.max_words: int = int(sub_cfg.get("max_words", 1))
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
            sub_cfg.get("passive_stroke_width", 2)
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
            sub_cfg.get("active_stroke_width", 3)
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

        if os.path.isfile(font_ref):
            try:
                return ImageFont.truetype(font_ref, font_size)
            except (OSError, IOError) as exc:
                self.logger.warning(
                    "Could not load font file '%s': %s. "
                    "Falling back to default.",
                    font_ref,
                    exc,
                )
                return ImageFont.load_default()

        # Try standard loading
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
    def _calculate_word_positions(
        self,
        sentence_words: List[Dict[str, Any]],
        font: str,
        font_size: int,
        frame_width: int,
        max_chars_per_line: int,
    ) -> List[Dict[str, Any]]:
        """Calculate the pixel position of every word inside a sentence.

        The sentence is wrapped according to *max_chars_per_line* and
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
        lines: List[List[Dict[str, Any]]] = [[]]
        current_line_chars = 0

        for w in sentence_words:
            word_text = w["word"]
            word_len = len(word_text)

            if (
                lines[-1]
                and current_line_chars + 1 + word_len > max_chars_per_line
            ):
                lines.append([])
                current_line_chars = 0

            lines[-1].append(w)
            current_line_chars += word_len + (1 if current_line_chars > 0 else 0)

        # ------ Measure each word ------
        space_w, _ = self._measure_text(" ", font, font_size)
        _, line_height = self._measure_text("Aqy|", font, font_size)
        line_spacing = int(line_height * 0.25)

        positions: List[Dict[str, Any]] = []

        for line_idx, line_words in enumerate(lines):
            word_widths: List[int] = []
            word_heights: List[int] = []
            for w in line_words:
                ww, wh = self._measure_text(w["word"], font, font_size)
                word_widths.append(ww)
                word_heights.append(wh)

            total_line_width = (
                sum(word_widths) + space_w * max(len(line_words) - 1, 0)
            )

            # Centre horizontally relative to the frame
            x_offset = (frame_width - total_line_width) // 2

            for i, w in enumerate(line_words):
                positions.append(
                    {
                        "word": w["word"],
                        "x": x_offset,
                        "y": line_idx * (line_height + line_spacing),
                        "width": word_widths[i],
                        "height": word_heights[i],
                        "line_index": line_idx,
                    }
                )
                x_offset += word_widths[i] + space_w

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

        font_ref = self._resolve_font()
        pil_font = self._get_pil_font(font_ref, font_size)

        # Measure text bounding box
        img = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(img)
        bbox = draw.multiline_textbbox((0, 0), text, font=pil_font, align=align)
        
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        # Determine canvas size (add padding for stroke/bounding safety)
        if size is not None:
            w = int(size[0])
            h = int(size[1] if size[1] is not None else (th + stroke_width * 2 + 10))
        else:
            w = int(tw + stroke_width * 2 + 10)
            h = int(th + stroke_width * 2 + 10)

        w, h = max(w, 1), max(h, 1)

        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # Calculate drawing origin
        if size is not None:
            tx = int((w - tw) // 2 - bbox[0])
        else:
            tx = int(stroke_width + 2 - bbox[0])
            
        ty = int((h - th) // 2 - bbox[1])

        draw.multiline_text(
            (tx, ty),
            text,
            font=pil_font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
            align=align,
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

        return clip, (w, h), tx, ty

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

        pil_font = self._get_pil_font(font_ref, font_size)

        w, h = size
        w, h = max(w, 1), max(h, 1)

        # Create transparent canvas matching the exact size of the passive sentence
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        # Render ONLY the active word at the matching sentence layout coordinate
        draw.text(
            draw_pos,
            word_text,
            font=pil_font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill=stroke_color,
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
    ) -> Tuple[Any, Tuple[int, int], int, int]:
        """Render the full sentence/chunk as a muted passive ImageClip.

        Args:
            sentence: Sentence dict with ``text``, ``start``, ``end``.
            frame_size: ``(width, height)`` of the video frame.

        Returns:
            Tuple of (clip, canvas_size, tx, ty).
        """
        frame_width, frame_height = frame_size

        wrapped = self._wrap_sentence_text(
            sentence["words"], self.max_chars_per_line
        )

        duration = sentence["end"] - sentence["start"]
        if duration <= 0:
            duration = 0.1

        canvas_width = int(frame_width * 0.90)

        clip, canvas_size, tx, ty = self._create_pil_text_clip(
            text=wrapped,
            font_size=self.font_size,
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

        Returns:
            List of MoviePy ``ImageClip`` objects.
        """
        frame_width, frame_height = frame_size
        y_base = int(frame_height * self.vertical_position)

        clips: List[Any] = []
        sent_words = sentence.get("words", [])
        font_ref = self._resolve_font()

        for i, w in enumerate(sent_words):
            if i >= len(positions):
                continue

            pos = positions[i]
            word_text = w["word"]
            word_start = w["start"]
            word_end = w["end"]
            duration = word_end - word_start
            if duration <= 0:
                duration = 0.05

            # Measure word_bbox to adjust the origin offset
            from PIL import Image, ImageDraw
            img_temp = Image.new("RGBA", (1, 1))
            draw_temp = ImageDraw.Draw(img_temp)
            pil_font = self._get_pil_font(font_ref, self.font_size)
            word_bbox = draw_temp.textbbox((0, 0), word_text, font=pil_font)
            
            # Absolute drawing coordinate inside the matching canvas
            # draw_x = word_x_canvas - word_bbox[0]
            draw_x = int(pos["x"] - (frame_width - canvas_size[0]) // 2 - word_bbox[0])
            draw_y = int(ty + pos["y"])

            word_clip = self._create_active_word_clip(
                word_text=word_text,
                font_ref=font_ref,
                font_size=self.font_size,
                color=self.active_color,
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
    def render(
        self,
        timestamp_data: Dict[str, Any],
        frame_size: Tuple[int, int] = (1080, 1920),
    ) -> List[Any]:
        """Produce double-pass subtitle clips for every sentence.

        Args:
            timestamp_data: Output of
                ``TimestampExtractor.extract()``, containing at least
                a ``sentences`` key.
            frame_size: ``(width, height)`` of the target video frame.
                Defaults to 1080 × 1920 (vertical short-form).

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

        all_clips: List[Any] = []
        total_passive = 0
        total_active = 0

        for idx, sentence in enumerate(short_sentences):
            sent_words = sentence.get("words", [])
            if not sent_words:
                continue

            # Calculate pixel positions for this sentence chunk
            positions = self._calculate_word_positions(
                sent_words,
                font_ref,
                self.font_size,
                frame_width,
                self.max_chars_per_line,
            )

            # Pass 1: passive (full sentence chunk)
            try:
                passive_clip, canvas_size, tx, ty = self._render_passive_sentence(
                    sentence, frame_size
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
                    sentence, positions, frame_size, canvas_size, tx, ty
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
