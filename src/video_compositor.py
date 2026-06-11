"""
RedditDaily-Bot — Video Compositor
====================================
Final compositing of gameplay, TTS audio, subtitles, and watermarks
into a publish-ready vertical Reel using MoviePy.

The :class:`VideoCompositor` receives pre-processed assets from upstream
pipeline stages and layers them together:

1. **Gameplay background** — hash-destroyed footage from
   :mod:`src.video_processor`.
2. **TTS narration** — audio from the ElevenLabs TTS stage.
3. **Subtitle overlays** — animated ``TextClip`` objects from the subtitle
   renderer.
4. **Part watermark** *(optional)* — "Part N of M" indicator for multi-part
   stories.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from moviepy.editor import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoFileClip,
    concatenate_videoclips,
    vfx,
    afx,
)

from src.utils import (
    BotLogger,
    ensure_dirs,
    resolve_path,
    sanitize_filename,
    timestamp_str,
)


class VideoCompositor:
    """Assembles all pipeline assets into the final vertical Reel.

    Attributes:
        config: Full application configuration dictionary.
        logger: BotLogger instance for structured logging.
        video_cfg: ``video`` sub-config (codec, resolution, fps …).
        splitter_cfg: ``splitter`` sub-config (watermark style …).
        output_dir: Resolved path where finished videos are written.
    """

    def __init__(self, config: dict, logger: BotLogger) -> None:
        """Initialise the compositor.

        Args:
            config: Parsed application config.
            logger: Shared BotLogger instance.
        """
        self.config: dict = config
        self.logger: BotLogger = logger
        self.video_cfg: dict = config.get("video", {})
        self.splitter_cfg: dict = config.get("splitter", {})

        output_dir_cfg: str = (
            config.get("pipeline", {}).get("output_dir")
            or self.video_cfg.get("output_dir", "output")
        )
        self.output_dir: Path = resolve_path(output_dir_cfg, create=True)
        self.logger.debug(
            f"VideoCompositor initialised — output: {self.output_dir}"
        )

    # ---- public API --------------------------------------------------------

    def compose(
        self,
        gameplay_path: Path,
        audio_path: Path,
        subtitle_clips: List[Any],
        part_info: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Composite all layers into the final output video.

        Args:
            gameplay_path: Path to the hash-destroyed gameplay ``.mp4``.
            audio_path: Path to the TTS narration audio file.
            subtitle_clips: List of MoviePy ``TextClip`` objects produced by
                the subtitle renderer.  Each clip already carries its own
                start/end timing and position.
            part_info: Optional dict for multi-part stories.  Expected keys::

                    {
                        "part_number": int,
                        "total_parts": int,
                        "subreddit": str,
                        "title": str,
                        "watermark_config": {
                            "text": "Part 1 of 2",
                            "position": "top-right",
                            "font_size": 28,
                            "color": "#FFFFFF",
                            "opacity": 0.7,
                        },
                    }

                When *None*, no watermark is rendered and the output filename
                omits the part suffix.

        Returns:
            Path to the finished ``.mp4`` file.

        Raises:
            FileNotFoundError: If *gameplay_path* or *audio_path* is missing.
            RuntimeError: On MoviePy compositing / encoding failure.
        """
        if not gameplay_path.exists():
            raise FileNotFoundError(
                f"Gameplay video not found: {gameplay_path}"
            )
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Gather metadata for the filename
        subreddit: str = (part_info or {}).get("subreddit", "reddit")
        title: str = (part_info or {}).get("title", "story")
        output_filename: str = self._generate_output_filename(
            part_info=part_info,
            subreddit=subreddit,
            title=title,
        )
        output_path: Path = self.output_dir / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Clip handles for cleanup
        clips_to_close: List[Any] = []

        try:
            # 1. Load gameplay video
            self.logger.info(f"Loading gameplay: {gameplay_path.name}")
            gameplay_clip: VideoFileClip = VideoFileClip(str(gameplay_path))
            clips_to_close.append(gameplay_clip)

            # Mix background music and transition SFX
            mixed_audio_path = self._mix_background_audio(audio_path)

            # 2. Load mixed audio (master timeline)
            self.logger.info(f"Loading mixed audio: {mixed_audio_path.name}")
            audio_clip: AudioFileClip = AudioFileClip(str(mixed_audio_path))
            # Apply standard fadeout transition over the last 1.5 seconds
            audio_clip = audio_clip.fx(afx.audio_fadeout, 1.5)
            clips_to_close.append(audio_clip)

            # 3. Audio duration is the master timeline
            audio_duration: float = audio_clip.duration
            self.logger.info(
                f"Audio duration (master): {audio_duration:.2f}s | "
                f"Gameplay duration: {gameplay_clip.duration:.2f}s"
            )

            # 4–5. Match gameplay length to audio
            if gameplay_clip.duration < audio_duration:
                self.logger.info(
                    "Gameplay shorter than audio — looping to match."
                )
                gameplay_clip = gameplay_clip.fx(
                    vfx.loop, duration=audio_duration
                )
                clips_to_close.append(gameplay_clip)
            elif gameplay_clip.duration > audio_duration:
                self.logger.info(
                    "Gameplay longer than audio — trimming to match."
                )
                gameplay_clip = gameplay_clip.subclip(0, audio_duration)
                clips_to_close.append(gameplay_clip)

            # 6. Replace gameplay audio with TTS narration
            gameplay_with_audio = gameplay_clip.set_audio(audio_clip)
            clips_to_close.append(gameplay_with_audio)

            # 7–8. Build the layer stack
            layers: List[Any] = [gameplay_with_audio]

            # Subtitle overlays
            if subtitle_clips:
                self.logger.info(
                    f"Adding {len(subtitle_clips)} subtitle clip(s)."
                )
                layers.extend(subtitle_clips)

            # Watermark for multi-part stories
            watermark_clip: Optional[Any] = None
            if part_info and part_info.get("watermark_config"):
                wm_cfg: Dict[str, Any] = part_info["watermark_config"]
                frame_size: Tuple[int, int] = (
                    self.video_cfg.get("resolution", {}).get("width", 1080),
                    self.video_cfg.get("resolution", {}).get("height", 1920),
                )
                watermark_clip = self._create_watermark(
                    text=wm_cfg.get("text", ""),
                    config=wm_cfg,
                    duration=audio_duration,
                    frame_size=frame_size,
                )
                clips_to_close.append(watermark_clip)
                layers.append(watermark_clip)
                self.logger.info(
                    f"Watermark added: '{wm_cfg.get('text', '')}'"
                )

            # 9. Composite all layers
            self.logger.info("Compositing final video …")
            frame_w: int = self.video_cfg.get("resolution", {}).get(
                "width", 1080
            )
            frame_h: int = self.video_cfg.get("resolution", {}).get(
                "height", 1920
            )
            final_clip: CompositeVideoClip = CompositeVideoClip(
                layers, size=(frame_w, frame_h)
            )
            clips_to_close.append(final_clip)

            # 10. Write the final output
            codec: str = self.video_cfg.get("codec", "libx264")
            audio_codec: str = self.video_cfg.get("audio_codec", "aac")
            fps: int = self.video_cfg.get("fps", 30)
            
            # Ensure preset is 'medium' or 'fast' for proper H.264 compression standards
            preset: str = self.video_cfg.get("preset", "medium")
            if preset not in ["medium", "fast"]:
                self.logger.warning(f"Video preset '{preset}' is not optimized for Reels. Forcing 'medium'.")
                preset = "medium"

            # Ensure video bitrate is explicitly set to 5000k
            bitrate: str = self.video_cfg.get("bitrate", "5000k")
            if not bitrate:
                bitrate = "5000k"

            # Ensure audio bitrate is set to 192k for high quality audio
            audio_bitrate: str = self.video_cfg.get("audio_bitrate", "192k")
            if not audio_bitrate:
                audio_bitrate = "192k"

            self.logger.info(
                f"Encoding → {output_path.name} "
                f"({codec}, {fps}fps, preset={preset}, "
                f"v_bitrate={bitrate}, a_bitrate={audio_bitrate})"
            )

            final_clip.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec=audio_codec,
                fps=fps,
                preset=preset,
                bitrate=bitrate,
                audio_bitrate=audio_bitrate,
                ffmpeg_params=["-map_metadata", "-1"],
                logger=None,  # suppress MoviePy's own progress bars
            )

            file_size_mb: float = output_path.stat().st_size / (1024 * 1024)
            self.logger.info(
                f"Final video saved: {output_path.name} "
                f"({file_size_mb:.1f} MB, {audio_duration:.1f}s)"
            )
            return output_path

        except Exception as exc:
            self.logger.error(f"Compositing failed: {exc}")
            raise RuntimeError(f"Video compositing failed: {exc}") from exc

        finally:
            # 11. Deterministic cleanup — close every clip handle
            for clip in clips_to_close:
                try:
                    clip.close()
                except Exception:
                    pass

    # ---- private helpers ---------------------------------------------------

    def _mix_background_audio(self, narration_path: Path) -> Path:
        """Mix background music and pop transition SFX into the narration audio.

        Args:
            narration_path: Path to the clean narration audio (MP3/WAV).

        Returns:
            Path to the new mixed audio file.
        """
        from pydub import AudioSegment
        from pydub.generators import Sine
        import random

        # Load narration audio
        self.logger.info("Loading narration audio for mixing...")
        narration = AudioSegment.from_file(str(narration_path))
        narration_dur_ms = len(narration)

        # 1. Resolve and load Transition SFX
        sfx_dir = resolve_path("assets/audio/sfx", create=True)
        sfx_files = (
            list(sfx_dir.glob("*.mp3"))
            + list(sfx_dir.glob("*.wav"))
            + list(sfx_dir.glob("*.m4a"))
        )
        
        sfx_clip = None
        if sfx_files:
            sfx_path = random.choice(sfx_files)
            self.logger.info(f"Selected random SFX: {sfx_path.name}")
            try:
                sfx_clip = AudioSegment.from_file(str(sfx_path))
            except Exception as e:
                self.logger.warning(f"Failed to load SFX {sfx_path.name}: {e}")
        
        # If no SFX found, generate a procedural pop SFX (synthesized)
        if sfx_clip is None:
            self.logger.info("Generating procedural bubble pop SFX...")
            p1 = Sine(600).to_audio_segment(duration=40).fade_in(5)
            p2 = Sine(900).to_audio_segment(duration=80).fade_out(60)
            sfx_clip = (p1 + p2) - 8
            sfx_path = sfx_dir / "pop.mp3"
            try:
                sfx_clip.export(str(sfx_path), format="mp3")
                self.logger.info(f"Saved procedural SFX to {sfx_path}")
            except Exception as e:
                self.logger.warning(f"Could not save procedural SFX to disk: {e}")

        # 2. Resolve and load Background Music
        bg_music_dir = resolve_path("assets/audio/bg_music", create=True)
        bg_music_files = (
            list(bg_music_dir.glob("*.mp3"))
            + list(bg_music_dir.glob("*.wav"))
            + list(bg_music_dir.glob("*.m4a"))
        )
        
        bg_music_clip = None
        if bg_music_files:
            bg_music_path = random.choice(bg_music_files)
            self.logger.info(f"Selected random background music: {bg_music_path.name}")
            try:
                bg_music_clip = AudioSegment.from_file(str(bg_music_path))
            except Exception as e:
                self.logger.warning(f"Failed to load background music {bg_music_path.name}: {e}")
                
        # If no background music found, generate a procedural ambient track
        if bg_music_clip is None:
            self.logger.info("Generating procedural ambient drone track...")
            drone_dur_ms = 180000
            d1 = Sine(65).to_audio_segment(duration=drone_dur_ms) - 22
            d2 = Sine(110).to_audio_segment(duration=drone_dur_ms) - 25
            d3 = Sine(130).to_audio_segment(duration=drone_dur_ms) - 28
            bg_music_clip = d1.overlay(d2).overlay(d3)
            bg_music_path = bg_music_dir / "ambient_drone.mp3"
            try:
                bg_music_clip.export(str(bg_music_path), format="mp3")
                self.logger.info(f"Saved procedural background music to {bg_music_path}")
            except Exception as e:
                self.logger.warning(f"Could not save procedural background music to disk: {e}")

        # Mix the pop SFX exactly at timestamp 00:00 (0ms)
        self.logger.info("Mixing pop SFX at 00:00...")
        mixed = narration.overlay(sfx_clip, position=0)

        # Mix the background music underneath at a lowered volume
        # Lowered volume: multiplier of 0.07, which is roughly -23.1 dB.
        self.logger.info("Mixing background music (volume = 7%)...")
        bg_music_volume_adjusted = bg_music_clip - 23.1

        # Match background music duration to narration duration (loop or slice)
        if len(bg_music_volume_adjusted) < narration_dur_ms:
            loops_needed = (narration_dur_ms // len(bg_music_volume_adjusted)) + 1
            bg_music_volume_adjusted = bg_music_volume_adjusted * loops_needed
        bg_music_final = bg_music_volume_adjusted[:narration_dur_ms]

        mixed = mixed.overlay(bg_music_final, position=0)

        # Save the final mixed audio to a temporary file
        temp_dir = resolve_path(self.config.get("pipeline", {}).get("temp_dir", "temp"), create=True)
        mixed_audio_path = temp_dir / f"mixed_{narration_path.name}"
        mixed.export(str(mixed_audio_path), format="mp3", bitrate="128k")
        self.logger.info(f"Mixed audio saved to: {mixed_audio_path}")
        return mixed_audio_path

    def _create_watermark(
        self,
        text: str,
        config: Dict[str, Any],
        duration: float,
        frame_size: Tuple[int, int],
    ) -> Any:
        """Create a semi-transparent watermark ``ImageClip``.

        Args:
            text: Watermark string (e.g. ``"Part 1 of 3"``).
            config: Watermark styling dict with keys *font_size*, *color*,
                *opacity*, *position*.
            duration: How long the watermark should be visible (seconds).
            frame_size: ``(width, height)`` of the output frame.

        Returns:
            A positioned, timed ``ImageClip`` ready for compositing.
        """
        from PIL import Image, ImageDraw, ImageFont
        import numpy as np
        from moviepy.editor import ImageClip

        font_size: int = config.get("font_size", self.splitter_cfg.get(
            "watermark_font_size", 28
        ))
        color: str = config.get("color", self.splitter_cfg.get(
            "watermark_color", "#FFFFFF"
        ))
        opacity: float = config.get("opacity", self.splitter_cfg.get(
            "watermark_opacity", 0.7
        ))
        position: str = config.get("position", self.splitter_cfg.get(
            "watermark_position", "top-right"
        ))

        # Padding from frame edges
        padding: int = 20
        frame_w, frame_h = frame_size

        # Resolve font
        try:
            pil_font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            pil_font = ImageFont.load_default()

        # Measure text bounding box
        img = Image.new("RGBA", (1, 1))
        draw = ImageDraw.Draw(img)
        bbox = draw.textbbox((0, 0), text, font=pil_font)
        
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        stroke_width = 1
        w = tw + stroke_width * 2 + 4
        h = th + stroke_width * 2 + 4

        # Create transparent canvas and draw text
        canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)

        tx = stroke_width + 2 - bbox[0]
        ty = (h - th) // 2 - bbox[1]

        draw.text(
            (tx, ty),
            text,
            font=pil_font,
            fill=color,
            stroke_width=stroke_width,
            stroke_fill="#000000",
        )

        # Convert to numpy array for MoviePy
        img_np = np.array(canvas)
        color_np = img_np[:, :, :3]
        alpha_np = img_np[:, :, 3] / 255.0

        wm_clip = ImageClip(color_np)
        mask = ImageClip(alpha_np, ismask=True)
        wm_clip = wm_clip.set_mask(mask)

        # Resolve position string to (x, y)
        wm_w, wm_h = w, h

        if position == "top-right":
            pos_x: int = frame_w - wm_w - padding
            pos_y: int = padding
        elif position == "top-left":
            pos_x = padding
            pos_y = padding
        elif position == "bottom-right":
            pos_x = frame_w - wm_w - padding
            pos_y = frame_h - wm_h - padding
        elif position == "bottom-left":
            pos_x = padding
            pos_y = frame_h - wm_h - padding
        else:
            # Default: top-right
            pos_x = frame_w - wm_w - padding
            pos_y = padding

        wm_clip = (
            wm_clip
            .set_position((pos_x, pos_y))
            .set_duration(duration)
            .set_opacity(opacity)
            .set_start(0.0)
        )

        self.logger.debug(
            f"Watermark '{text}' at ({pos_x}, {pos_y}), "
            f"opacity={opacity}, duration={duration:.1f}s"
        )
        return wm_clip

    def _generate_output_filename(
        self,
        part_info: Optional[Dict[str, Any]],
        subreddit: str,
        title: str,
    ) -> str:
        """Build a descriptive, filesystem-safe output filename.

        Format::

            {timestamp}_{subreddit}_{title}.mp4          (single-part)
            {timestamp}_{subreddit}_{title}_part1.mp4     (multi-part)

        Args:
            part_info: Part metadata dict or *None* for single-part stories.
            subreddit: Source subreddit name.
            title: Original post title (will be sanitised).

        Returns:
            Filename string ending in ``.mp4``.
        """
        ts: str = timestamp_str()
        safe_sub: str = sanitize_filename(subreddit, max_len=30)
        safe_title: str = sanitize_filename(title, max_len=40)

        if part_info and part_info.get("part_number") is not None:
            part_n: int = part_info["part_number"]
            return f"{ts}_{safe_sub}_{safe_title}_part{part_n}.mp4"

        return f"{ts}_{safe_sub}_{safe_title}.mp4"
