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
        word_timestamps: Optional[List[Dict[str, Any]]] = None,
        dialogue_timings: Optional[List[Dict[str, Any]]] = None,
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
            word_timestamps: Optional word start/end timings for audio ducking.

        Returns:
            Path to the finished ``.mp4`` file.

        Raises:
            FileNotFoundError: If *gameplay_path* or *audio_path* is missing.
            RuntimeError: On MoviePy compositing / encoding failure.
        """
        conv_cfg = self.config.get("conversational", {})
        pipeline_mode = self.config.get("pipeline", {}).get("pipeline_mode", "monologue")
        use_static_bg = conv_cfg.get("use_static_backgrounds", False)
        if pipeline_mode == "riddle":
            use_static_bg = True

        if not (pipeline_mode in ("conversational", "riddle") and use_static_bg):
            if not gameplay_path or not gameplay_path.exists():
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
            frame_w: int = self.video_cfg.get("resolution", {}).get("width", 1080)
            frame_h: int = self.video_cfg.get("resolution", {}).get("height", 1920)

            # Mix background music and transition SFX
            mixed_audio_path = self._mix_background_audio(audio_path, word_timestamps)

            # 2. Load mixed audio (master timeline)
            self.logger.info(f"Loading mixed audio: {mixed_audio_path.name}")
            audio_clip: AudioFileClip = AudioFileClip(str(mixed_audio_path))
            # Apply standard fadeout transition over the last 1.5 seconds
            audio_clip = audio_clip.fx(afx.audio_fadeout, 1.5)
            clips_to_close.append(audio_clip)

            # 3. Audio duration is the master timeline
            audio_duration: float = audio_clip.duration

            # 1. Load background
            conv_cfg = self.config.get("conversational", {})
            pipeline_mode = self.config.get("pipeline", {}).get("pipeline_mode", "monologue")
            use_static_bg = conv_cfg.get("use_static_backgrounds", False)

            if pipeline_mode in ("conversational", "riddle") and use_static_bg:
                self.logger.info("Using static background images for conversational/riddle mode...")
                bg_clips = []
                last_end = 0.0

                # Smart Scene Auto-Matcher
                # Gather content text to detect setting
                title_text = f"{(part_info or {}).get('title', '')} {(part_info or {}).get('text', '')}".lower()
                setting = "living room" # Default
                
                # Match keywords
                if any(k in title_text for k in ("kitchen", "cook", "dining", "dinner", "breakfast", "lunch", "eat", "plate", "fridge", "stove")):
                    setting = "kitchen"
                elif any(k in title_text for k in ("office", "work", "boss", "colleague", "job", "career", "study", "desk", "computer", "meeting")):
                    setting = "office"
                elif any(k in title_text for k in ("bedroom", "sleep", "night", "bed", "dream", "nocturnal", "midnight", "awake")):
                    setting = "bedroom"
                
                self.logger.info(f"Smart Scene Matcher: Detected setting '{setting}' from story content.")
                
                # Find matching files in assets/backgrounds
                bg_dir = resolve_path("assets/backgrounds")
                matched_left = None
                matched_right = None
                
                if bg_dir.exists():
                    for f in bg_dir.iterdir():
                        name_lower = f.name.lower()
                        # Match left
                        if "left" in name_lower:
                            if setting == "kitchen" and "kitchen" in name_lower:
                                matched_left = f
                            elif setting == "office" and ("study" in name_lower or "office" in name_lower):
                                matched_left = f
                            elif setting == "bedroom" and "late-night" in name_lower:
                                matched_left = f
                            elif setting == "living room" and "living room" in name_lower:
                                matched_left = f
                        # Match right
                        elif "right" in name_lower:
                            if setting == "kitchen" and "kitchen" in name_lower:
                                matched_right = f
                            elif setting == "office" and ("study" in name_lower or "office" in name_lower):
                                matched_right = f
                            elif setting == "bedroom" and "late-night" in name_lower:
                                matched_right = f
                            elif setting == "living room" and "living room" in name_lower:
                                matched_right = f

                # Read configured paths
                cfg_male = resolve_path(conv_cfg.get("male_background_path", "assets/backgrounds/home_male.png"))
                cfg_female = resolve_path(conv_cfg.get("female_background_path", "assets/backgrounds/home_female.png"))
                cfg_old_male = resolve_path(conv_cfg.get("old_male_background_path", "assets/backgrounds/home_male.png"))
                cfg_old_female = resolve_path(conv_cfg.get("old_female_background_path", "assets/backgrounds/home_female.png"))
                cfg_chibi_male = resolve_path(conv_cfg.get("chibi_male_background_path", "assets/backgrounds/home_male.png"))
                cfg_chibi_female = resolve_path(conv_cfg.get("chibi_female_background_path", "assets/backgrounds/home_female.png"))

                # Prioritize configured paths if they exist and are not default placeholders
                male_bg = cfg_male if (cfg_male.exists() and cfg_male.name != "home_male.png") else (matched_left if matched_left else cfg_male)
                female_bg = cfg_female if (cfg_female.exists() and cfg_female.name != "home_female.png") else (matched_right if matched_right else cfg_female)
                old_male_bg = cfg_old_male if (cfg_old_male.exists() and cfg_old_male.name != "home_male.png") else (matched_left if matched_left else cfg_old_male)
                old_female_bg = cfg_old_female if (cfg_old_female.exists() and cfg_old_female.name != "home_female.png") else (matched_right if matched_right else cfg_old_female)
                chibi_male_bg = cfg_chibi_male if (cfg_chibi_male.exists() and cfg_chibi_male.name != "home_male.png") else (matched_left if matched_left else cfg_chibi_male)
                chibi_female_bg = cfg_chibi_female if (cfg_chibi_female.exists() and cfg_chibi_female.name != "home_female.png") else (matched_right if matched_right else cfg_chibi_female)

                self.logger.info(f"Using Left BG: {male_bg.name if hasattr(male_bg, 'name') else male_bg}")
                self.logger.info(f"Using Right BG: {female_bg.name if hasattr(female_bg, 'name') else female_bg}")

                if dialogue_timings:
                    for turn in dialogue_timings:
                        start = turn["start"]
                        end = turn["end"]
                        speaker = turn.get("speaker", "MALE").strip().upper()
                        duration = end - start

                        if duration <= 0:
                            continue

                        # Fill silence gaps
                        if start > last_end:
                            gap_clip = ImageClip(str(male_bg)).set_start(last_end).set_duration(start - last_end)
                            bg_clips.append(gap_clip)

                        # Determine image path
                        if speaker in ("CHILD_MALE", "CHIBI_MALE"):
                            bg_img_path = chibi_male_bg
                        elif speaker in ("CHILD_FEMALE", "CHIBI_FEMALE"):
                            bg_img_path = chibi_female_bg
                        elif speaker == "OLD_MALE":
                            bg_img_path = old_male_bg
                        elif speaker == "OLD_FEMALE":
                            bg_img_path = old_female_bg
                        else:
                            bg_img_path = male_bg if speaker == "MALE" else female_bg

                        if not bg_img_path.exists():
                            bg_img_path = male_bg

                        shot_clip = ImageClip(str(bg_img_path)).set_start(start).set_duration(duration)
                        bg_clips.append(shot_clip)
                        last_end = end

                    if last_end < audio_duration:
                        final_gap_clip = ImageClip(str(male_bg)).set_start(last_end).set_duration(audio_duration - last_end)
                        bg_clips.append(final_gap_clip)
                else:
                    full_bg_clip = ImageClip(str(male_bg)).set_start(0).set_duration(audio_duration)
                    bg_clips.append(full_bg_clip)

                gameplay_clip = CompositeVideoClip(bg_clips, size=(frame_w, frame_h))
                clips_to_close.append(gameplay_clip)
            else:
                self.logger.info(f"Loading gameplay: {gameplay_path.name}")
                gameplay_clip: VideoFileClip = VideoFileClip(str(gameplay_path))
                clips_to_close.append(gameplay_clip)

            self.logger.info(
                f"Audio duration (master): {audio_duration:.2f}s | "
                f"Gameplay duration: {audio_duration if (pipeline_mode in ('conversational', 'riddle') and use_static_bg) else gameplay_clip.duration:.2f}s"
            )

            # 4–5. Match gameplay length to audio
            if not (pipeline_mode in ("conversational", "riddle") and use_static_bg):
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

            # Speaker avatar stickers overlay (rendered behind subtitles)
            stickers_enabled = self.config.get("conversational", {}).get("stickers_enabled", True)
            if stickers_enabled and dialogue_timings:
                self.logger.info("Generating speaker avatar stickers overlay...")
                sticker_clips = self._create_speaker_stickers(
                    dialogue_timings=dialogue_timings,
                    frame_size=(frame_w, frame_h),
                )
                if sticker_clips:
                    self.logger.info(f"Adding {len(sticker_clips)} speaker avatar sticker overlay(s).")
                    layers.extend(sticker_clips)
                    clips_to_close.extend(sticker_clips)

            # Subtitle overlays (rendered on top of stickers)
            if subtitle_clips:
                self.logger.info(
                    f"Adding {len(subtitle_clips)} subtitle clip(s)."
                )
                layers.extend(subtitle_clips)

            # Watermark for multi-part stories
            watermark_clip: Optional[Any] = None
            if part_info and part_info.get("watermark_config"):
                wm_cfg: Dict[str, Any] = part_info["watermark_config"]
                watermark_clip = self._create_watermark(
                    text=wm_cfg.get("text", ""),
                    config=wm_cfg,
                    duration=audio_duration,
                    frame_size=(frame_w, frame_h),
                )
                clips_to_close.append(watermark_clip)
                layers.append(watermark_clip)
                self.logger.info(
                    f"Watermark added: '{wm_cfg.get('text', '')}'"
                )

            # 9. Composite all layers
            self.logger.info("Compositing final video …")
            final_clip: CompositeVideoClip = CompositeVideoClip(
                layers, size=(frame_w, frame_h)
            ).set_duration(audio_duration)
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

            threads: int = int(self.video_cfg.get("threads", 1))

            self.logger.info(
                f"Encoding → {output_path.name} "
                f"({codec}, {fps}fps, preset={preset}, "
                f"v_bitrate={bitrate}, a_bitrate={audio_bitrate}, threads={threads})"
            )

            final_clip.write_videofile(
                str(output_path),
                codec=codec,
                audio_codec=audio_codec,
                fps=fps,
                preset=preset,
                bitrate=bitrate,
                audio_bitrate=audio_bitrate,
                threads=threads,
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

    def _mix_background_audio(
        self,
        narration_path: Path,
        word_timestamps: Optional[List[Dict[str, Any]]] = None,
    ) -> Path:
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

        # Mix the background music underneath (either dynamically ducked or constant volume)
        if len(bg_music_clip) < narration_dur_ms:
            loops_needed = (narration_dur_ms // len(bg_music_clip)) + 1
            bg_music_loop = bg_music_clip * loops_needed
        else:
            bg_music_loop = bg_music_clip
        bg_music_final = bg_music_loop[:narration_dur_ms]

        ducking_enabled = self.config.get("pipeline", {}).get("audio_ducking", True)
        if ducking_enabled and word_timestamps:
            self.logger.info("Applying dynamic smart audio ducking around speaker pauses...")
            
            # segment list of gaps/non-gaps in ms
            segments = []
            last_t = 0
            gap_threshold_ms = 350  # gaps > 350ms get boosted
            
            # Sort words by start time
            words = sorted(word_timestamps, key=lambda w: w["start"])
            
            for w in words:
                start_ms = int(w["start"] * 1000)
                end_ms = int(w["end"] * 1000)
                
                # Gap before this word
                if start_ms > last_t:
                    segments.append((last_t, start_ms, True))
                
                # The word itself
                segments.append((start_ms, end_ms, False))
                last_t = end_ms
                
            # Final gap
            if narration_dur_ms > last_t:
                segments.append((last_t, narration_dur_ms, True))
                
            # Build ducked background music track
            bg_track = AudioSegment.silent(duration=0)
            
            # normal ducked volume (-26 dB, roughly 5% volume)
            bg_ducked = bg_music_final - 26.0
            # louder volume during pauses (-16 dB, roughly 16% volume)
            bg_boosted = bg_music_final - 16.0
            
            for start, end, is_gap in segments:
                dur = end - start
                if dur <= 0:
                    continue
                    
                if is_gap and dur > gap_threshold_ms:
                    # Louder gap with fades
                    fade_in_len = min(150, dur // 3)
                    fade_out_len = min(150, dur // 3)
                    slice_boosted = bg_boosted[start:end].fade_in(fade_in_len).fade_out(fade_out_len)
                    bg_track += slice_boosted
                else:
                    # Normal ducked segment
                    slice_ducked = bg_ducked[start:end]
                    bg_track += slice_ducked
            
            bg_final_track = bg_track[:narration_dur_ms]
        else:
            self.logger.info("Mixing constant background music (volume = 7%)...")
            bg_final_track = bg_music_final - 23.1

        mixed = mixed.overlay(bg_final_track, position=0)

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
        from pathlib import Path
        font_file = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "ArialBold.ttf"
        try:
            if font_file.exists():
                pil_font = ImageFont.truetype(str(font_file), font_size)
            else:
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

    def _create_speaker_stickers(
        self,
        dialogue_timings: List[Dict[str, Any]],
        frame_size: Tuple[int, int],
    ) -> List[Any]:
        """Create animated speaker avatar sticker clips.

        Args:
            dialogue_timings: List of dicts containing 'start', 'end', 'speaker', and 'text'.
            frame_size: (width, height) of the video frame.

        Returns:
            List of timed, positioned, animated ImageClip instances.
        """
        from PIL import Image
        import numpy as np
        from moviepy.editor import ImageClip, vfx
        
        conv_cfg = self.config.get("conversational", {})
        
        # Load parameters
        male_sticker_path_str = conv_cfg.get("male_sticker_path", "assets/stickers/male_neutral.png")
        female_sticker_path_str = conv_cfg.get("female_sticker_path", "assets/stickers/female_neutral.png")
        sticker_size = int(conv_cfg.get("sticker_size", 450))
        sticker_y = float(conv_cfg.get("sticker_y_position", 0.58))
        
        male_base_path = resolve_path(male_sticker_path_str)
        female_base_path = resolve_path(female_sticker_path_str)
        
        # Graceful fallback (fail-silent) if base neutral assets are missing
        if not male_base_path.exists() or not female_base_path.exists():
            self.logger.warning(
                f"Speaker sticker base assets not found at '{male_base_path}' or '{female_base_path}'. "
                "Skipping sticker overlays (fail-silent)."
            )
            return []
            
        # Extract prefixes from configured paths dynamically (e.g. "chibi_male_neutral" -> "chibi_male")
        male_prefix = male_base_path.stem[:-8] if male_base_path.stem.endswith("_neutral") else male_base_path.stem
        female_prefix = female_base_path.stem[:-8] if female_base_path.stem.endswith("_neutral") else female_base_path.stem

        frame_w, frame_h = frame_size
        y_pos = int(frame_h * sticker_y)
        
        sticker_clips = []
        for turn in dialogue_timings:
            speaker = str(turn.get("speaker", "MALE")).strip().upper()
            text = turn.get("text", "")
            start = turn.get("start", 0.0)
            end = turn.get("end", 0.0)
            duration = end - start
            
            if duration <= 0:
                continue
                
            if speaker == "OUTRO":
                continue
                
            # Detect emotion from turn dialogue (fallback to keyword-based detection if not provided by LLM or invalid)
            emotion = str(turn.get("emotion", "")).strip().lower()
            
            # Map synonymous/related emotions to our 15 supported sticker files
            synonyms = {
                "defensive": "angry",
                "furious": "angry",
                "outraged": "angry",
                "frustrated": "angry",
                "annoyed": "angry",
                "concerned": "worried",
                "anxious": "worried",
                "scared": "stressed",
                "fear": "stressed",
                "sad": "crying",
                "hurt": "crying",
                "depressed": "crying",
                "alarmed": "surprised",
                "shocked": "surprised",
                "calm": "talking",
                "excited": "happy",
                "glad": "happy",
            }
            if emotion in synonyms:
                emotion = synonyms[emotion]
                
            if emotion not in (
                "neutral", "happy", "angry", "crying", "surprised",
                "thinking", "explaining", "worried", "sighing", "thumbsup",
                "talking", "waving", "stressed", "lovestruck", "sleeping"
            ):
                emotion = self._detect_emotion(text)
            
            # Map path based on speaker and detected emotion
            if speaker in ("CHILD_MALE", "CHIBI_MALE"):
                gender_prefix = "chibi_male"
            elif speaker in ("CHILD_FEMALE", "CHIBI_FEMALE"):
                gender_prefix = "chibi_female"
            elif speaker == "OLD_MALE":
                gender_prefix = "old_male"
            elif speaker == "OLD_FEMALE":
                gender_prefix = "old_female"
            else:
                gender_prefix = male_prefix if speaker == "MALE" else female_prefix
            
            path = resolve_path(f"assets/stickers/{gender_prefix}_{emotion}.png")
            
            # Fall back to base path (neutral) if specific emotion asset is missing
            if not path.exists():
                if speaker in ("CHILD_MALE", "CHIBI_MALE"):
                    path = resolve_path("assets/stickers/chibi_male_neutral.png")
                elif speaker in ("CHILD_FEMALE", "CHIBI_FEMALE"):
                    path = resolve_path("assets/stickers/chibi_female_neutral.png")
                elif speaker == "OLD_MALE":
                    path = resolve_path("assets/stickers/old_male_neutral.png")
                elif speaker == "OLD_FEMALE":
                    path = resolve_path("assets/stickers/old_female_neutral.png")
                else:
                    path = male_base_path if speaker == "MALE" else female_base_path
            
            try:
                # Load with PIL to resize and handle alpha cleanly
                img = Image.open(str(path)).convert("RGBA")
                img_w, img_h = img.size
                
                scale_factor = sticker_size / max(img_w, img_h)
                target_w = int(img_w * scale_factor)
                target_h = int(img_h * scale_factor)
                
                img_resized = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
                
                # Pad the image on all sides to make room for the thick white stroke outline
                from PIL import ImageFilter
                padding = 20
                new_w = target_w + padding * 2
                new_h = target_h + padding * 2
                
                padded_img = Image.new("RGBA", (new_w, new_h), (0, 0, 0, 0))
                padded_img.paste(img_resized, (padding, padding))
                
                # Split channels to clean up background alpha noise & black edge bleed
                r_ch, g_ch, b_ch, a_ch = padded_img.split()
                a_np = np.array(a_ch)
                # Clean up low-opacity compression artifacts and binarize for crisp edges
                a_np[a_np < 120] = 0
                a_np[a_np >= 120] = 255
                
                a_clean = Image.fromarray(a_np)
                
                # Create white outline by dilating the cleaned alpha channel smoothly using circular offsets
                from PIL import ImageChops
                import math
                dilated_a = a_clean.copy()
                stroke_radius = 15
                num_steps = 24
                for i in range(num_steps):
                    angle = 2 * math.pi * i / num_steps
                    dx = int(round(stroke_radius * math.cos(angle)))
                    dy = int(round(stroke_radius * math.sin(angle)))
                    offset_a = ImageChops.offset(a_clean, dx, dy)
                    dilated_a = ImageChops.lighter(dilated_a, offset_a)
                
                white_canvas = Image.new("RGBA", (new_w, new_h), (255, 255, 255, 255))
                outline = Image.merge("RGBA", (white_canvas.split()[0], white_canvas.split()[1], white_canvas.split()[2], dilated_a))
                
                # Merge outline and cleaned character image
                char_clean = Image.merge("RGBA", (r_ch, g_ch, b_ch, a_clean))
                sticker_final = Image.alpha_composite(outline, char_clean)
                
                # Convert back to numpy array for MoviePy
                img_np = np.array(sticker_final)
                rgb_np = img_np[:, :, :3]
                alpha_np = img_np[:, :, 3] / 255.0
                
                clip = ImageClip(rgb_np)
                mask_clip = ImageClip(alpha_np, ismask=True)
                clip = clip.set_mask(mask_clip)
                
                # Update target dimensions to padded dimensions so position calculations center correctly
                target_w, target_h = new_w, new_h
                
                # Determine horizontal side: male/boy on left, female/girl on right
                is_male = speaker in ("MALE", "CHILD_MALE", "CHIBI_MALE", "OLD_MALE")
                center_x_pct = 0.22 if is_male else 0.78
                
                # Factory function to capture position parameters by value to avoid loop closure scope bugs
                def make_sticker_pos(cx, tw, th, yp):
                    def sticker_pos(t):
                        x = (frame_w * cx) - (tw / 2)
                        y = yp - (th / 2)
                        return (x, y)
                    return sticker_pos
                
                pos_func = make_sticker_pos(center_x_pct, target_w, target_h, y_pos)
                
                clip = (
                    clip
                    .set_start(start)
                    .set_duration(duration)
                    .set_position(pos_func)
                )
                
                sticker_clips.append(clip)
            except Exception as e:
                self.logger.warning(f"Failed to create sticker clip for turn starting at {start}s ({speaker}, {emotion}): {e}")
                continue
                
        return sticker_clips

    def _detect_emotion(self, text: str) -> str:
        """Detect the speaker's facial expression emotion from turn text.

        Args:
            text: Dialogue string.

        Returns:
            Emotion string mapping to files (e.g. 'angry', 'sad', 'crying', etc.)
        """
        text_lower = text.lower()
        
        # Check angry
        if any(w in text_lower for w in ["angry", "mad", "furious", "hate", "yell", "scream", "argue", "shut up", "stupid", "idiot", "annoyed"]):
            return "angry"
        # Check crying/sad
        if any(w in text_lower for w in ["sad", "cry", "tear", "depressed", "apologize", "sorry", "hurt", "pain", "miss", "weep"]):
            return "crying"
        # Check surprised
        if any(w in text_lower for w in ["surprise", "shock", "what", "omg", "suddenly", "believe", "insane", "crazy", "gasp"]):
            return "surprised"
        # Check worried/stressed
        if any(w in text_lower for w in ["worry", "stressed", "stress", "anxious", "scared", "fear", "nervous", "tight"]):
            return "stressed"
        # Check lovestruck
        if any(w in text_lower for w in ["love", "heart", "darling", "sweetheart", "babe", "date", "marry", "wedding"]):
            return "lovestruck"
        # Check happy
        if any(w in text_lower for w in ["happy", "laugh", "excited", "glad", "smile", "awesome", "great", "nice", "fun"]):
            return "happy"
        # Check thinking/explaining
        if "?" in text_lower:
            return "thinking"
        
        # Default to talking
        return "talking"
