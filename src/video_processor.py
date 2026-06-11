"""
RedditDaily-Bot — Video Processor
==================================
Gameplay downloading via yt-dlp and perceptual-hash destruction via FFmpeg.

This module provides two classes:

- **GameplayDownloader**: Downloads background gameplay footage from YouTube
  (or other yt-dlp-supported sites) and manages the local gameplay library.
- **HashDestructionPipeline**: Applies a deterministic FFmpeg filter chain that
  subtly alters every frame so that the output video has a unique perceptual
  hash, defeating automated duplicate-content detection on social platforms.
"""

import json
import random
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from src.utils import (
    BotLogger,
    check_ffmpeg,
    check_yt_dlp,
    resolve_path,
)


# ---------------------------------------------------------------------------
# Gameplay Downloader
# ---------------------------------------------------------------------------
class GameplayDownloader:
    """Downloads and manages background gameplay footage via yt-dlp.

    Attributes:
        config: Full application configuration dictionary.
        logger: BotLogger instance for structured logging.
        gameplay_dir: Local directory where gameplay clips are stored.
    """

    def __init__(self, config: dict, logger: BotLogger) -> None:
        """Initialise the downloader.

        Args:
            config: Parsed application config (expects ``video.gameplay_dir``
                and ``video.gameplay_sources``).
            logger: Shared BotLogger instance.
        """
        self.config: dict = config
        self.logger: BotLogger = logger

        gameplay_dir_cfg: str = config.get("video", {}).get(
            "gameplay_dir", "assets/gameplay"
        )
        self.gameplay_dir: Path = resolve_path(gameplay_dir_cfg, create=True)
        self.logger.debug(f"GameplayDownloader initialised — dir: {self.gameplay_dir}")

    # ---- public API --------------------------------------------------------

    def download(self, url: str, label: str) -> Path:
        """Download a single gameplay video via *yt-dlp*.

        Args:
            url: Video URL (YouTube, Twitch VOD, etc.).
            label: Filesystem-safe label used as the output filename stem.

        Returns:
            Path to the downloaded ``.mp4`` file.

        Raises:
            FileNotFoundError: If *yt-dlp* is not installed.
            RuntimeError: If the download subprocess fails.
        """
        output_path: Path = self.gameplay_dir / f"{label}.mp4"

        # Skip if already downloaded
        if output_path.exists() and output_path.stat().st_size > 0:
            self.logger.info(
                f"Gameplay already downloaded — skipping: {output_path.name}"
            )
            return output_path

        # Pre-flight check
        if not check_yt_dlp():
            raise FileNotFoundError(
                "yt-dlp is not installed or not on PATH. "
                "Install with: pip install yt-dlp"
            )

        cmd: List[str] = [
            "yt-dlp",
            "-f",
            "bestvideo[height>=1920]+bestaudio/best",
            "--merge-output-format",
            "mp4",
            "-o",
            str(output_path),
            url,
        ]

        self.logger.info(f"Downloading gameplay '{label}' from {url} …")
        self.logger.debug(f"yt-dlp command: {' '.join(cmd)}")

        try:
            result: subprocess.CompletedProcess = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30-min hard limit
            )

            if result.returncode != 0:
                stderr_tail: str = (result.stderr or "")[-500:]
                self.logger.error(
                    f"yt-dlp failed for '{label}' (rc={result.returncode}): "
                    f"{stderr_tail}"
                )
                raise RuntimeError(
                    f"yt-dlp download failed for '{label}': {stderr_tail}"
                )

            self.logger.info(f"Download complete: {output_path.name}")
            return output_path

        except subprocess.TimeoutExpired:
            self.logger.error(
                f"yt-dlp timed out after 30 minutes for '{label}'."
            )
            raise RuntimeError(
                f"yt-dlp timed out downloading '{label}' from {url}"
            )
        except FileNotFoundError:
            # subprocess raises this when the executable is not found
            raise FileNotFoundError(
                "yt-dlp executable not found. Install with: pip install yt-dlp"
            )

    def get_random_gameplay(self) -> Path:
        """Return a random ``.mp4`` file from the gameplay directory.

        Returns:
            Path to a randomly selected gameplay clip.

        Raises:
            FileNotFoundError: If no ``.mp4`` files exist in the gameplay
                directory.
        """
        mp4_files: List[Path] = sorted(self.gameplay_dir.glob("*.mp4"))

        if not mp4_files:
            raise FileNotFoundError(
                f"No gameplay videos found in {self.gameplay_dir}. "
                "Download gameplay footage first or place .mp4 files there."
            )

        chosen: Path = random.choice(mp4_files)
        self.logger.info(
            f"Selected random gameplay: {chosen.name} "
            f"(from {len(mp4_files)} available)"
        )
        return chosen

    def download_all(self) -> List[Path]:
        """Download every source listed in ``video.gameplay_sources``.

        Sources whose URL contains the substring ``PLACEHOLDER`` are silently
        skipped with a warning.

        Returns:
            List of Paths to successfully downloaded files.
        """
        sources: List[Dict[str, str]] = (
            self.config.get("video", {}).get("gameplay_sources", [])
        )

        if not sources:
            self.logger.warning(
                "No gameplay_sources defined in config — nothing to download."
            )
            return []

        downloaded: List[Path] = []

        for source in sources:
            url: str = source.get("url", "")
            label: str = source.get("label", "unknown")

            if "PLACEHOLDER" in url.upper():
                self.logger.warning(
                    f"Skipping placeholder URL for '{label}' — "
                    f"update config with a real URL."
                )
                continue

            try:
                path: Path = self.download(url, label)
                downloaded.append(path)
            except Exception as exc:
                self.logger.error(
                    f"Failed to download '{label}': {exc}"
                )

        self.logger.info(
            f"Gameplay download complete: {len(downloaded)}/{len(sources)} "
            f"sources succeeded."
        )
        return downloaded


# ---------------------------------------------------------------------------
# Hash Destruction Pipeline
# ---------------------------------------------------------------------------
class HashDestructionPipeline:
    """Applies an FFmpeg filter chain that destroys perceptual hashes.

    The pipeline performs subtle but deterministic audio-visual
    transformations — spatial rescaling, chroma shift, luma adjustment,
    temporal noise injection, and audio tempo warping — so that each
    output is unique to content-ID systems while remaining visually
    indistinguishable to viewers.

    Attributes:
        config: Full application configuration dictionary.
        logger: BotLogger instance for structured logging.
        video_cfg: Video sub-config (resolution, codec, etc.).
        filter_cfg: Filter sub-config (scale, hue, noise, etc.).
    """

    def __init__(self, config: dict, logger: BotLogger) -> None:
        """Initialise the pipeline.

        Args:
            config: Parsed application config (expects ``video`` and
                ``pipeline`` sections).
            logger: Shared BotLogger instance.
        """
        self.config: dict = config
        self.logger: BotLogger = logger
        self.video_cfg: dict = config.get("video", {})
        self.filter_cfg: dict = self.video_cfg.get("filters", {})

    # ---- public API --------------------------------------------------------

    def process(
        self,
        input_path: Path,
        target_duration: float,
        output_path: Path,
    ) -> Path:
        """Run the full hash-destruction pipeline on a gameplay clip.

        A random segment of *target_duration* seconds is extracted from
        the source, then every frame is processed through the filter chain.

        Args:
            input_path: Path to the source gameplay ``.mp4``.
            target_duration: Desired output length in seconds.
            output_path: Where to write the processed result.

        Returns:
            *output_path* on success.

        Raises:
            FileNotFoundError: If FFmpeg/ffprobe is missing or *input_path*
                does not exist.
            RuntimeError: If FFmpeg exits with a non-zero return code.
        """
        if not check_ffmpeg():
            raise FileNotFoundError(
                "ffmpeg is not installed or not on PATH. "
                "Download from https://ffmpeg.org/download.html"
            )

        if not input_path.exists():
            raise FileNotFoundError(f"Source video not found: {input_path}")

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Determine a random start point
        source_duration: float = self._get_video_duration(input_path)

        if source_duration <= target_duration:
            random_start = 0.0
            self.logger.warning(
                f"Source ({source_duration:.1f}s) is shorter or equal to "
                f"target ({target_duration:.1f}s) — starting from 0."
            )
        else:
            random_start = float(
                random.randint(0, int(source_duration - target_duration))
            )

        # Build the FFmpeg command
        vf_chain: str = self._build_filter_chain()
        af_chain: str = self._build_audio_filter()

        codec: str = self.video_cfg.get("codec", "libx264")
        
        # Ensure preset is 'medium' or 'fast' for proper H.264 compression standards
        preset: str = self.video_cfg.get("preset", "medium")
        if preset not in ["medium", "fast"]:
            self.logger.warning(f"Video preset '{preset}' is not optimized for Reels. Forcing 'medium'.")
            preset = "medium"

        crf: int = self.video_cfg.get("crf", 18)
        audio_codec: str = self.video_cfg.get("audio_codec", "aac")
        fps: int = self.video_cfg.get("fps", 30)
        pixel_format: str = self.video_cfg.get("pixel_format", "yuv420p")

        cmd: List[str] = [
            "ffmpeg",
            "-y",
            "-ss", f"{random_start:.3f}",
            "-i", str(input_path),
            "-t", f"{target_duration:.3f}",
            "-vf", vf_chain,
            "-af", af_chain,
            "-c:v", codec,
            "-preset", preset,
            "-crf", str(crf),
            "-c:a", audio_codec,
            "-b:a", "192k",
            "-r", str(fps),
            "-pix_fmt", pixel_format,
            "-map_metadata", "-1",
            "-map_chapters", "-1",
            "-movflags", "+faststart",
            str(output_path),
        ]

        self.logger.info(
            f"Processing gameplay: {input_path.name} → {output_path.name}"
        )
        self.logger.debug(f"FFmpeg start={random_start:.1f}s, duration={target_duration:.1f}s")
        self.logger.debug(f"Video filter chain: {vf_chain}")
        self.logger.debug(f"Audio filter chain: {af_chain}")
        self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")

        try:
            result: subprocess.CompletedProcess = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10-min hard limit
            )

            if result.returncode != 0:
                stderr_tail: str = (result.stderr or "")[-1000:]
                self.logger.error(
                    f"FFmpeg failed (rc={result.returncode}): {stderr_tail}"
                )
                raise RuntimeError(
                    f"FFmpeg processing failed: {stderr_tail}"
                )

            self.logger.info(
                f"Hash-destruction complete: {output_path.name} "
                f"({output_path.stat().st_size / (1024 * 1024):.1f} MB)"
            )
            return output_path

        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg timed out after 10 minutes.")
            raise RuntimeError("FFmpeg processing timed out.")

    # ---- private helpers ---------------------------------------------------

    def _get_video_duration(self, path: Path) -> float:
        """Retrieve video duration in seconds via *ffprobe*.

        Args:
            path: Path to a video file.

        Returns:
            Duration in seconds as a float.

        Raises:
            FileNotFoundError: If *ffprobe* is not available.
            RuntimeError: If *ffprobe* fails or the output cannot be parsed.
        """
        cmd: List[str] = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(path),
        ]

        self.logger.debug(f"Probing duration: {path.name}")

        try:
            result: subprocess.CompletedProcess = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                stderr_tail: str = (result.stderr or "")[-500:]
                raise RuntimeError(
                    f"ffprobe failed for {path.name}: {stderr_tail}"
                )

            probe_data: dict = json.loads(result.stdout)
            duration_str: Optional[str] = (
                probe_data.get("format", {}).get("duration")
            )

            if duration_str is None:
                raise RuntimeError(
                    f"ffprobe returned no duration for {path.name}. "
                    f"Output: {result.stdout[:300]}"
                )

            duration: float = float(duration_str)
            self.logger.debug(f"Duration of {path.name}: {duration:.2f}s")
            return duration

        except FileNotFoundError:
            raise FileNotFoundError(
                "ffprobe is not installed or not on PATH. "
                "It ships with ffmpeg — ensure ffmpeg is properly installed."
            )
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse ffprobe JSON for {path.name}: {exc}"
            )

    def _build_filter_chain(self) -> str:
        """Construct the video filter (``-vf``) string from config.

        Returns:
            A comma-separated FFmpeg video filter string, e.g.::

                mpdecimate,setpts=N/FRAME_RATE/TB,scale=1200:2130,...
        """
        scale_w: int = self.filter_cfg.get("scale_width", 1200)
        scale_h: int = self.filter_cfg.get("scale_height", 2130)
        crop_w: int = self.filter_cfg.get("crop_width", 1080)
        crop_h: int = self.filter_cfg.get("crop_height", 1920)
        hue_h: float = self.filter_cfg.get("hue_shift", 0.5)
        sat: float = self.filter_cfg.get("saturation_mult", 1.02)
        bright: float = self.filter_cfg.get("brightness_shift", 0.01)
        noise_s: int = self.filter_cfg.get("noise_strength", 2)
        noise_f: str = self.filter_cfg.get("noise_flags", "t")

        filters: List[str] = [
            "mpdecimate",
            "setpts=N/FRAME_RATE/TB",
            f"scale={scale_w}:{scale_h}",
            f"crop={crop_w}:{crop_h}",
            f"hue=h={hue_h}:s={sat}:b={bright}",
            f"noise=alls={noise_s}:allf={noise_f}",
        ]

        return ",".join(filters)

    def _build_audio_filter(self) -> str:
        """Construct the audio filter (``-af``) string from config.

        Returns:
            FFmpeg audio filter string, e.g. ``atempo=1.01``.
        """
        tempo: float = self.filter_cfg.get("audio_tempo", 1.01)
        return f"atempo={tempo}"
