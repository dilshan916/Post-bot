#!/usr/bin/env python3
"""
RedditDaily-Bot — Main Orchestrator
=====================================
CLI entry point that drives the complete pipeline:
  1. Scrape Reddit stories  →  clean & split scripts
  2. Synthesize TTS audio   →  apply acoustic cadence
  3. Extract word timestamps →  render double-pass subtitles
  4. Process gameplay video  →  composite final 9:16 Reel

Usage
-----
  # Full pipeline (single story)
  python main.py

  # Batch mode
  python main.py --batch 3

  # Run a single component
  python main.py --component scraper
  python main.py --component tts --test-phrase "Testing TTS"
  python main.py --component whisper --test-audio path/to/audio.wav
  python main.py --component video --test-clip path/to/clip.mp4

  # Custom config
  python main.py --config my_config.yaml

  # Dry-run (validates config, no actions)
  python main.py --dry-run
"""

import argparse
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    BotLogger,
    load_config,
    resolve_path,
    ensure_dirs,
    save_json,
    load_json,
    estimate_duration_sec,
    format_duration,
    sanitize_filename,
    timestamp_str,
    check_ffmpeg,
    check_yt_dlp,
)
from src.reddit_scraper import RedditScraper
from src.script_rewriter import ScriptRewriter
from src.smart_splitter import SmartSplitter
from src.tts_engine import TTSEngine
from src.acoustic_cadence import AcousticCadenceProcessor
from src.whisper_timestamps import TimestampExtractor
from src.subtitle_renderer import SubtitleRenderer
from src.video_processor import GameplayDownloader, HashDestructionPipeline
from src.video_compositor import VideoCompositor
from src.screenshot_manager import ScreenshotManager


# ===================================================================
# Pipeline Orchestrator
# ===================================================================
class RedditDailyBot:
    """End-to-end pipeline orchestrator for generating Reddit story Reels."""

    def __init__(self, config: Dict[str, Any], logger: BotLogger):
        self.config = config
        self.logger = logger

        # Resolve & create working directories
        self.temp_dir = resolve_path(config["pipeline"]["temp_dir"], create=True)
        self.output_dir = resolve_path(config["video"]["output_dir"], create=True)
        self.log_dir = resolve_path(config["pipeline"]["log_dir"], create=True)

        self.logger.info("RedditDailyBot initialised")
        self.logger.info(f"  Temp dir  : {self.temp_dir}")
        self.logger.info(f"  Output dir: {self.output_dir}")

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run(self, batch_size: int = 1) -> List[Path]:
        """Run the full pipeline for *batch_size* stories.

        Returns
        -------
        list[Path]
            Paths to all generated Reel files.
        """
        outputs: List[Path] = []
        self.logger.info(f"Starting pipeline — batch size {batch_size}")

        # ── Step 0: Pre-flight checks ────────────────────────────────
        self._preflight_checks()

        # ── Step 1: Scrape Reddit stories ────────────────────────────
        self.logger.info("═" * 60)
        self.logger.info("STEP 1 / 5  ►  Scraping Reddit stories")
        self.logger.info("═" * 60)
        scraper = RedditScraper(self.config, self.logger)
        rewriter = ScriptRewriter(self.config, self.logger)
        splitter = SmartSplitter(self.config, self.logger)

        # Fetch more candidates per subreddit to bypass sticky posts and history matches
        fetch_limit = max(batch_size * 5, 25)
        stories = scraper.scrape_stories(limit=fetch_limit)  # fetch extra for filtering
        if not stories:
            self.logger.error("No stories found. Check subreddit config & Reddit credentials.")
            return outputs

        self.logger.info(f"Fetched {len(stories)} candidate stories")

        # Process up to batch_size stories
        processed_count = 0
        for story in stories:
            if processed_count >= batch_size:
                break

            try:
                result = self._process_single_story(
                    story, scraper, rewriter, splitter
                )
                if result:
                    outputs.extend(result)
                    processed_count += 1
            except Exception as exc:
                self.logger.error(
                    f"Failed to process story '{story.get('title', '?')[:50]}': {exc}"
                )
                self.logger.debug(traceback.format_exc())
                continue

        # ── Cleanup ──────────────────────────────────────────────────
        if not self.config["pipeline"].get("keep_temp_files", False):
            self._cleanup_temp()

        self.logger.info("═" * 60)
        self.logger.info(f"Pipeline complete — {len(outputs)} Reel(s) generated")
        for p in outputs:
            self.logger.info(f"  ✓ {p}")
        self.logger.info("═" * 60)
        return outputs

    # ------------------------------------------------------------------
    # Single-story processing
    # ------------------------------------------------------------------
    def _process_single_story(
        self,
        story: Dict[str, Any],
        scraper: RedditScraper,
        rewriter: ScriptRewriter,
        splitter: SmartSplitter,
    ) -> List[Path]:
        """Process one Reddit story through the full pipeline.

        Returns list of output paths (one per part if story is split).
        """
        title_short = story.get("title", "untitled")[:60]
        subreddit = story.get("subreddit", "unknown")
        self.logger.info(f"Processing: r/{subreddit} — \"{title_short}\"")
        if story.get("virality_reason"):
            self.logger.info(f"  Virality Reason: {story['virality_reason']}")

        # ── 1a: Rewrite script ───────────────────────────────────────
        script = rewriter.rewrite(story)
        self.logger.info(
            f"  Script: {len(script.split())} words, "
            f"est. {format_duration(estimate_duration_sec(script))}"
        )

        # ── 1b: Smart split ──────────────────────────────────────────
        parts = splitter.split(script)
        self.logger.info(f"  Split into {len(parts)} part(s)")

        # Mark story as scraped
        scraper.mark_scraped(story["id"])

        # ── Process each part ────────────────────────────────────────
        output_paths: List[Path] = []
        for part in parts:
            try:
                path = self._render_part(part, story)
                output_paths.append(path)
            except Exception as exc:
                self.logger.error(
                    f"  Failed part {part['part_number']}/{part['total_parts']}: {exc}"
                )
                self.logger.debug(traceback.format_exc())

        return output_paths

    def _render_part(self, part: Dict[str, Any], story: Dict[str, Any]) -> Path:
        """Render a single part (or the whole story if single-part) into a Reel."""

        part_label = f"Part {part['part_number']}/{part['total_parts']}"
        self.logger.info(f"  ── Rendering {part_label} ──")

        # ── Step 2: TTS Synthesis ────────────────────────────────────
        self.logger.info("  STEP 2 ►  Synthesising TTS audio")
        tts = TTSEngine(self.config, self.logger)
        cadence = AcousticCadenceProcessor(self.config, self.logger)

        # Determine narrator gender and select voice dynamically
        narrator_gender = (story or {}).get("narrator_gender")
        config_voice = self.config.get("tts", {}).get("voice", "en-US-ChristopherNeural")
        
        if narrator_gender == "MALE":
            voice = "en-US-ChristopherNeural"
            self.logger.info(f"    Dynamic Voice Selector: Narrator detected as MALE -> selected '{voice}'")
        elif narrator_gender == "FEMALE":
            voice = "en-US-JennyNeural"
            self.logger.info(f"    Dynamic Voice Selector: Narrator detected as FEMALE -> selected '{voice}'")
        else:
            voice = config_voice
            self.logger.info(f"    Dynamic Voice Selector: Narrator gender undetected -> falling back to config voice '{voice}'")

        # Append tail phrase for vocal decay hack
        script_with_tail = cadence.prepare_script_with_tail(part["script_text"])
        audio_path = tts.synthesize(script_with_tail, voice=voice)
        self.logger.info(f"    TTS audio saved: {audio_path.name}")

        # ── Step 3: Whisper Timestamps ───────────────────────────────
        self.logger.info("  STEP 3 ►  Extracting word-level timestamps")
        extractor = TimestampExtractor(self.config, self.logger)
        ts_data = extractor.extract(audio_path)
        self.logger.info(
            f"    Engine: {ts_data['engine']} — "
            f"{len(ts_data['words'])} words, "
            f"{len(ts_data['sentences'])} sentences"
        )

        # Save timestamps for debugging
        ts_json_path = self.temp_dir / f"timestamps_p{part['part_number']}.json"
        extractor.save_timestamps(ts_data, ts_json_path)

        # ── Step 2b: Acoustic Cadence (post-Whisper) ─────────────────
        self.logger.info("  STEP 2b ►  Applying acoustic cadence processing")
        processed_audio = cadence.process_audio(audio_path, ts_data["words"])
        self.logger.info(f"    Processed audio: {processed_audio.name}")

        # Re-extract timestamps on the processed audio (without tail)
        self.logger.info("  STEP 3b ►  Re-extracting timestamps on processed audio")
        ts_data_final = extractor.extract(processed_audio)
        self.logger.info(
            f"    Final: {len(ts_data_final['words'])} words, "
            f"{len(ts_data_final['sentences'])} sentences"
        )

        # ── Step 3c: Subtitle Rendering ──────────────────────────────
        self.logger.info("  STEP 3c ►  Rendering double-pass subtitles")
        sub_renderer = SubtitleRenderer(self.config, self.logger)
        frame_size = (
            self.config["video"]["resolution"]["width"],
            self.config["video"]["resolution"]["height"],
        )
        subtitle_clips = sub_renderer.render(ts_data_final, frame_size)
        self.logger.info(f"    Generated {len(subtitle_clips)} subtitle clips")

        # ── Step 3d: Reddit Screenshot Hook ──────────────────────────
        screenshot_cfg = self.config.get("screenshot", {})
        if screenshot_cfg.get("enabled", True):
            self.logger.info("  STEP 3d ►  Capturing Reddit screenshot hook")
            try:
                ss_manager = ScreenshotManager(self.config, self.logger)
                hook_png = ss_manager.capture(story)
                hook_clip = ss_manager.create_hook_clip(
                    screenshot_path=hook_png,
                    frame_size=frame_size,
                )
                subtitle_clips.append(hook_clip)
                self.logger.info("    Hook overlay added (%.1fs)",
                                 screenshot_cfg.get("display_duration_sec", 3.5))
            except Exception as exc:
                self.logger.warning(
                    "  Screenshot hook failed (non-fatal): %s", exc
                )

        # ── Step 4: Video Processing ─────────────────────────────────
        self.logger.info("  STEP 4 ►  Processing gameplay background")
        downloader = GameplayDownloader(self.config, self.logger)
        hash_pipeline = HashDestructionPipeline(self.config, self.logger)

        # Ensure we have gameplay footage
        try:
            gameplay_source = downloader.get_random_gameplay()
        except FileNotFoundError:
            self.logger.info("    No gameplay files found locally. Attempting to download from configured sources...")
            downloader.download_all()
            gameplay_source = downloader.get_random_gameplay()
        self.logger.info(f"    Source gameplay: {gameplay_source.name}")

        # Get audio duration for matching gameplay length
        from pydub import AudioSegment

        audio_seg = AudioSegment.from_file(str(processed_audio))
        audio_duration_sec = len(audio_seg) / 1000.0

        # Process gameplay through hash destruction pipeline
        processed_gameplay = self.temp_dir / f"gameplay_processed_p{part['part_number']}.mp4"
        hash_pipeline.process(
            input_path=gameplay_source,
            target_duration=audio_duration_sec + 2.0,  # small buffer
            output_path=processed_gameplay,
        )
        self.logger.info(f"    Processed gameplay: {processed_gameplay.name}")

        # ── Step 5: Final Compositing ────────────────────────────────
        self.logger.info("  STEP 5 ►  Compositing final Reel")
        compositor = VideoCompositor(self.config, self.logger)

        part_info = None
        if part["total_parts"] > 1:
            part_info = {
                **part,
                "subreddit": story.get("subreddit", "reddit"),
                "title": story.get("title", "story"),
            }
        else:
            # Single-part: still pass metadata for filename generation
            part_info = {
                "subreddit": story.get("subreddit", "reddit"),
                "title": story.get("title", "story"),
            }

        final_path = compositor.compose(
            gameplay_path=processed_gameplay,
            audio_path=processed_audio,
            subtitle_clips=subtitle_clips,
            part_info=part_info,
        )
        self.logger.info(f"  ✓ Reel complete: {final_path}")
        return final_path

    # ------------------------------------------------------------------
    # Pre-flight validation
    # ------------------------------------------------------------------
    def _preflight_checks(self) -> None:
        """Validate tools, credentials, and directories before running."""
        self.logger.info("Running pre-flight checks...")

        # FFmpeg
        if not check_ffmpeg():
            self.logger.critical(
                "FFmpeg not found on PATH. Install from https://ffmpeg.org"
            )
            raise RuntimeError("FFmpeg is required but not found on PATH")
        self.logger.info("  ✓ FFmpeg found")

        # yt-dlp (warning only — gameplay might be pre-downloaded)
        if not check_yt_dlp():
            self.logger.warning(
                "yt-dlp not found on PATH. "
                "Gameplay downloads will fail. Pre-place .mp4 files in assets/gameplay/"
            )
        else:
            self.logger.info("  ✓ yt-dlp found")

        # Reddit config checks
        reddit_cfg = self.config.get("reddit", {})
        subreddits = reddit_cfg.get("subreddits", [])
        if not subreddits:
            self.logger.critical(
                "No subreddits configured in config.yaml under reddit.subreddits"
            )
            raise RuntimeError("Subreddits list is empty")
        self.logger.info(f"  ✓ Reddit scraper configured with {len(subreddits)} subreddit(s)")

        # Edge-TTS voice check
        tts_cfg = self.config.get("tts", {})
        tts_voice = tts_cfg.get("voice", "en-US-ChristopherNeural")
        self.logger.info(f"  ✓ Edge-TTS voice configured: {tts_voice}")

        # Gameplay directory
        gp_dir = resolve_path(self.config["video"]["gameplay_dir"])
        if gp_dir.exists():
            mp4s = list(gp_dir.glob("*.mp4"))
            if mp4s:
                self.logger.info(f"  ✓ {len(mp4s)} gameplay file(s) in {gp_dir}")
            else:
                self.logger.warning(
                    f"  No .mp4 files in {gp_dir}. "
                    "Will attempt yt-dlp download from configured URLs."
                )
        else:
            ensure_dirs(str(gp_dir))
            self.logger.warning(f"  Created empty gameplay dir: {gp_dir}")

        self.logger.info("Pre-flight checks passed ✓")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def _cleanup_temp(self) -> None:
        """Remove temporary files."""
        if self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.logger.info("Cleaned up temp directory")
            except OSError as exc:
                self.logger.warning(f"Could not clean temp dir: {exc}")


# ===================================================================
# Component Runners (for --component flag)
# ===================================================================
def run_scraper_only(config: Dict, logger: BotLogger, dry_run: bool = False) -> None:
    """Run only the Reddit scraper component."""
    logger.info("Running scraper component only")
    scraper = RedditScraper(config, logger)

    if dry_run:
        logger.info("Dry-run mode — validating config only")
        logger.info(f"  Subreddits: {config['reddit']['subreddits']}")
        logger.info(f"  Sort: {config['reddit']['sort']}")
        logger.info(f"  Min upvotes: {config['reddit']['min_upvotes']}")
        return

    stories = scraper.scrape_stories()
    logger.info(f"Found {len(stories)} stories")
    for s in stories[:5]:
        logger.info(
            f"  [{s['score']:>5}] r/{s['subreddit']} — {s['title'][:60]}"
        )


def run_tts_only(
    config: Dict, logger: BotLogger, test_phrase: Optional[str] = None
) -> None:
    """Run only the TTS component with a test phrase."""
    logger.info("Running TTS component only")
    tts = TTSEngine(config, logger)

    phrase = test_phrase or "This is a test of the RedditDaily Bot text to speech engine. It should sound natural, with good pacing."
    logger.info(f"Test phrase: {phrase[:80]}...")

    audio_path = tts.synthesize(phrase)
    logger.info(f"Audio saved to: {audio_path}")


def run_whisper_only(
    config: Dict, logger: BotLogger, test_audio: Optional[str] = None
) -> None:
    """Run only the Whisper timestamp component."""
    logger.info("Running Whisper component only")

    if not test_audio:
        logger.error("--test-audio path required for whisper component test")
        return

    audio_path = Path(test_audio)
    if not audio_path.exists():
        logger.error(f"Audio file not found: {audio_path}")
        return

    extractor = TimestampExtractor(config, logger)
    data = extractor.extract(audio_path)
    logger.info(f"Engine: {data['engine']}")
    logger.info(f"Words: {len(data['words'])}")
    logger.info(f"Sentences: {len(data['sentences'])}")

    # Print first few words
    for w in data["words"][:10]:
        logger.info(
            f"  [{w['start']:.2f} - {w['end']:.2f}] "
            f"{w['word']} (conf: {w['confidence']:.2f})"
        )

    # Save to temp
    out = resolve_path(config["pipeline"]["temp_dir"], create=True) / "test_timestamps.json"
    extractor.save_timestamps(data, out)
    logger.info(f"Timestamps saved to: {out}")


def run_video_only(
    config: Dict, logger: BotLogger, test_clip: Optional[str] = None
) -> None:
    """Run only the video processing component."""
    logger.info("Running video processing component only")

    if not test_clip:
        logger.error("--test-clip path required for video component test")
        return

    clip_path = Path(test_clip)
    if not clip_path.exists():
        logger.error(f"Clip not found: {clip_path}")
        return

    pipeline = HashDestructionPipeline(config, logger)
    out = resolve_path(config["pipeline"]["temp_dir"], create=True) / "test_processed.mp4"
    pipeline.process(clip_path, target_duration=30.0, output_path=out)
    logger.info(f"Processed clip saved to: {out}")


# ===================================================================
# CLI Entry Point
# ===================================================================
def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="RedditDailyBot",
        description=(
            "Automated Reddit Story Reel Generator — "
            "scrape, narrate, subtitle, and composite 9:16 vertical Reels"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          Full pipeline, 1 story
  python main.py --batch 5                Process 5 stories
  python main.py --dry-run                Validate config only
  python main.py --component scraper      Run scraper only
  python main.py --component tts --test-phrase "Hello world"
  python main.py --component whisper --test-audio audio.wav
  python main.py --component video --test-clip gameplay.mp4
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config.yaml (default: project root)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=None,
        help="Number of stories to process (overrides config)",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["scraper", "tts", "whisper", "video"],
        help="Run a single component in isolation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running the pipeline",
    )
    parser.add_argument(
        "--test-phrase",
        type=str,
        default=None,
        help="Test phrase for TTS component testing",
    )
    parser.add_argument(
        "--test-audio",
        type=str,
        default=None,
        help="Path to audio file for Whisper component testing",
    )
    parser.add_argument(
        "--test-clip",
        type=str,
        default=None,
        help="Path to video clip for video component testing",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging",
    )

    return parser


def main() -> int:
    """Main entry point."""
    parser = build_parser()
    args = parser.parse_args()

    # ── Load config ──────────────────────────────────────────────────
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[CRITICAL] {exc}", file=sys.stderr)
        return 1

    # ── Override batch size ──────────────────────────────────────────
    if args.batch is not None:
        config["pipeline"]["batch_size"] = args.batch

    # ── Set up logger ────────────────────────────────────────────────
    log_level = "DEBUG" if args.verbose else config["pipeline"].get("log_level", "INFO")
    log_dir = str(resolve_path(config["pipeline"]["log_dir"], create=True))
    logger = BotLogger(name="RedditDailyBot", log_dir=log_dir, level=log_level)

    logger.info("=" * 60)
    logger.info("  RedditDaily-Bot v1.0.0")
    logger.info("  Automated Reddit Story Reel Generator")
    logger.info("=" * 60)

    # ── Dry-run mode ─────────────────────────────────────────────────
    if args.dry_run:
        logger.info("DRY-RUN MODE — validating configuration")
        try:
            bot = RedditDailyBot(config, logger)
            bot._preflight_checks()
            logger.info("All checks passed. Ready to run.")
            return 0
        except RuntimeError as exc:
            logger.critical(f"Pre-flight failed: {exc}")
            return 1

    # ── Component mode ───────────────────────────────────────────────
    if args.component:
        try:
            if args.component == "scraper":
                run_scraper_only(config, logger, dry_run=args.dry_run)
            elif args.component == "tts":
                run_tts_only(config, logger, test_phrase=args.test_phrase)
            elif args.component == "whisper":
                run_whisper_only(config, logger, test_audio=args.test_audio)
            elif args.component == "video":
                run_video_only(config, logger, test_clip=args.test_clip)
            return 0
        except Exception as exc:
            logger.critical(f"Component '{args.component}' failed: {exc}")
            logger.debug(traceback.format_exc())
            return 1

    # ── Full pipeline ────────────────────────────────────────────────
    try:
        bot = RedditDailyBot(config, logger)
        batch_size = config["pipeline"].get("batch_size", 1)
        outputs = bot.run(batch_size=batch_size)

        if not outputs:
            logger.warning("No Reels were generated")
            return 1

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except RuntimeError as exc:
        logger.critical(f"Pipeline failed: {exc}")
        return 1
    except Exception as exc:
        logger.critical(f"Unexpected error: {exc}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
