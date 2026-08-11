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
from typing import Any, Dict, List, Optional, Tuple
from colorama import Fore, Style

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
    resolve_speaker_category,
)
from src.reddit_scraper import RedditScraper
from src.facebook_publisher import FacebookReelsPublisher

# Lazy/Safe imports for heavy media modules
try:
    from src.script_rewriter import ScriptRewriter
    from src.smart_splitter import SmartSplitter
    from src.tts_engine import TTSEngine
    from src.acoustic_cadence import AcousticCadenceProcessor
    from src.whisper_timestamps import TimestampExtractor
    from src.subtitle_renderer import SubtitleRenderer
    from src.video_processor import GameplayDownloader, HashDestructionPipeline
    from src.video_compositor import VideoCompositor
    from src.screenshot_manager import ScreenshotManager
except ImportError as _import_err:
    # Allow lightweight CLI tools (like --check-proxies) to run even if virtualenv is not activated
    pass
# ---------------------------------------------------------------------------
# Global Session Settings
# ---------------------------------------------------------------------------
pipeline_mode = "monologue"

class NoStoriesFoundError(Exception):
    """Raised when no qualifying stories are found on Reddit."""
    pass


def generate_countdown_sfx(duration_sec: float = 3.0) -> Any:
    """Generate a ticking sound effect countdown audio segment using pydub."""
    from pydub import AudioSegment
    from pydub.generators import Sine
    
    segment = AudioSegment.silent(duration=int(duration_sec * 1000))
    # Create high-pitch short tick
    tick = Sine(1200).to_audio_segment(duration=50).fade_out(20).apply_gain(-12)
    # Create final low-pitch pop/beep
    pop = Sine(800).to_audio_segment(duration=150).fade_out(50).apply_gain(-8)
    
    # Overlay ticks at 0s, 1s, 2s
    for i in range(int(duration_sec)):
        segment = segment.overlay(tick, position=i * 1000)
    # Overlay pop at end of countdown
    segment = segment.overlay(pop, position=int(duration_sec * 1000) - 150)
    return segment


def generate_reel_caption_and_comment(
    story: Dict[str, Any],
    pipeline_mode: str,
    config: Dict[str, Any],
    part_script_text: Optional[str] = None
) -> Tuple[str, Optional[str]]:
    """Generate high-engagement, story-specific Facebook Reel caption and first comment."""
    title = str(story.get("title", "")).strip()
    subreddit = str(story.get("subreddit", "RedditStories")).strip()
    if subreddit.lower().startswith("r/"):
        subreddit = subreddit[2:]
    
    caption = ""
    comment = ""

    # 1. Try parsing JSON script blocks (for riddle mode or structured LLM rewrites)
    if part_script_text:
        try:
            import json
            parsed = json.loads(part_script_text)
            if isinstance(parsed, dict):
                caption = parsed.get("caption", "").strip()
                comment = parsed.get("pinned_comment", "").strip()
        except Exception:
            pass

    # 2. If caption is still empty or is generic fallback, build story-specific dynamic caption
    fb_cfg = config.get("facebook", {})
    if not caption or "solve this famous riddle" in caption.lower():
        if pipeline_mode == "monologue":
            custom_template = fb_cfg.get("monologue_caption")
            if custom_template:
                try:
                    caption = custom_template.format(title=title, subreddit=subreddit)
                except Exception:
                    caption = custom_template
            elif title:
                caption = f"{title} 📖🤐 What would you do? Drop your thoughts below!\n\n#storytime #redditstories #reddit #truestory #{subreddit.lower()} #drama"
            else:
                caption = f"Wild Reddit story you won't believe! 📖🤐 What are your thoughts?\n\n#storytime #redditstories #reddit #{subreddit.lower()} #truestory"
            
            if not comment:
                comment = "What would you do if you were in this situation? Let me know your honest thoughts below! 👇"

        elif pipeline_mode == "conversational":
            custom_template = fb_cfg.get("conversational_caption")
            if custom_template:
                try:
                    caption = custom_template.format(title=title, subreddit=subreddit)
                except Exception:
                    caption = custom_template
            elif title:
                caption = f"{title} 🎭🔥 What would you do in this situation? Drop a comment below!\n\n#storytime #relationship #drama #redditstories #reddit"
            else:
                caption = f"Insane drama you have to hear! 🎭🔥 Who was in the wrong here?\n\n#storytime #relationship #drama #redditstories"
            
            if not comment:
                comment = "Who do you think was right here? Lock your verdict below! 👇"

        elif pipeline_mode == "riddle":
            custom_template = fb_cfg.get("riddle_caption")
            if custom_template:
                caption = custom_template
            elif title and "compilation" not in title.lower():
                caption = f"{title} 🧠💡 Can you solve it before the timer ends?\n\n#riddle #brainteaser #mindgames #puzzle #riddles"
            else:
                caption = f"Can you solve this riddle? 🧠💡 Comment your answer below!\n\n#riddle #brainteaser #mindgames #puzzle #riddles"
            
            if not comment:
                comment = "Lock your answer in the comments before checking! 🧠👇"
        else:
            if title:
                caption = f"{title} 📖🔥 What are your thoughts?\n\n#storytime #redditstories #reddit #{subreddit.lower()}"
            else:
                caption = fb_cfg.get("caption", "Incredible story! What do you think? 💬👇 #redditstories #storytime")

    return caption, comment


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

        pipeline_mode = self.config.get("pipeline", {}).get("pipeline_mode", "monologue")
        if pipeline_mode == "auto_schedule":
            return self.run_mode_5()

        # ── Step 1: Scrape Reddit stories ────────────────────────────
        self.logger.info("═" * 60)
        self.logger.info("STEP 1 / 5  ►  Scraping Reddit stories")
        self.logger.info("═" * 60)
        scraper = RedditScraper(self.config, self.logger)
        rewriter = ScriptRewriter(self.config, self.logger)
        splitter = SmartSplitter(self.config, self.logger)

        pipeline_mode = self.config.get("pipeline", {}).get("pipeline_mode", "monologue")
        pure_gen_cfg = self.config.get("riddle", {}).get("pure_generation", False) or self.config.get("shower", {}).get("pure_generation", False)

        stories = []
        if pipeline_mode == "riddle":
            self.logger.info("  [Fun Riddle Mode] Pure Generation mode active: Bypassing Reddit scraper.")
            compilation_stories = [{
                "id": f"riddle_compilation_{timestamp_str()}",
                "title": "Fun Riddle Compilation",
                "body": "",
                "selftext": "",
                "subreddit": "Riddles",
                "author": "anonymous",
                "score": 9999,
                "candidates": []
            }]
            stories = compilation_stories
        elif pipeline_mode == "shower" and pure_gen_cfg:
            self.logger.info("  [Would You Rather] Pure Generation mode active: Bypassing Reddit scraper.")
        else:
            # Fetch more candidates per subreddit to bypass sticky posts and history matches
            fetch_limit = max(batch_size * 5, 25)
            try:
                stories = scraper.scrape_stories(limit=fetch_limit)  # fetch extra for filtering
            except Exception as e:
                self.logger.warning(f"Reddit scraping failed: {e}")
                stories = []

            if not stories:
                if pipeline_mode in ("shower", "riddle"):
                    self.logger.warning("No qualifying stories found on Reddit. Falling back to pure generation mode.")
                    pure_gen_cfg = True
                else:
                    self.logger.warning("No qualifying stories found on Reddit (either subreddits are empty, posts did not meet upvote/word-count filters, or all posts were already processed).")
                    raise NoStoriesFoundError("No qualifying stories found on Reddit.")

        if pipeline_mode == "shower":
            if pure_gen_cfg:
                compilation_stories = [{
                    "id": f"shower_compilation_{timestamp_str()}",
                    "title": "Would You Rather Compilation",
                    "subreddit": "WouldYouRather",
                    "author": "anonymous",
                    "score": 9999,
                    "candidates": []
                }]
            else:
                max_thoughts = self.config.get("shower", {}).get("max_thoughts", 5)
                compilation_stories = []
                for chunk_idx, i in enumerate(range(0, len(stories), max_thoughts)):
                    chunk = stories[i:i + max_thoughts]
                    if len(chunk) < max_thoughts:
                        break
                    compilation_story = {
                        "id": f"shower_compilation_{timestamp_str()}_{chunk_idx}",
                        "title": f"Would You Rather Compilation {chunk_idx + 1}",
                        "subreddit": "WouldYouRather",
                        "author": "anonymous",
                        "score": 9999,
                        "candidates": chunk
                    }
                    compilation_stories.append(compilation_story)
                if not compilation_stories and stories:
                    compilation_stories = [{
                        "id": f"shower_compilation_{timestamp_str()}",
                        "title": "Would You Rather Compilation",
                        "subreddit": "WouldYouRather",
                        "author": "anonymous",
                        "score": 9999,
                        "candidates": stories
                    }]
            stories = compilation_stories
        elif pipeline_mode != "riddle":
            self.logger.info(f"Fetched {len(stories)} candidate stories")

        # Process up to batch_size stories
        processed_count = 0
        story_index = 0
        while processed_count < batch_size:
            if story_index >= len(stories):
                if pipeline_mode in ("riddle", "shower") and pure_gen_cfg:
                    # Dynamically generate a new dummy story so we can try again
                    story_id = f"{pipeline_mode}_compilation_{timestamp_str()}_{story_index}"
                    new_story = {
                        "id": story_id,
                        "title": "Fun Riddle Compilation" if pipeline_mode == "riddle" else "Would You Rather Compilation",
                        "body": "",
                        "selftext": "",
                        "subreddit": "Riddles" if pipeline_mode == "riddle" else "WouldYouRather",
                        "author": "anonymous",
                        "score": 9999,
                        "candidates": []
                    }
                    stories.append(new_story)
                else:
                    break

            story = stories[story_index]
            story_index += 1

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

    def get_next_schedule_time(self, mode: str) -> int:
        """Calculate the next available scheduling slot for a given mode, persisting to schedule_history.json.
        Uses atomic write to prevent corruption.
        """
        import os
        import json
        import datetime
        from pathlib import Path
        
        history_path = Path("data/schedule_history.json")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read current history
        history = {}
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to read schedule history, resetting: {e}")
                 
        last_time_val = history.get(mode)
        
        # Define Sri Lankan Timezone
        slt_tz = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
        now_slt = datetime.datetime.now(datetime.timezone.utc).astimezone(slt_tz)
        
        # Start candidate search
        # If we have a last scheduled time, start searching from the next day or time after that
        if last_time_val:
            start_search = datetime.datetime.fromtimestamp(last_time_val, tz=slt_tz)
        else:
            start_search = now_slt
             
        # Generate possible slots starting from start_search date up to 30 days out
        candidate_slots = []
        current_date = start_search.date()
        for day_offset in range(30):
            date_val = current_date + datetime.timedelta(days=day_offset)
            slot_1 = datetime.datetime.combine(date_val, datetime.time(9, 30), tzinfo=slt_tz)
            slot_2 = datetime.datetime.combine(date_val, datetime.time(19, 30), tzinfo=slt_tz)
            candidate_slots.extend([slot_1, slot_2])
             
        # Filter slots
        # Must be at least 1 hour in the future from 'now_slt' AND strictly greater than 'start_search' (plus small margin, e.g. 5 minutes)
        min_future = now_slt + datetime.timedelta(hours=1)
        min_after_last = start_search + datetime.timedelta(minutes=5) if last_time_val else now_slt
         
        chosen_slot = None
        for slot in candidate_slots:
            if slot > min_future and slot > min_after_last:
                chosen_slot = slot
                break
                 
        if not chosen_slot:
            # Fallback to tomorrow 9:30 AM
            tomorrow = now_slt.date() + datetime.timedelta(days=1)
            chosen_slot = datetime.datetime.combine(tomorrow, datetime.time(9, 30), tzinfo=slt_tz)
             
        scheduled_timestamp = int(chosen_slot.timestamp())
         
        # Update history
        history[mode] = scheduled_timestamp
         
        # Atomic write to prevent corruption
        tmp_path = history_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            os.replace(str(tmp_path), str(history_path))
        except Exception as e:
            self.logger.error(f"Failed to persist schedule history: {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
                     
        # Log localized time for debugging
        slt_time_str = chosen_slot.strftime("%Y-%m-%d %I:%M %p Sri Lankan Time")
        self.logger.info(f"Calculated next scheduling slot for {mode}: {slt_time_str} (UNIX: {scheduled_timestamp})")
        return scheduled_timestamp

    def run_mode_5(self) -> List[Path]:
        """Runs the Batch Hybrid Scheduler Mode (Mode 5) to generate and schedule Reels."""
        from colorama import Fore, Style
        self.logger.info("============================================================")
        self.logger.info("  STARTING MODE 5: BATCH HYBRID SCHEDULER")
        self.logger.info("============================================================")
        import time
        import traceback

        # ── Step 0.1: Prompt User for Target Page ──────────────────
        print("\n" + Fore.GREEN + Style.BRIGHT + "============================================================")
        print("Select target Facebook Page / Mode to schedule:")
        print("  [1] Daily Stories (Monologue Mode — Page 1)")
        print("  [2] Reddit Stories (Conversational Mode — Page 2)")
        print("  [3] Pick your poison (Riddle Mode — Page 3)")
        print("  [4] All Pages (Full 42 Reels batch across all 3 pages)")
        print("============================================================")
        
        while True:
            try:
                page_choice = input(Fore.GREEN + "Enter choice (1, 2, 3 or 4): ").strip()
                if page_choice in ("1", "2", "3", "4"):
                    break
                print(Fore.RED + "Invalid option. Please enter 1, 2, 3 or 4.")
            except (KeyboardInterrupt, EOFError):
                page_choice = "4"
                break

        if page_choice == "1":
            modes_to_process = [{"mode": "monologue", "target_count": 14}]
        elif page_choice == "2":
            modes_to_process = [{"mode": "conversational", "target_count": 14}]
        elif page_choice == "3":
            modes_to_process = [{"mode": "riddle", "target_count": 14}]
        else:
            modes_to_process = [
                {"mode": "monologue", "target_count": 14},
                {"mode": "conversational", "target_count": 14},
                {"mode": "riddle", "target_count": 14}
            ]

        # Force approve_scripts to false during inner generation so we can ask for script permissions up-front
        original_approve = self.config.get("pipeline", {}).get("approve_scripts", True)
        self.config["pipeline"]["approve_scripts"] = False
        # Set mode_5_active so rewriter outputs JSON monologue scripts
        self.config["pipeline"]["mode_5_active"] = True

        outputs: List[Path] = []
        splitter = SmartSplitter(self.config, self.logger)

        for item in modes_to_process:
            mode_name = item["mode"]
            target = item["target_count"]
            self.logger.info(f"\n--- Processing Mode: {mode_name.upper()} (Target: {target} Reels) ---")

            # 1. Fetch & Rank/Select Stories
            selected_stories = []
            if mode_name in ("monologue", "conversational"):
                # Temporarily override subreddits and configuration for scraping
                original_subreddits = self.config["reddit"].get("subreddits", [])
                mode_cfg = self.config.get(mode_name, {})
                self.config["reddit"]["subreddits"] = mode_cfg.get("subreddits", original_subreddits)

                # Set voice overrides dynamically for scraping/TTS
                if mode_name == "monologue" and "voice" in mode_cfg:
                    self.config["tts"]["voice"] = mode_cfg["voice"]

                self.logger.info(f"Scraping candidates for {mode_name}...")
                scraper = RedditScraper(self.config, self.logger)
                try:
                    candidates = scraper.scrape_stories(limit=35)
                except Exception as e:
                    self.logger.error(f"Failed to scrape stories for {mode_name}: {e}")
                    candidates = []

                # Restore subreddits
                self.config["reddit"]["subreddits"] = original_subreddits

                if candidates:
                    self.logger.info(f"Fetched {len(candidates)} candidates. Asking Gemini to rank/select top {target}...")
                    try:
                        # Build candidate summaries text block for the LLM
                        candidate_texts = []
                        for p in candidates:
                            body_summary = p.get("body", p.get("selftext", ""))[:300]
                            candidate_texts.append(
                                f"ID: {p.get('id')}\n"
                                f"Subreddit: r/{p.get('subreddit')}\n"
                                f"Title: {p.get('title')}\n"
                                f"Body: {body_summary}...\n"
                                f"---"
                            )
                        candidates_str = "\n".join(candidate_texts)

                        system_prompt = (
                            "You are an advanced Social Media Analytics Engine specialized in algorithmic virality for vertical videos.\n"
                            "Your purpose is to analyze a batch of Reddit posts, evaluate them for virality, cognitive conflict, and hook potential, and select the top 14 posts.\n\n"
                            "Output ONLY a raw, minified JSON object containing a single key 'top_14' mapping to a list of the 14 selected story IDs in order of virality. No markdown wrappers, no backticks, no prose.\n"
                            "Example Output:\n"
                            "{\"top_14\":[\"id1\",\"id2\",...,\"id14\"]}"
                        )

                        llm_cfg = self.config.get("llm", {})
                        api_keys = llm_cfg.get("api_keys", [])
                        api_key = llm_cfg.get("api_key") or (api_keys[0] if isinstance(api_keys, list) and api_keys else "")
                        from google import genai
                        client = genai.Client(api_key=api_key)
                        response = client.models.generate_content(
                            model=self.config.get("llm", {}).get("model", "gemini-2.5-flash"),
                            contents=f"{system_prompt}\n\nCandidate Posts:\n{candidates_str}"
                        )
                        result = (response.text or "").strip()
                        if result.startswith("```"):
                            lines = result.splitlines()
                            if lines[0].startswith("```"):
                                lines = lines[1:]
                            if lines and lines[-1].startswith("```"):
                                lines = lines[:-1]
                            result = "\n".join(lines).strip()

                        import json
                        data = json.loads(result)
                        top_ids = data.get("top_14", [])
                        
                        # Reorder & pick candidates
                        for sub_id in top_ids:
                            match_post = next((p for p in candidates if p["id"] == sub_id), None)
                            if match_post:
                                selected_stories.append(match_post)
                    except Exception as gemini_err:
                        self.logger.error(f"Gemini ranking failed for {mode_name}: {gemini_err}. Falling back to default order.")
                    
                    # Fill up if selection failed or returned fewer than target
                    if len(selected_stories) < target:
                        for p in candidates:
                            if p not in selected_stories:
                                selected_stories.append(p)
                    
                    selected_stories = selected_stories[:target]
                else:
                    self.logger.warning(f"No candidates found for {mode_name}!")

            elif mode_name == "riddle":
                self.logger.info("Generating 14 riddle concepts using Gemini...")
                try:
                    system_prompt = (
                        "You are an elite riddle compilation generator. Generate exactly 14 unique, clever, and high-retention riddle concepts/clues (a mix of famous word riddles and math riddles).\n\n"
                        "Output ONLY a raw, minified JSON object containing a single key 'riddles' mapping to a list of 14 strings, where each string is a description of the riddle concept.\n"
                        "Example Output:\n"
                        "{\"riddles\":[\"Riddle about a candle...\",\"Riddle about a lock...\",...,\"Riddle about a pattern...\"]}"
                    )
                    llm_cfg = self.config.get("llm", {})
                    api_keys = llm_cfg.get("api_keys", [])
                    api_key = llm_cfg.get("api_key") or (api_keys[0] if isinstance(api_keys, list) and api_keys else "")
                    from google import genai
                    client = genai.Client(api_key=api_key)
                    response = client.models.generate_content(
                        model=self.config.get("llm", {}).get("model", "gemini-2.5-flash"),
                        contents=system_prompt
                    )
                    result = (response.text or "").strip()
                    if result.startswith("```"):
                        lines = result.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        result = "\n".join(lines).strip()

                    import json
                    data = json.loads(result)
                    riddle_concepts = data.get("riddles", [])
                    for idx, concept in enumerate(riddle_concepts):
                        selected_stories.append({
                            "id": f"riddle_{timestamp_str()}_{idx}",
                            "title": f"Riddle Concept {idx + 1}",
                            "body": concept,
                            "selftext": concept,
                            "subreddit": "Riddles",
                            "author": "anonymous",
                            "score": 9999,
                            "candidates": []
                        })
                except Exception as riddle_err:
                    self.logger.error(f"Failed to generate riddle concepts via Gemini: {riddle_err}. Falling back to default list.")
                    fallback_concepts = [
                        "What has a head and a tail but no body? A coin.",
                        "What is full of holes but still holds water? A sponge.",
                        "What goes up but never comes down? Your age.",
                        "I shave every day, but my beard stays the same. What am I? A barber.",
                        "What has hands but cannot clap? A clock.",
                        "The more of them you take, the more you leave behind. What are they? Footsteps.",
                        "I have keys but open no locks. What am I? A piano.",
                        "What has one eye but cannot see? A needle.",
                        "What has many needles but cannot sew? A pine tree.",
                        "What has a neck but no head? A bottle.",
                        "If you feed it, it lives. If you give it water, it dies. What is it? Fire.",
                        "What gets wetter the more it dries? A towel.",
                        "I am light as a feather, yet the strongest man cannot hold me for long. What am I? Breath.",
                        "What has a thumb and four fingers, but is not alive? A glove."
                    ]
                    for idx, concept in enumerate(fallback_concepts[:target]):
                        selected_stories.append({
                            "id": f"riddle_fallback_{timestamp_str()}_{idx}",
                            "title": f"Riddle Concept {idx + 1}",
                            "body": concept,
                            "selftext": concept,
                            "subreddit": "Riddles",
                            "author": "anonymous",
                            "score": 9999,
                            "candidates": []
                        })
                
                selected_stories = selected_stories[:target]

            # 2. Rewrite & Permission loop
            self.logger.info(f"Selected {len(selected_stories)} stories for {mode_name}. Requesting script permissions one by one...")
            
            self.config["pipeline"]["pipeline_mode"] = mode_name
            rewriter = ScriptRewriter(self.config, self.logger)
            scraper = RedditScraper(self.config, self.logger)

            approved_items = []
            for idx, story in enumerate(selected_stories):
                self.logger.info(f"\n[Script {idx+1}/{len(selected_stories)}] Mode: {mode_name} — Title: \"{story.get('title')}\"")
                
                # Retrieve script with up to 4 retries
                script = None
                max_retries = 4
                for attempt in range(1, max_retries + 1):
                    try:
                        self.logger.info(f"Rewriting script with Groq (Attempt {attempt}/{max_retries})...")
                        script = rewriter.rewrite(story)
                        import json
                        parsed = json.loads(script)
                        if isinstance(parsed, dict) and "script" in parsed:
                            break
                    except Exception as e:
                        self.logger.warning(f"Groq rewrite attempt {attempt} failed: {e}")
                        if attempt == max_retries:
                            self.logger.error(f"Failed to generate valid script for story {story.get('id')} after {max_retries} attempts.")
                            script = None
                
                if not script:
                    self.logger.warning(f"Skipping story {story.get('id')} due to generation failures.")
                    continue

                # Present script to user for approval
                import json
                try:
                    parsed = json.loads(script)
                    caption_val = parsed.get("caption", "").strip()
                    script_blocks = parsed.get("script", "")
                    
                    if isinstance(script_blocks, list):
                        display_text = ""
                        for block in script_blocks:
                            display_text += f"[{block.get('speaker', 'SPEAKER')}]: {block.get('text', '')}\n"
                    else:
                        display_text = str(script_blocks)
                        
                    print("\n" + Fore.CYAN + "=" * 80)
                    print(Fore.YELLOW + Style.BRIGHT + f" [MODE 5 APPROVAL] REWRITTEN STORY FOR: {mode_name.upper()}")
                    print(Fore.YELLOW + f" Title  : {story.get('title')}")
                    print(Fore.YELLOW + f" Caption: {caption_val}")
                    print(Fore.CYAN + "=" * 80)
                    print(Style.BRIGHT + display_text)
                    print(Fore.CYAN + "=" * 80)
                    
                    try:
                        action = input(Fore.GREEN + "Approve this script for scheduling? (y: yes, n: skip/reject): ").strip().lower()
                    except (KeyboardInterrupt, EOFError):
                        action = "n"
                        print()
                        
                    if action == "y":
                        approved_items.append({"story": story, "script": script})
                        self.logger.info("Script approved.")
                    else:
                        self.logger.info("Script rejected/skipped.")
                except Exception as parse_err:
                    self.logger.error(f"Error parsing generated script for approval: {parse_err}")

            # 3. Render & Schedule approved items
            if approved_items:
                self.logger.info(f"\nStarting rendering and scheduling for {len(approved_items)} approved {mode_name} Reels...")
                for idx, approved in enumerate(approved_items):
                    story = approved["story"]
                    script = approved["script"]
                    
                    self.logger.info(f"\n[Rendering approved Reel {idx+1}/{len(approved_items)}] Title: \"{story.get('title')}\"")
                    parts = splitter.split(script)
                    self.logger.info(f"Split into {len(parts)} parts.")

                    # Render parts and schedule
                    for part in parts:
                        try:
                            # Render video file
                            video_path = self._render_part(part, story)
                            if not video_path or not video_path.exists():
                                self.logger.warning(f"Failed to render video for story {story.get('id')}.")
                                continue

                            # Generate dynamic caption and comment
                            caption_to_use, comment_to_use = generate_reel_caption_and_comment(
                                story=story,
                                pipeline_mode=mode_name,
                                config=self.config,
                                part_script_text=part.get("script_text")
                            )

                            # Calculate next SLT schedule timestamp
                            schedule_time = self.get_next_schedule_time(mode_name)

                            # Publish / Schedule (No pinned comment, comment=None)
                            from src.facebook_publisher import FacebookReelsPublisher
                            publisher = FacebookReelsPublisher(self.config, self.logger, pipeline_mode=mode_name)
                            publish_success = publisher.publish_reel(
                                video_path=video_path,
                                caption=caption_to_use,
                                pinned_comment=None,
                                scheduled_publish_time=schedule_time
                            )

                            if publish_success:
                                outputs.append(video_path)
                                self.logger.info(f"Successfully scheduled video: {video_path.name}")
                            else:
                                self.logger.warning(f"Failed to schedule video on Facebook: {video_path.name}")

                            # Hardware cool-down
                            self.logger.info("Pacing: Entering 30-second hardware cool-down...")
                            time.sleep(30)

                        except Exception as render_err:
                            self.logger.error(f"Error rendering/scheduling part: {render_err}")
                            self.logger.debug(traceback.format_exc())

                    # Mark story as scraped
                    try:
                        scraper.mark_scraped(story["id"])
                    except Exception:
                        pass
            else:
                self.logger.info(f"No approved scripts for {mode_name}. Skipping rendering/scheduling.")

        # Restore original settings
        self.config["pipeline"]["approve_scripts"] = original_approve
        self.config["pipeline"]["mode_5_active"] = False
        self.config["pipeline"]["pipeline_mode"] = "auto_schedule"

        self.logger.info("============================================================")
        self.logger.info(f"  MODE 5 COMPLETED: {len(outputs)} REELS SCHEDULED")
        self.logger.info("============================================================")
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
        # If thread mode, fetch comments on-demand
        pipeline_mode = self.config.get("pipeline", {}).get("pipeline_mode", "monologue")
        reel_caption = ""
        reel_pinned_comment = ""
        if pipeline_mode == "thread":
            self.logger.info(f"  Thread Mode: Fetching comments for post {story['id']}...")
            comments = scraper.fetch_comments(story["id"], story["subreddit"])
            if not comments:
                self.logger.warning(f"  No comments found for post {story['id']}. Skipping story.")
                # Mark story as scraped so we don't try it again
                scraper.mark_scraped(story["id"])
                return []
            story["comments"] = comments

        script = rewriter.rewrite(story)
        self.logger.info(
            f"  Script: {len(script.split())} words, "
            f"est. {format_duration(estimate_duration_sec(script))}"
        )

        approve_required = self.config.get("pipeline", {}).get("approve_scripts", False) and sys.stdin.isatty()
        if approve_required:
            feedback_str = None
            while True:
                # If they chose to rewrite with feedback:
                if feedback_str:
                    script = rewriter.rewrite(story, feedback=feedback_str)
                    feedback_str = None # Reset
                
                print("\n" + Fore.CYAN + "=" * 80)
                print(Fore.YELLOW + Style.BRIGHT + f" [PENDING APPROVAL] REWRITTEN STORY SCRIPT FOR: r/{subreddit}")
                print(Fore.YELLOW + f" Title: {story.get('title', 'Untitled')}")
                print(Fore.YELLOW + f" Estimated Duration: {format_duration(estimate_duration_sec(script))}")
                if story.get("virality_reason"):
                    print(Fore.RED + f" Virality Reason: {story['virality_reason']}")
                print(Fore.CYAN + "=" * 80)
                print(Style.BRIGHT + script)
                print(Fore.CYAN + "=" * 80)

                try:
                    action = None
                    while True:
                        prompt_msg = Fore.GREEN + Style.BRIGHT + "Approve script? (y: yes, n: no, e: edit, r: rewrite with feedback): "
                        choice = input(prompt_msg).strip().lower()
                        if choice in ("y", "yes", "n", "no", "e", "edit", "r", "rewrite"):
                            action = choice[0] # 'y', 'n', 'e', or 'r'
                            break
                        else:
                            print(Fore.RED + "Invalid input. Please enter 'y', 'n', 'e', or 'r'.")
                except (KeyboardInterrupt, EOFError):
                    self.logger.warning("\n[INTERRUPTED] Script approval cancelled. Defaulting to reject/skip.")
                    action = 'n'

                if action == 'y':
                    # Prompt for caption if riddle/conversational/monologue mode and facebook integration is enabled
                    fb_cfg = self.config.get("facebook", {})
                    if pipeline_mode in ("monologue", "riddle", "conversational") and fb_cfg.get("enabled", False):
                        print("\n" + Fore.GREEN + Style.BRIGHT + "Configure Facebook Reel Caption:")
                        
                        # Extract the hook, caption, and comment from script JSON
                        hook_txt = ""
                        gemini_caption = ""
                        gemini_comment = ""
                        try:
                            import json
                            blocks = json.loads(script)
                            if isinstance(blocks, dict):
                                script_arr = blocks.get("script", [])
                                gemini_caption = blocks.get("caption", "").strip()
                                gemini_comment = blocks.get("pinned_comment", "").strip()
                            else:
                                script_arr = blocks
                                
                            if isinstance(script_arr, list) and len(script_arr) > 0:
                                hook_txt = script_arr[0].get("text", "")
                        except Exception:
                            # Monologue fallback: first sentence or first 100 characters
                            hook_txt = script.strip().split('.')[0] if '.' in script else script.strip()[:100]
                        
                        cfg_caption = fb_cfg.get("caption", "")
                        if not gemini_caption:
                            gemini_caption = cfg_caption
                            
                        print(Fore.YELLOW + f"[1] Gemini-Generated Caption: \"{gemini_caption}\"")
                        print(Fore.YELLOW + f"[2] Default Config Caption: \"{cfg_caption}\"")
                        if hook_txt:
                            if pipeline_mode == "riddle":
                                print(Fore.YELLOW + f"[3] Dynamic Riddle Hook: \"{hook_txt} 🧠🔥 #riddle #brainteaser\"")
                            elif pipeline_mode == "conversational":
                                print(Fore.YELLOW + f"[3] Dynamic Story Hook: \"{hook_txt} 🎭🔥 #storytime #relationship #drama\"")
                            else:
                                print(Fore.YELLOW + f"[3] Dynamic Monologue Hook: \"{hook_txt} 📖🗣️ #storytime #redditstories #audiobook\"")
                        print(Fore.YELLOW + "[4] Custom Caption (write your own)")
                        
                        while True:
                            try:
                                cap_choice = input(Fore.GREEN + "Choose caption option [1]: ").strip()
                            except (KeyboardInterrupt, EOFError):
                                cap_choice = "1"
                                print()
                            if not cap_choice:
                                cap_choice = "1"
                            if cap_choice == "1":
                                reel_caption = gemini_caption
                                break
                            elif cap_choice == "2":
                                reel_caption = cfg_caption
                                break
                            elif cap_choice == "3" and hook_txt:
                                if pipeline_mode == "riddle":
                                    reel_caption = f"{hook_txt} 🧠🔥 #riddle #brainteaser"
                                elif pipeline_mode == "conversational":
                                    reel_caption = f"{hook_txt} 🎭🔥 #storytime #relationship #drama"
                                else:
                                    reel_caption = f"{hook_txt} 📖🗣️ #storytime #redditstories #audiobook"
                                break
                            elif cap_choice == "4":
                                try:
                                    reel_caption = input(Fore.GREEN + "Enter custom caption: ").strip()
                                except (KeyboardInterrupt, EOFError):
                                    reel_caption = gemini_caption
                                    print()
                                break
                            else:
                                print(Fore.RED + "Invalid option. Please enter 1, 2, 3, or 4.")
                        
                        self.logger.info(Fore.GREEN + f"Selected Reel Caption: \"{reel_caption}\"")
                        
                        # Prompt for Pinned Comment selection
                        print("\n" + Fore.GREEN + Style.BRIGHT + "Configure Facebook Reel Pinned Comment:")
                        
                        if pipeline_mode == "riddle":
                            fallback_comment = "Can you solve this famous riddle? Comment your answer below! 🧠🔥"
                        elif pipeline_mode == "conversational":
                            fallback_comment = "Who was in the wrong here? Drop your honest thoughts! 🎭"
                        else:
                            fallback_comment = "What are your thoughts on this story? Comment below! 🎭"
                            
                        if not gemini_comment:
                            gemini_comment = fallback_comment
                            
                        print(Fore.YELLOW + f"[1] Gemini-Generated Comment: \"{gemini_comment}\"")
                        print(Fore.YELLOW + f"[2] Default Fallback Comment: \"{fallback_comment}\"")
                        print(Fore.YELLOW + "[3] Custom Comment (write your own)")
                        
                        while True:
                            try:
                                comm_choice = input(Fore.GREEN + "Choose comment option [1]: ").strip()
                            except (KeyboardInterrupt, EOFError):
                                comm_choice = "1"
                                print()
                            if not comm_choice:
                                comm_choice = "1"
                            if comm_choice == "1":
                                reel_pinned_comment = gemini_comment
                                break
                            elif comm_choice == "2":
                                reel_pinned_comment = fallback_comment
                                break
                            elif comm_choice == "3":
                                try:
                                    reel_pinned_comment = input(Fore.GREEN + "Enter custom comment: ").strip()
                                except (KeyboardInterrupt, EOFError):
                                    reel_pinned_comment = gemini_comment
                                    print()
                                break
                            else:
                                print(Fore.RED + "Invalid option. Please enter 1, 2, or 3.")
                                
                        self.logger.info(Fore.GREEN + f"Selected Reel Pinned Comment: \"{reel_pinned_comment}\"")
                    break
                elif action == 'n':
                    self.logger.info(Fore.YELLOW + f"Script rejected/skipped for post {story['id']}. Skipping to the next story...")
                    # Mark story as scraped so we don't pick it up again
                    scraper.mark_scraped(story["id"])
                    return []
                elif action == 'e':
                    # Let the user edit the script in the default system text editor (opens Notepad on Windows)
                    import os
                    import subprocess
                    
                    edit_file = self.temp_dir / "edit_script.txt"
                    try:
                        # Write current script to temp file
                        edit_file.write_text(script, encoding="utf-8")
                        
                        # Determine editor
                        if os.name == 'nt':
                            editor = "notepad.exe"
                        else:
                            editor = os.environ.get("EDITOR", "nano")
                            
                        self.logger.info(f"Opening script in editor: {editor}. Please make your edits and save/close the file.")
                        subprocess.run([editor, str(edit_file)], check=True)
                        
                        # Read modified script back
                        if edit_file.exists():
                            script = edit_file.read_text(encoding="utf-8").strip()
                            self.logger.info("Edited script loaded successfully.")
                    except Exception as e:
                        self.logger.error(f"Failed to edit script: {e}")
                    finally:
                        if edit_file.exists():
                            try:
                                edit_file.unlink()
                            except Exception:
                                pass
                elif action == 'r':
                    # Prompt for feedback to rewrite
                    try:
                        feedback_str = input(Fore.GREEN + Style.BRIGHT + "Enter rewriting feedback for the LLM: ").strip()
                        if not feedback_str:
                            self.logger.info("Feedback empty. Reloading current script.")
                            feedback_str = None
                    except (KeyboardInterrupt, EOFError):
                        self.logger.warning("\n[INTERRUPTED] Feedback input cancelled.")
                        feedback_str = None

        # ── 1b: Smart split ──────────────────────────────────────────
        parts = splitter.split(script)
        self.logger.info(f"  Split into {len(parts)} part(s)")

        # Mark story as scraped
        if pipeline_mode in ("shower", "riddle"):
            import json
            try:
                parsed_script = json.loads(script)
                blocks_arr = parsed_script.get("script", []) if isinstance(parsed_script, dict) else parsed_script
                if isinstance(blocks_arr, list):
                    for block in blocks_arr:
                        block_text = block.get("text", "").lower()
                        for c in story.get("candidates", []):
                            c_title = c.get("title", "").lower()
                            if c["id"] not in scraper._history and (c_title in block_text or c["id"] in block.get("author", "")):
                                scraper.mark_scraped(c["id"])
            except Exception as e:
                self.logger.warning(f"Failed to auto-mark compiled shower/riddle thought ids as scraped: {e}")
        else:
            scraper.mark_scraped(story["id"])

        # ── Process each part ────────────────────────────────────────
        output_paths: List[Path] = []
        for part in parts:
            try:
                path = self._render_part(part, story)
                output_paths.append(path)
                
                # Auto-post to Facebook Reels if enabled and mode is monologue/riddle/conversational
                fb_cfg = self.config.get("facebook", {})
                if pipeline_mode in ("monologue", "riddle", "conversational") and fb_cfg.get("enabled", False) and path:
                    caption_to_use, comment_to_use = generate_reel_caption_and_comment(
                        story=story,
                        pipeline_mode=pipeline_mode,
                        config=self.config,
                        part_script_text=part.get("script_text")
                    )
                        
                    print("\n" + Fore.GREEN + Style.BRIGHT + "============================================================")
                    print(Fore.GREEN + Style.BRIGHT + f"  Video rendered successfully: {path.name}")
                    print(Fore.GREEN + Style.BRIGHT + "============================================================")
                    
                    fb_enabled = self.config.get("facebook", {}).get("enabled", False)
                    if not sys.stdin.isatty() or not self.config.get("pipeline", {}).get("approve_scripts", False):
                        post_choice = 'y' if fb_enabled else 'n'
                    else:
                        try:
                            post_choice = input(Fore.GREEN + "Post this video to Facebook Reels? (y: yes/post, n: save only): ").strip().lower()
                        except (KeyboardInterrupt, EOFError):
                            post_choice = "y" if fb_enabled else "n"
                            print()
                        
                    if post_choice == 'y':
                        self.logger.info(Fore.GREEN + Style.BRIGHT + "  PUBLISHING TO FACEBOOK REELS...")
                        self.logger.info("Initializing Facebook Reels auto-poster...")
                        from src.facebook_publisher import FacebookReelsPublisher
                        publisher = FacebookReelsPublisher(self.config, self.logger, pipeline_mode=pipeline_mode)
                        publish_success = publisher.publish_reel(path, caption_to_use, comment_to_use)
                        if publish_success:
                            self.logger.info("Facebook Reel successfully published!")
                        else:
                            self.logger.warning("Facebook Reel publishing failed.")

                        # Auto-post to YouTube Shorts if enabled or YOUTUBE credentials present
                        try:
                            from src.youtube_publisher import YouTubeShortsPublisher
                            yt_publisher = YouTubeShortsPublisher(self.config, self.logger)
                            if yt_publisher.enabled:
                                self.logger.info(Fore.RED + Style.BRIGHT + "  PUBLISHING TO YOUTUBE SHORTS...")
                                yt_title = story.get("title", "Reddit Story")
                                yt_publisher.upload_short(
                                    video_path=path,
                                    title=yt_title,
                                    description=caption_to_use,
                                )
                        except Exception as yt_err:
                            self.logger.warning(f"YouTube Shorts upload encountered an error: {yt_err}")
                    else:
                        self.logger.info(Fore.YELLOW + "  Post skipped. Video saved only in the output folder.")
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
        from pydub import AudioSegment

        pipeline_mode = self.config.get("pipeline", {}).get("pipeline_mode", "monologue")
        speed_factor = 1.20 if pipeline_mode == "shower" else (1.00 if pipeline_mode == "riddle" else 1.07)
        
        is_conversational = False
        is_thread = False
        dialogue_blocks = []
        if pipeline_mode in ("conversational", "riddle"):
            import json
            try:
                parsed = json.loads(part["script_text"])
                if isinstance(parsed, dict) and "script" in parsed:
                    is_conversational = True
                    dialogue_blocks = parsed["script"]
                elif isinstance(parsed, list) and all(isinstance(x, dict) and "speaker" in x and "text" in x for x in parsed):
                    is_conversational = True
                    dialogue_blocks = parsed
            except Exception as e:
                self.logger.warning(f"    Failed to parse dialogue blocks as JSON: {e}. Falling back to monologue TTS.")
        elif pipeline_mode in ("thread", "shower"):
            import json
            try:
                parsed = json.loads(part["script_text"])
                if isinstance(parsed, list) and all(isinstance(x, dict) and "speaker" in x and "text" in x for x in parsed):
                    is_thread = True
                    dialogue_blocks = parsed
            except Exception as e:
                self.logger.warning(f"    Failed to parse thread dialogue blocks as JSON: {e}. Falling back to monologue TTS.")
        
        if is_conversational or is_thread:
            self.logger.info(f"    {'Thread' if is_thread else 'Conversational'} Mode: Synthesising {len(dialogue_blocks)} dialogue blocks block-by-block")
            
            engine_type = self.config.get("tts", {}).get("engine", "kokoro").lower()
            tts_cfg = self.config.get("tts", {})
            conv_cfg = self.config.get("conversational", {})

            if is_thread:
                if pipeline_mode == "shower":
                    default_voice = self.config.get("shower", {}).get("default_voice", "am_adam" if engine_type == "kokoro" else "en-US-ChristopherNeural")
                else:
                    default_voice = self.config.get("thread", {}).get("default_voice", "am_adam" if engine_type == "kokoro" else "en-US-ChristopherNeural")
            else:
                if engine_type == "kokoro":
                    male_voice = tts_cfg.get("male_voice", "am_adam")
                    female_voice = tts_cfg.get("female_voice", "af_bella")
                else:
                    male_voice = conv_cfg.get("male_voice", "en-US-GuyNeural")
                    female_voice = conv_cfg.get("female_voice", "en-US-MichelleNeural")
            
            segments_to_combine = []
            block_timings = []
            current_time_ms = 0
            
            # Make a copy of blocks to avoid mutating text across multiple retries/rewrites
            dialogue_blocks = [dict(b) for b in dialogue_blocks]
            
            outro_enabled = self.config.get("pipeline", {}).get("outro_enabled", False)
            outro_text = self.config.get("pipeline", {}).get("outro_text", "Follow for more!")
            if outro_enabled and outro_text:
                if pipeline_mode == "shower":
                    dialogue_blocks.append({
                        "speaker": "OUTRO",
                        "voice": self.config.get("shower", {}).get("default_voice", "am_adam" if engine_type == "kokoro" else "en-US-ChristopherNeural"),
                        "author": "system",
                        "score": 99999,
                        "text": outro_text
                    })
                elif is_thread:
                    dialogue_blocks.append({
                        "speaker": "OUTRO",
                        "voice": self.config.get("thread", {}).get("default_voice", "am_adam" if engine_type == "kokoro" else "en-US-ChristopherNeural"),
                        "author": "system",
                        "score": 99999,
                        "text": outro_text
                    })
                else:
                    dialogue_blocks.append({
                        "speaker": "OUTRO",
                        "text": outro_text
                    })
            
            # Prepare the last block with tail phrase for vocal decay hack (except for shower and riddle modes)
            if pipeline_mode not in ("shower", "riddle") and len(dialogue_blocks) > 0:
                last_block = dialogue_blocks[-1]
                last_block["text"] = cadence.prepare_script_with_tail(last_block["text"])
            
            for idx, block in enumerate(dialogue_blocks):
                speaker = block.get("speaker", "MALE").strip().upper()
                if pipeline_mode == "conversational":
                    speaker = resolve_speaker_category(speaker)
                block["speaker"] = speaker
                text = block.get("text", "")
                
                if is_thread:
                    voice = block.get("voice")
                    if not voice or (engine_type == "kokoro" and ("Neural" in str(voice) or "-" in str(voice))):
                        voice = default_voice
                else:
                    if engine_type == "kokoro":
                        if speaker in ("OLD_FEMALE", "MOTHER", "MOM"):
                            voice = tts_cfg.get("old_female_voice", "af_nicole")
                        elif speaker in ("OLD_MALE", "FATHER", "DAD"):
                            voice = tts_cfg.get("old_male_voice", "bm_george")
                        elif speaker in ("CHILD_FEMALE", "CHIBI_FEMALE"):
                            voice = tts_cfg.get("child_female_voice", "af_sky")
                        elif speaker in ("CHILD_MALE", "CHIBI_MALE"):
                            voice = tts_cfg.get("child_male_voice", "am_puck")
                        elif speaker in ("FEMALE", "GIRLFRIEND", "WIFE"):
                            voice = tts_cfg.get("female_voice", "af_bella")
                        else:
                            voice = tts_cfg.get("male_voice", "am_adam")
                    else:
                        is_male = speaker in ("MALE", "CHILD_MALE", "CHIBI_MALE", "OLD_MALE")
                        voice = male_voice if is_male else female_voice
                
                self.logger.info(
                    f"      Block {idx+1}/{len(dialogue_blocks)} [{speaker}] -> "
                    f"using voice '{voice}': \"{text[:50]}...\""
                )
                block_audio_path = tts.synthesize(text, voice=voice)
                block_audio = AudioSegment.from_file(str(block_audio_path))
                segments_to_combine.append(block_audio)
                
            # Combine individual audio clips with a 200ms pause cushion between speakers
            # For 'shower' (Would You Rather) mode, we append 3s ticking countdown and 2s reveal pause after each question.
            master_audio = AudioSegment.empty()
            for idx, block in enumerate(dialogue_blocks):
                clip_audio = segments_to_combine[idx]
                clip_dur = len(clip_audio)
                speaker = block.get("speaker", "MALE")
                
                if pipeline_mode == "shower" and speaker.startswith("Q"):
                    # Generate ticking countdown (5 seconds) and reveal pause (3 seconds)
                    countdown_audio = generate_countdown_sfx(duration_sec=5.0)
                    reveal_audio = AudioSegment.silent(duration=3000)
                    
                    master_audio += clip_audio + countdown_audio + reveal_audio
                    
                    block_timings.append({
                        "start_ms": current_time_ms,
                        "end_ms": current_time_ms + clip_dur,
                        "countdown_start_ms": current_time_ms + clip_dur,
                        "countdown_end_ms": current_time_ms + clip_dur + 5000,
                        "reveal_start_ms": current_time_ms + clip_dur + 5000,
                        "reveal_end_ms": current_time_ms + clip_dur + 8000,
                        "speaker": speaker,
                        "emotion": block.get("emotion", "talking"),
                        "text": block.get("text", ""),
                        "option_a": block.get("option_a", ""),
                        "option_b": block.get("option_b", ""),
                        "percentage_a": block.get("percentage_a", 50),
                        "author": block.get("author", "reddit_user"),
                        "score": block.get("score", 0),
                    })
                    current_time_ms += clip_dur + 8000
                else:
                    block_timings.append({
                        "start_ms": current_time_ms,
                        "end_ms": current_time_ms + clip_dur,
                        "speaker": speaker,
                        "emotion": block.get("emotion", "talking"),
                        "text": block.get("text", ""),
                        "author": block.get("author", "reddit_user"),
                        "score": block.get("score", 0),
                    })
                    
                    master_audio += clip_audio
                    current_time_ms += clip_dur
                    
                    if idx < len(dialogue_blocks) - 1:
                        master_audio += AudioSegment.silent(duration=200)
                        current_time_ms += 200
            
            raw_audio_path = self.temp_dir / f"tts_raw_p{part['part_number']}.mp3"
            master_audio.export(str(raw_audio_path), format="mp3", bitrate="128k")
            self.logger.info(f"    Combined raw {'thread' if is_thread else 'conversational'} audio saved: {raw_audio_path.name}")
        else:
            # Determine narrator gender and select voice dynamically
            narrator_gender = (story or {}).get("narrator_gender")
            engine_type = self.config.get("tts", {}).get("engine", "kokoro").lower()
            if engine_type == "kokoro":
                male_voice = self.config.get("tts", {}).get("male_voice", "am_adam")
                female_voice = self.config.get("tts", {}).get("female_voice", "af_heart")
                config_voice = self.config.get("tts", {}).get("voice", "am_adam")
            else:
                male_voice = self.config.get("tts", {}).get("edge_tts_voice", "en-US-ChristopherNeural")
                female_voice = self.config.get("tts", {}).get("edge_tts_female_voice", "en-US-JennyNeural")
                config_voice = self.config.get("tts", {}).get("voice", "en-US-ChristopherNeural")
            
            if narrator_gender == "MALE":
                voice = male_voice
                self.logger.info(f"    Dynamic Voice Selector ({engine_type.upper()}): Narrator detected as MALE -> selected '{voice}'")
            elif narrator_gender == "FEMALE":
                voice = female_voice
                self.logger.info(f"    Dynamic Voice Selector ({engine_type.upper()}): Narrator detected as FEMALE -> selected '{voice}'")
            else:
                voice = config_voice
                self.logger.info(f"    Dynamic Voice Selector ({engine_type.upper()}): Narrator gender undetected -> selected default voice '{voice}'")

            # Append outro text if enabled
            outro_enabled = self.config.get("pipeline", {}).get("outro_enabled", False)
            outro_text = self.config.get("pipeline", {}).get("outro_text", "Follow for more!")
            script_text = part["script_text"]
            import json
            try:
                parsed = json.loads(script_text)
                if isinstance(parsed, dict) and "script" in parsed and isinstance(parsed["script"], str):
                    script_text = parsed["script"]
            except Exception:
                pass

            if outro_enabled and outro_text:
                script_text = script_text.strip()
                if not script_text.endswith((".", "!", "?")):
                    script_text += "."
                script_text += f" {outro_text}"

            # Append tail phrase for vocal decay hack
            script_with_tail = cadence.prepare_script_with_tail(script_text)
            raw_audio_path = tts.synthesize(script_with_tail, voice=voice)
            self.logger.info(f"    Raw TTS audio saved: {raw_audio_path.name}")

        # Speed up the raw audio by speed_factor before extracting timestamps
        speed_audio_path = raw_audio_path.parent / f"speed_{raw_audio_path.name}"
        self.logger.info(f"    Applying {speed_factor}x audio speed-up using FFmpeg...")
        
        import subprocess
        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(raw_audio_path),
            "-filter:a", f"atempo={speed_factor}",
            str(speed_audio_path),
        ]
        try:
            self.logger.debug(f"    Running FFmpeg speed-up command: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=60)
            audio_path = speed_audio_path
            self.logger.info(f"    Accelerated audio saved: {audio_path.name}")
        except Exception as exc:
            self.logger.error(
                f"    Failed to speed up audio using FFmpeg: {exc}. "
                "Falling back to raw audio."
            )
            audio_path = raw_audio_path

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
        if pipeline_mode != "shower":
            self.logger.info("  STEP 2b ►  Applying acoustic cadence processing")
            processed_audio = cadence.process_audio(audio_path, ts_data["words"])
            self.logger.info(f"    Processed audio: {processed_audio.name}")

            # Re-extract timestamps on the processed audio (without tail)
            self.logger.info("  STEP 3b ►  Re-extracting timestamps on processed audio")
            ts_data_final = extractor.extract(processed_audio)
        else:
            self.logger.info("  STEP 2b ►  Skipping acoustic cadence processing for shower mode")
            processed_audio = audio_path
            ts_data_final = ts_data
        self.logger.info(
            f"    Final: {len(ts_data_final['words'])} words, "
            f"{len(ts_data_final['sentences'])} sentences"
        )

        # ── Step 3b (Speaker mapping) ────────────────────────────────
        pipeline_mode = self.config.get("pipeline", {}).get("pipeline_mode", "monologue")
        if (is_conversational or is_thread) and 'block_timings' in locals():
            self.logger.info("    Multi-speaker mode active: Tagging words with speaker metadata")
            for w in ts_data_final["words"]:
                w_start = w["start"]
                best_speaker = "MALE"
                min_dist = float("inf")
                for b_info in block_timings:
                    b_start = b_info["start_ms"] / 1000.0 / speed_factor
                    b_end = b_info["end_ms"] / 1000.0 / speed_factor
                    if b_start <= w_start <= b_end:
                        best_speaker = b_info["speaker"]
                        break
                    else:
                        dist = min(abs(w_start - b_start), abs(w_start - b_end))
                        if dist < min_dist:
                            min_dist = dist
                            best_speaker = b_info["speaker"]
                w["speaker"] = best_speaker
            
            # Regenerate sentences since words have been updated with speaker info
            ts_data_final["sentences"] = extractor._group_into_sentences(ts_data_final["words"])

        # ── Step 3c: Subtitle Rendering ──────────────────────────────
        self.logger.info("  STEP 3c ►  Rendering double-pass subtitles")
        screenshot_cfg = self.config.get("screenshot", {})
        show_screenshot = screenshot_cfg.get("enabled", True) and part.get("part_number", 1) == 1
        
        # Calculate dynamic title card duration based on the first sentence
        hook_duration = screenshot_cfg.get("display_duration_sec", 3.5)
        if show_screenshot and ts_data_final.get("sentences") and len(ts_data_final["sentences"]) > 1:
            hook_duration = ts_data_final["sentences"][0]["end"]
            # Clamp between 2.0s and 6.0s for aesthetic safety
            hook_duration = max(2.0, min(hook_duration, 6.0))

        sub_renderer = SubtitleRenderer(self.config, self.logger)
        frame_size = (
            self.config["video"]["resolution"]["width"],
            self.config["video"]["resolution"]["height"],
        )
        # Render subtitles from 0.0 seconds so they are visible during the Reddit screenshot card.
        # Overlapping is avoided since we repositioned the screenshot card higher up.
        subtitle_clips = sub_renderer.render(ts_data_final, frame_size, skip_until=0.0)
        self.logger.info(f"    Generated {len(subtitle_clips)} subtitle clips")

        # ── Step 3d: Reddit Screenshot Hook & Comment overlays ────────
        if show_screenshot:
            if pipeline_mode == "shower":
                self.logger.info("  STEP 3d ►  Capturing Would You Rather multi-state poll cards")
                try:
                    ss_manager = ScreenshotManager(self.config, self.logger)
                    for idx, b_timing in enumerate(block_timings):
                        start_sec = b_timing["start_ms"] / 1000.0 / speed_factor
                        end_sec = b_timing["end_ms"] / 1000.0 / speed_factor
                        dur_sec = end_sec - start_sec
                        
                        speaker = b_timing.get("speaker", "MALE")
                        
                        if speaker == "INTRO":
                            # Standard post card for intro hook
                            intro_story = {
                                "subreddit": "WouldYouRather",
                                "title": b_timing["text"],
                                "author": b_timing["author"],
                                "score": b_timing["score"],
                            }
                            post_card_png = ss_manager.capture(intro_story)
                            post_card_clip = ss_manager.create_hook_clip(
                                screenshot_path=post_card_png,
                                frame_size=frame_size,
                                duration=dur_sec,
                            )
                            post_card_clip = post_card_clip.set_start(start_sec)
                            subtitle_clips.append(post_card_clip)
                            self.logger.info(f"    WYR Intro card overlay added: start={start_sec:.2f}s, duration={dur_sec:.2f}s")
                            
                        elif speaker.startswith("Q"):
                            # WYR Question poll card
                            option_a = b_timing.get("option_a", "Option A")
                            option_b = b_timing.get("option_b", "Option B")
                            percentage_a = b_timing.get("percentage_a", 50)
                            
                            poll_data = {
                                "option_a": option_a,
                                "option_b": option_b,
                                "percentage_a": percentage_a,
                            }
                            
                            q_num = 1
                            try:
                                q_num = int(speaker[1:])
                            except ValueError:
                                pass
                                
                            # 1. State: "question" (until countdown starts)
                            q_png = ss_manager.capture_poll_card(poll_data, q_num, state="question")
                            q_clip = ss_manager.create_hook_clip(
                                screenshot_path=q_png,
                                frame_size=frame_size,
                                duration=dur_sec,
                                fade_duration=0.0,
                                ken_burns=False,
                            )
                            q_clip = q_clip.set_start(start_sec)
                            subtitle_clips.append(q_clip)
                            
                            # Countdown tick metrics
                            countdown_start_sec = b_timing["countdown_start_ms"] / 1000.0 / speed_factor
                            tick_dur = 1.0 / speed_factor
                            
                            # 2. States: "5" down to "1"
                            for tick_val in range(5, 0, -1):
                                tick_state = str(tick_val)
                                t_png = ss_manager.capture_poll_card(poll_data, q_num, state=tick_state)
                                t_clip = ss_manager.create_hook_clip(
                                    screenshot_path=t_png,
                                    frame_size=frame_size,
                                    duration=tick_dur,
                                    fade_duration=0.0,
                                    ken_burns=False,
                                )
                                offset = (5 - tick_val) * tick_dur
                                t_clip = t_clip.set_start(countdown_start_sec + offset)
                                subtitle_clips.append(t_clip)
                            
                            # 5. State: "reveal" (during reveal silence duration)
                            reveal_start_sec = b_timing["reveal_start_ms"] / 1000.0 / speed_factor
                            reveal_end_sec = b_timing["reveal_end_ms"] / 1000.0 / speed_factor
                            reveal_dur = reveal_end_sec - reveal_start_sec
                            
                            r_png = ss_manager.capture_poll_card(poll_data, q_num, state="reveal")
                            r_clip = ss_manager.create_hook_clip(
                                screenshot_path=r_png,
                                frame_size=frame_size,
                                duration=reveal_dur,
                                fade_duration=min(0.5, reveal_dur * 0.15),
                                ken_burns=False,
                            )
                            r_clip = r_clip.set_start(reveal_start_sec)
                            subtitle_clips.append(r_clip)
                            
                            self.logger.info(
                                f"    WYR Card Q{q_num} overlays added: "
                                f"q_start={start_sec:.2f}s, q_dur={dur_sec:.2f}s, "
                                f"countdown_start={countdown_start_sec:.2f}s, "
                                f"reveal_start={reveal_start_sec:.2f}s, reveal_dur={reveal_dur:.2f}s"
                            )
                        elif speaker == "OUTRO":
                            # Render beautiful Outro card
                            outro_png = ss_manager.capture_outro_card(b_timing["text"])
                            outro_clip = ss_manager.create_hook_clip(
                                screenshot_path=outro_png,
                                frame_size=frame_size,
                                duration=dur_sec,
                                fade_duration=min(0.5, dur_sec * 0.15),
                                ken_burns=False,
                            )
                            outro_clip = outro_clip.set_start(start_sec)
                            subtitle_clips.append(outro_clip)
                            self.logger.info(f"    WYR Outro card overlay added: start={start_sec:.2f}s, duration={dur_sec:.2f}s")
                except Exception as exc:
                    self.logger.warning(
                        "  Would You Rather poll screenshots failed (non-fatal): %s", exc
                    )
            elif is_thread:
                self.logger.info("  STEP 3d ►  Capturing Thread hook and comment screenshots")
                try:
                    import random
                    ss_manager = ScreenshotManager(self.config, self.logger)
                    
                    # 1. Thread hook (question card)
                    hook_duration_thread = 3.5
                    if len(block_timings) > 0:
                        hook_duration_thread = (block_timings[0]["end_ms"] - block_timings[0]["start_ms"]) / 1000.0 / speed_factor
                    
                    hook_png = ss_manager.capture(story)
                    hook_clip = ss_manager.create_hook_clip(
                        screenshot_path=hook_png,
                        frame_size=frame_size,
                        duration=hook_duration_thread,
                    )
                    subtitle_clips.append(hook_clip)
                    self.logger.info(f"    Thread hook overlay added ({hook_duration_thread:.2f}s)")
                    
                    # 2. Comment cards (subsequent blocks)
                    for idx, b_timing in enumerate(block_timings[1:], start=1):
                        start_sec = b_timing["start_ms"] / 1000.0 / speed_factor
                        end_sec = b_timing["end_ms"] / 1000.0 / speed_factor
                        dur_sec = end_sec - start_sec
                        
                        if b_timing.get("speaker") == "OUTRO":
                            outro_png = ss_manager.capture_outro_card(b_timing["text"])
                            outro_clip = ss_manager.create_hook_clip(
                                screenshot_path=outro_png,
                                frame_size=frame_size,
                                duration=dur_sec,
                                fade_duration=min(0.5, dur_sec * 0.15),
                                ken_burns=False,
                            )
                            outro_clip = outro_clip.set_start(start_sec)
                            subtitle_clips.append(outro_clip)
                            self.logger.info(f"    Thread Outro card overlay added: start={start_sec:.2f}s, duration={dur_sec:.2f}s")
                            continue
                        
                        comment_data = {
                            "author": b_timing.get("author", "reddit_user"),
                            "text": b_timing.get("text", ""),
                            "score": b_timing.get("score", random.randint(100, 2500)),
                        }
                        
                        comment_png = ss_manager.capture_comment(comment_data, idx)
                        comment_clip = ss_manager.create_comment_clip(
                            screenshot_path=comment_png,
                            frame_size=frame_size,
                            start=start_sec,
                            duration=dur_sec,
                        )
                        subtitle_clips.append(comment_clip)
                        self.logger.info(f"    Comment {idx} overlay added: start={start_sec:.2f}s, duration={dur_sec:.2f}s")
                except Exception as exc:
                    self.logger.warning(
                        "  Thread screenshots failed (non-fatal): %s", exc
                    )
            else:
                self.logger.info(f"  STEP 3d ►  Capturing Reddit screenshot hook (duration={hook_duration:.2f}s)")
                try:
                    ss_manager = ScreenshotManager(self.config, self.logger)
                    hook_png = ss_manager.capture(story)
                    hook_clip = ss_manager.create_hook_clip(
                        screenshot_path=hook_png,
                        frame_size=frame_size,
                        duration=hook_duration,
                    )
                    subtitle_clips.append(hook_clip)
                    self.logger.info(f"    Hook overlay added ({hook_duration:.2f}s)")
                    
                    # Check if there is an OUTRO block in block_timings (conversational mode)
                    if 'block_timings' in locals() and block_timings:
                        for b_timing in block_timings:
                            if b_timing.get("speaker") == "OUTRO":
                                frame_w, frame_h = frame_size
                                start_sec = b_timing["start_ms"] / 1000.0 / speed_factor
                                end_sec = b_timing["end_ms"] / 1000.0 / speed_factor
                                dur_sec = end_sec - start_sec
                                
                                outro_png = None
                                if pipeline_mode == "riddle":
                                    outro_dir = Path("assets/outro")
                                    if outro_dir.exists():
                                        for ext in ("*.png", "*.jpg", "*.jpeg", "*.PNG", "*.JPG", "*.JPEG"):
                                            img_files = list(outro_dir.glob(ext))
                                            img_files = [f for f in img_files if f.is_file() and f.name.lower() != "readme.md"]
                                            if img_files:
                                                outro_png = img_files[0]
                                                break
                                
                                if not outro_png:
                                    outro_png = ss_manager.capture_outro_card(b_timing["text"])
                                    outro_clip = ss_manager.create_hook_clip(
                                        screenshot_path=outro_png,
                                        frame_size=frame_size,
                                        duration=dur_sec,
                                        fade_duration=min(0.5, dur_sec * 0.15),
                                        ken_burns=False,
                                    )
                                else:
                                    self.logger.info(f"    Custom Riddle Outro card detected: {outro_png.name} (creating full-screen clip)")
                                    from moviepy.editor import ImageClip
                                    import moviepy.video.fx.all as vfx
                                    
                                    outro_clip = ImageClip(str(outro_png)).set_duration(dur_sec)
                                    
                                    # Scale image to cover/fill the frame while preserving aspect ratio
                                    outro_clip = outro_clip.resize(height=frame_h)
                                    if outro_clip.w < frame_w:
                                        outro_clip = outro_clip.resize(width=frame_w)
                                        
                                    outro_clip = outro_clip.set_position(("center", "center"))
                                    
                                    fade_sec = min(0.5, dur_sec * 0.15)
                                    if fade_sec > 0.0:
                                        outro_clip = outro_clip.fx(vfx.fadeout, fade_sec)
                                    
                                outro_clip = outro_clip.set_start(start_sec)
                                subtitle_clips.append(outro_clip)
                                self.logger.info(f"    Conversational Outro card overlay added: start={start_sec:.2f}s, duration={dur_sec:.2f}s")
                except Exception as exc:
                    self.logger.warning(
                        "  Screenshot hook failed (non-fatal): %s", exc
                    )

        # ── Step 4: Video Processing ─────────────────────────────────
        pipeline_mode = self.config.get("pipeline", {}).get("pipeline_mode", "monologue")
        use_static_bg = self.config.get("conversational", {}).get("use_static_backgrounds", False)

        if pipeline_mode == "conversational" and use_static_bg:
            self.logger.info("  STEP 4 ►  Skipping gameplay video processing (using static backgrounds)")
            processed_gameplay = None
        else:
            self.logger.info("  STEP 4 ►  Processing gameplay background (Multi-Clip Stitching)")
            downloader = GameplayDownloader(self.config, self.logger)
            hash_pipeline = HashDestructionPipeline(self.config, self.logger)

            # Get audio duration for matching gameplay length
            from pydub import AudioSegment

            audio_seg = AudioSegment.from_file(str(processed_audio))
            audio_duration_sec = len(audio_seg) / 1000.0
            target_gameplay_duration = audio_duration_sec + 2.0  # small buffer

            multi_clip_enabled = self.config.get("video", {}).get("multi_clip_stitching", True)
            segment_dur = float(self.config.get("video", {}).get("clip_segment_duration", 20.0))

            processed_gameplay = self.temp_dir / f"gameplay_processed_p{part['part_number']}.mp4"

            if multi_clip_enabled:
                try:
                    clips_plan = downloader.get_stitched_gameplay_plan(
                        target_duration=target_gameplay_duration,
                        segment_duration=segment_dur,
                    )
                except FileNotFoundError:
                    self.logger.info("    No gameplay files found locally. Attempting to download from configured sources...")
                    downloader.download_all()
                    clips_plan = downloader.get_stitched_gameplay_plan(
                        target_duration=target_gameplay_duration,
                        segment_duration=segment_dur,
                    )

                hash_pipeline.process_multi_clip(
                    clips=clips_plan,
                    target_duration=target_gameplay_duration,
                    output_path=processed_gameplay,
                )
            else:
                try:
                    gameplay_source = downloader.get_random_gameplay()
                except FileNotFoundError:
                    self.logger.info("    No gameplay files found locally. Attempting to download from configured sources...")
                    downloader.download_all()
                    gameplay_source = downloader.get_random_gameplay()
                self.logger.info(f"    Source gameplay: {gameplay_source.name}")

                hash_pipeline.process(
                    input_path=gameplay_source,
                    target_duration=target_gameplay_duration,
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

        dialogue_timings = None
        if pipeline_mode in ("conversational", "riddle") and 'block_timings' in locals():
            dialogue_timings = [
                {
                    "start": b["start_ms"] / 1000.0 / speed_factor,
                    "end": b["end_ms"] / 1000.0 / speed_factor,
                    "speaker": b["speaker"],
                    "emotion": b.get("emotion", "talking"),
                    "text": b.get("text", "")
                }
                for b in block_timings
            ]

        final_path = compositor.compose(
            gameplay_path=processed_gameplay,
            audio_path=processed_audio,
            subtitle_clips=subtitle_clips,
            part_info=part_info,
            word_timestamps=ts_data_final["words"],
            dialogue_timings=dialogue_timings,
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

        # TTS engine and voice check
        tts_cfg = self.config.get("tts", {})
        tts_engine = tts_cfg.get("engine", "kokoro")
        tts_voice = tts_cfg.get("voice", "am_adam")
        self.logger.info(f"  ✓ TTS Engine configured: {tts_engine} (voice: {tts_voice})")

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


def run_kokoro_emotion_test(config: Dict, logger: BotLogger) -> None:
    """Run Kokoro-82M Voice & Emotion Tester mode (audio output only)."""
    logger.info("=" * 60)
    logger.info("  Kokoro-82M Voice & Emotion Test Suite (Audio Output Only)")
    logger.info("=" * 60)

    # Force tts engine to kokoro in config copy
    test_cfg = dict(config)
    test_cfg["tts"] = dict(test_cfg.get("tts", {}))
    test_cfg["tts"]["engine"] = "kokoro"
    
    tts = TTSEngine(test_cfg, logger)

    output_dir = resolve_path("output/kokoro_test", create=True)
    logger.info(f"Test audio output directory: {output_dir}")

    samples = [
        {
            "filename": "01_female_joy_laughter.wav",
            "voice": "af_bella",
            "speaker_name": "Bella (Female Lead)",
            "emotion": "Joy & Laughter",
            "text": "Haha! Oh my goodness, I can't stop laughing! That was honestly the funniest story I've ever heard! Haha!"
        },
        {
            "filename": "02_male_angry_furious.wav",
            "voice": "am_adam",
            "speaker_name": "Adam (Male Lead)",
            "emotion": "Anger & Frustration",
            "text": "Are you serious right now?! I told you a hundred times not to touch my things! How could you be so careless?!"
        },
        {
            "filename": "03_old_male_dramatic_pause.wav",
            "voice": "bm_george",
            "speaker_name": "George (Old Male)",
            "emotion": "Wisdom & Dramatic Suspense",
            "text": "Listen to me closely... In my seventy years on this earth... I have never seen anything... quite as extraordinary as this."
        },
        {
            "filename": "04_old_female_whisper_concern.wav",
            "voice": "af_nicole",
            "speaker_name": "Nicole (Old Female)",
            "emotion": "Whisper & Soft Concern",
            "text": "Psst... keep your voice down. Be very quiet... We don't want anyone to discover what happened here."
        },
        {
            "filename": "05_child_male_excited.wav",
            "voice": "am_michael",
            "speaker_name": "Michael (Youthful/Child Male)",
            "emotion": "Childish Excitement",
            "text": "Yay! Look at that giant rocket ship! Can we go to space right now? Please, please, please?!"
        },
        {
            "filename": "06_child_female_happy_sweet.wav",
            "voice": "af_sky",
            "speaker_name": "Sky (Child Female)",
            "emotion": "Happy & Cheerful",
            "text": "I love ice cream and rainbows! Today is the happiest day of my entire life!"
        },
        {
            "filename": "07_multi_speaker_drama.wav",
            "voice": "af_bella",
            "speaker_name": "Multi-Speaker Dialogue Showcase",
            "emotion": "Conversational Drama",
            "dialogue": [
                ("af_bella", "Wait... did you really just say that to my face?"),
                ("am_adam", "I did. And I meant every single word of it, Bella."),
                ("bm_george", "Calm down, both of you! This arguing won't solve anything."),
                ("af_nicole", "Oh dear, my heart... somebody bring me a glass of water."),
                ("am_michael", "Are you guys gonna fight now? Can I get ice cream first?"),
                ("af_sky", "Me too! Strawberry ice cream! Yay!")
            ]
        }
    ]

    from pydub import AudioSegment

    for idx, sample in enumerate(samples, start=1):
        filename = sample["filename"]
        out_file = output_dir / filename

        if "dialogue" in sample:
            logger.info(f"\n[{idx}/{len(samples)}] Synthesizing: {sample['speaker_name']} ({sample['emotion']})")
            combined_audio = AudioSegment.silent(duration=0)
            for spk_voice, text in sample["dialogue"]:
                logger.info(f"  -> Voice '{spk_voice}': \"{text[:40]}...\"")
                seg_path = tts.synthesize(text, voice=spk_voice)
                seg_audio = AudioSegment.from_file(str(seg_path))
                combined_audio += seg_audio + AudioSegment.silent(duration=350)
            
            combined_audio.export(str(out_file), format="wav")
            logger.info(f"  ✓ Saved multi-speaker audio: {out_file.name}")
        else:
            logger.info(f"\n[{idx}/{len(samples)}] Synthesizing: {sample['speaker_name']} - {sample['emotion']}")
            logger.info(f"  Voice: {sample['voice']} | Text: \"{sample['text']}\"")
            raw_path = tts.synthesize(sample["text"], voice=sample["voice"])
            shutil.copy(raw_path, out_file)
            logger.info(f"  ✓ Saved audio: {out_file.name}")

    logger.info("=" * 60)
    logger.info(f"  Kokoro Emotion Test Complete! All audio files saved in:\n  {output_dir}")
    logger.info("=" * 60)


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
        "--mode",
        type=str,
        choices=["monologue", "conversational", "thread", "riddle", "auto_schedule", "kokoro_test"],
        default=None,
        help="Pipeline mode (monologue | conversational | thread | riddle | auto_schedule | kokoro_test)",
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
        choices=["scraper", "tts", "whisper", "video", "kokoro_test"],
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
        "--check-proxies",
        action="store_true",
        help="Test all Facebook Page proxy IP connections and report public egress IPs",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG level logging",
    )

    return parser


def main() -> int:
    """Main entry point."""
    global pipeline_mode

    parser = build_parser()
    args = parser.parse_args()

    # ── Interactive Pipeline Mode Prompt ─────────────────────────────
    # Skip startup prompt if mode, component, dry-run, check-proxies or pytest is active
    in_pytest = "pytest" in sys.modules or any("pytest" in arg for arg in sys.argv)
    if args.mode:
        pipeline_mode = args.mode
    elif not in_pytest and not args.component and not args.dry_run and not args.check_proxies:
        print("Select Pipeline Mode:")
        print("[1] Standard Monologue Mode (Single Narrator Story)")
        print("[2] Conversational Dialogue Mode (Multi-Voice Drama Text Chat)")
        print("[3] AskReddit Thread Mode (Multi-Voice Q&A Compilation with Comment Cards)")
        print("[4] Fun Riddle Mode (Comment-to-Pin System / 30s Loop)")
        print("[5] Batch Hybrid Scheduler Mode (Schedule 42 Reels for monologue/conversational/riddle)")
        print("[6] Kokoro Voice & Emotion Tester Mode (Audio Output Only - Test all voices & emotions)")
        while True:
            try:
                choice = input("Enter choice (1, 2, 3, 4, 5 or 6): ").strip()
                if choice == "1":
                    pipeline_mode = "monologue"
                    break
                elif choice == "2":
                    pipeline_mode = "conversational"
                    break
                elif choice == "3":
                    pipeline_mode = "thread"
                    break
                elif choice == "4":
                    pipeline_mode = "riddle"
                    break
                elif choice == "5":
                    pipeline_mode = "auto_schedule"
                    break
                elif choice == "6":
                    pipeline_mode = "kokoro_test"
                    break
                else:
                    print("Invalid choice. Please enter 1, 2, 3, 4, 5 or 6.")
            except (KeyboardInterrupt, EOFError):
                print("\nDefaulting to Monologue Mode.")
                pipeline_mode = "monologue"
                break
    else:
        pipeline_mode = "monologue"

    # ── Load config ──────────────────────────────────────────────────
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[CRITICAL] {exc}", file=sys.stderr)
        return 1

    # Inject the chosen pipeline mode globally into config
    config["pipeline"]["pipeline_mode"] = pipeline_mode
    if args.mode:
        config["pipeline"]["approve_scripts"] = False

    # Layer settings from config based on mode
    mode_cfg = config.get(pipeline_mode, {})
    if mode_cfg:
        if "subreddits" in mode_cfg:
            config["reddit"]["subreddits"] = mode_cfg["subreddits"]
        if pipeline_mode == "monologue" and "voice" in mode_cfg:
            config["tts"]["voice"] = mode_cfg["voice"]

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

    # ── Proxy verification check ─────────────────────────────────────
    if args.check_proxies:
        FacebookReelsPublisher.verify_all_pages_proxies(config, logger)
        return 0

    # ── Kokoro Emotion Test Mode ────────────────────────────────────
    if pipeline_mode == "kokoro_test" or args.component == "kokoro_test":
        try:
            run_kokoro_emotion_test(config, logger)
            return 0
        except Exception as exc:
            logger.critical(f"Kokoro test mode failed: {exc}")
            logger.debug(traceback.format_exc())
            return 1

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
            elif args.component == "kokoro_test":
                run_kokoro_emotion_test(config, logger)
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
            logger.info("No Reels were generated (pipeline completed cleanly)")
            return 0

        return 0

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    except NoStoriesFoundError as exc:
        logger.info(f"Pipeline finished cleanly: {exc}")
        return 0
    except RuntimeError as exc:
        logger.critical(f"Pipeline failed: {exc}")
        return 1
    except Exception as exc:
        logger.critical(f"Unexpected error: {exc}")
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
