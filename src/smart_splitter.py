"""
RedditDaily-Bot — Smart Splitter
==================================
Duration-aware script splitter that breaks long narration scripts into
logical chapters at sentence boundaries, and generates watermark overlay
configuration for multi-part Reels.
"""

import re
from typing import Any, Dict, List

from src.utils import BotLogger, estimate_duration_sec, format_duration


class SmartSplitter:
    """Split narration scripts that exceed the configured max duration.

    Parameters
    ----------
    config : dict
        Full application config (uses ``splitter`` section).
    logger : BotLogger
        Structured logger instance.
    """

    def __init__(self, config: Dict[str, Any], logger: BotLogger) -> None:
        self.logger = logger

        splitter_cfg = config.get("splitter", {})
        self.max_duration_sec: float = splitter_cfg.get("max_duration_sec", 180)
        self.wpm: int = splitter_cfg.get("words_per_minute", 155)
        self.min_part_sec: float = splitter_cfg.get("min_part_duration_sec", 45)

        # Watermark template
        self.wm_text_tpl: str = splitter_cfg.get("watermark_text", "Part {n} of {total}")
        self.wm_position: str = splitter_cfg.get("watermark_position", "top-right")
        self.wm_font_size: int = splitter_cfg.get("watermark_font_size", 28)
        self.wm_color: str = splitter_cfg.get("watermark_color", "#FFFFFF")
        self.wm_opacity: float = splitter_cfg.get("watermark_opacity", 0.7)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def split(self, script: str) -> List[Dict[str, Any]]:
        """Split *script* into one or more parts based on estimated duration.

        Parameters
        ----------
        script : str
            The cleaned narration text to evaluate.

        Returns
        -------
        list[dict]
            Each dict contains:
            - ``part_number``  (int)
            - ``total_parts``  (int)
            - ``script_text``  (str)
            - ``estimated_duration``  (float, seconds)
            - ``watermark_config``  (dict | None)
        """
        import json
        is_json = False
        is_monologue_json = False
        dialogue_blocks = []
        parsed_dict = None
        try:
            parsed = json.loads(script)
            if isinstance(parsed, dict):
                parsed_dict = parsed
                if "script" in parsed:
                    if isinstance(parsed["script"], list):
                        dialogue_blocks = parsed["script"]
                        is_json = True
                    elif isinstance(parsed["script"], str):
                        is_monologue_json = True
            elif isinstance(parsed, list) and all(isinstance(x, dict) and "speaker" in x and "text" in x for x in parsed):
                is_json = True
                dialogue_blocks = parsed
        except Exception:
            pass

        if is_json:
            total_dur = sum(estimate_duration_sec(block["text"], self.wpm) for block in dialogue_blocks)
            self.logger.info(
                f"SmartSplitter: estimated duration {format_duration(total_dur)} (Conversational) "
                f"(max {format_duration(self.max_duration_sec)})"
            )

            if total_dur <= self.max_duration_sec:
                self.logger.info("Script fits in a single part — no split needed")
                return [
                    self._build_part(
                        script_text=script,
                        part_number=1,
                        total_parts=1,
                    )
                ]

            # Distribute dialogue blocks across parts
            parts: List[List[Dict[str, Any]]] = []
            current_part: List[Dict[str, Any]] = []
            current_dur: float = 0.0

            for block in dialogue_blocks:
                block_dur = estimate_duration_sec(block["text"], self.wpm)
                if current_dur + block_dur > self.max_duration_sec and current_part:
                    parts.append(current_part)
                    current_part = [block]
                    current_dur = block_dur
                else:
                    current_part.append(block)
                    current_dur += block_dur
            if current_part:
                parts.append(current_part)

            # Build output dicts
            result: List[Dict[str, Any]] = []
            total_parts = len(parts)
            for idx, part_blocks in enumerate(parts, start=1):
                if parsed_dict and ("caption" in parsed_dict or "pinned_comment" in parsed_dict):
                    part_obj = dict(parsed_dict)
                    part_obj["script"] = part_blocks
                    part_script = json.dumps(part_obj)
                else:
                    part_script = json.dumps(part_blocks)
                result.append(
                    self._build_part(
                        script_text=part_script,
                        part_number=idx,
                        total_parts=total_parts,
                    )
                )

            # For logging
            for p in result:
                self.logger.info(
                    f"  Part {p['part_number']}/{p['total_parts']} — "
                    f"{format_duration(p['estimated_duration'])} "
                    f"(Conversational, {len(json.loads(p['script_text']))} blocks)"
                )
            return result

        if is_monologue_json and parsed_dict is not None:
            monologue_text = parsed_dict["script"]
            total_dur = estimate_duration_sec(monologue_text, self.wpm)
            self.logger.info(
                f"SmartSplitter: estimated duration {format_duration(total_dur)} (Monologue JSON) "
                f"(max {format_duration(self.max_duration_sec)})"
            )

            if total_dur <= self.max_duration_sec:
                self.logger.info("Script fits in a single part — no split needed")
                return [
                    self._build_part(
                        script_text=script,
                        part_number=1,
                        total_parts=1,
                    )
                ]

            # Determine how many parts we need
            num_parts = max(2, int(total_dur // self.max_duration_sec) + 1)
            self.logger.info(f"Script requires splitting into ~{num_parts} parts")

            # Split into sentences
            sentences = self._split_sentences(monologue_text)
            if len(sentences) < num_parts:
                self.logger.warning(
                    f"Only {len(sentences)} sentence(s) — cannot split into "
                    f"{num_parts} parts, returning single part"
                )
                return [
                    self._build_part(
                        script_text=script,
                        part_number=1,
                        total_parts=1,
                    )
                ]

            # Distribute sentences across parts
            parts_text = self._distribute_sentences(sentences, num_parts)

            # Build output dicts
            result: List[Dict[str, Any]] = []
            total_parts = len(parts_text)
            for idx, part_txt in enumerate(parts_text, start=1):
                part_script = json.dumps({
                    "caption": parsed_dict.get("caption", ""),
                    "pinned_comment": parsed_dict.get("pinned_comment", ""),
                    "script": part_txt
                })
                result.append(
                    self._build_part(
                        script_text=part_script,
                        part_number=idx,
                        total_parts=total_parts,
                    )
                )

            # Validate no part is too short
            result = self._enforce_min_duration(result)

            for p in result:
                self.logger.info(
                    f"  Part {p['part_number']}/{p['total_parts']} — "
                    f"{format_duration(p['estimated_duration'])} "
                    f"(Monologue JSON, {len(json.loads(p['script_text'])['script'].split())} words)"
                )

            return result

        # Fallback to standard raw text splitting
        total_dur = estimate_duration_sec(script, self.wpm)
        self.logger.info(
            f"SmartSplitter: estimated duration {format_duration(total_dur)} "
            f"(max {format_duration(self.max_duration_sec)})"
        )

        if total_dur <= self.max_duration_sec:
            self.logger.info("Script fits in a single part — no split needed")
            return [
                self._build_part(
                    script_text=script,
                    part_number=1,
                    total_parts=1,
                )
            ]

        # Determine how many parts we need
        num_parts = max(2, int(total_dur // self.max_duration_sec) + 1)
        self.logger.info(f"Script requires splitting into ~{num_parts} parts")

        # Split into sentences
        sentences = self._split_sentences(script)
        if len(sentences) < num_parts:
            # Not enough sentences to split — force single part
            self.logger.warning(
                f"Only {len(sentences)} sentence(s) — cannot split into "
                f"{num_parts} parts, returning single part"
            )
            return [
                self._build_part(
                    script_text=script,
                    part_number=1,
                    total_parts=1,
                )
            ]

        # Distribute sentences across parts
        parts_text = self._distribute_sentences(sentences, num_parts)

        # Build output dicts
        result: List[Dict[str, Any]] = []
        total_parts = len(parts_text)
        for idx, part_script in enumerate(parts_text, start=1):
            result.append(
                self._build_part(
                    script_text=part_script.strip(),
                    part_number=idx,
                    total_parts=total_parts,
                )
            )

        # Validate no part is too short
        result = self._enforce_min_duration(result)

        for p in result:
            self.logger.info(
                f"  Part {p['part_number']}/{p['total_parts']} — "
                f"{format_duration(p['estimated_duration'])} "
                f"({len(p['script_text'].split())} words)"
            )

        return result

    # ------------------------------------------------------------------
    # Sentence splitting (NLTK-free)
    # ------------------------------------------------------------------
    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        """Split text into sentences using regex heuristics.

        Splits on sentence-ending punctuation (.!?) followed by whitespace
        and an uppercase letter, while avoiding false splits on common
        abbreviations (Mr., Mrs., Dr., etc.).
        """
        # Protect abbreviations
        abbr_pattern = r"\b(Mr|Mrs|Ms|Dr|Prof|Jr|Sr|St|vs|etc|Inc|Ltd|Co)\."
        text = re.sub(abbr_pattern, r"\1<PERIOD>", text)

        # Split on .!? followed by space + uppercase (or end of string)
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)

        # Restore abbreviation periods
        sentences = [s.replace("<PERIOD>", ".").strip() for s in raw if s.strip()]

        return sentences

    # ------------------------------------------------------------------
    # Distribute sentences across parts
    # ------------------------------------------------------------------
    def _distribute_sentences(
        self, sentences: List[str], num_parts: int
    ) -> List[str]:
        """Assign sentences to parts so each part is close to the target
        duration (``max_duration_sec``)."""

        target_dur = self.max_duration_sec
        parts: List[List[str]] = []
        current_part: List[str] = []
        current_dur: float = 0.0

        for sentence in sentences:
            sent_dur = estimate_duration_sec(sentence, self.wpm)

            if current_dur + sent_dur > target_dur and current_part:
                # Current part is full — seal it
                parts.append(current_part)
                current_part = [sentence]
                current_dur = sent_dur
            else:
                current_part.append(sentence)
                current_dur += sent_dur

        # Don't forget the last part
        if current_part:
            parts.append(current_part)

        # Join sentences within each part
        return [" ".join(p) for p in parts]

    # ------------------------------------------------------------------
    # Enforce minimum part duration
    # ------------------------------------------------------------------
    def _enforce_min_duration(
        self, parts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge any part shorter than ``min_part_sec`` into its neighbour."""

        if len(parts) <= 1:
            return parts

        merged: List[Dict[str, Any]] = []
        carry: str = ""

        for part in parts:
            if part["estimated_duration"] < self.min_part_sec:
                # Carry this part's text forward
                carry += " " + part["script_text"]
                self.logger.debug(
                    f"  Part {part['part_number']} too short "
                    f"({format_duration(part['estimated_duration'])}) — merging"
                )
            else:
                if carry:
                    part["script_text"] = carry.strip() + " " + part["script_text"]
                    part["estimated_duration"] = estimate_duration_sec(
                        part["script_text"], self.wpm
                    )
                    carry = ""
                merged.append(part)

        # If there's leftover carry, append to last part
        if carry and merged:
            merged[-1]["script_text"] += " " + carry.strip()
            merged[-1]["estimated_duration"] = estimate_duration_sec(
                merged[-1]["script_text"], self.wpm
            )
        elif carry:
            # Edge case: all parts were too short — combine everything
            merged.append(
                self._build_part(
                    script_text=carry.strip(),
                    part_number=1,
                    total_parts=1,
                )
            )

        # Renumber parts
        total = len(merged)
        for idx, part in enumerate(merged, start=1):
            part["part_number"] = idx
            part["total_parts"] = total
            if total == 1:
                part["watermark_config"] = None
            else:
                part["watermark_config"] = self._make_watermark_config(idx, total)

        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_part(
        self,
        script_text: str,
        part_number: int,
        total_parts: int,
    ) -> Dict[str, Any]:
        """Build a single part dictionary."""
        import json
        is_json = False
        try:
            parsed = json.loads(script_text)
            if isinstance(parsed, list):
                is_json = True
                est_dur = sum(estimate_duration_sec(block["text"], self.wpm) for block in parsed)
        except Exception:
            pass

        if not is_json:
            est_dur = estimate_duration_sec(script_text, self.wpm)

        watermark = (
            self._make_watermark_config(part_number, total_parts)
            if total_parts > 1
            else None
        )
        return {
            "part_number": part_number,
            "total_parts": total_parts,
            "script_text": script_text,
            "estimated_duration": est_dur,
            "watermark_config": watermark,
        }

    def _make_watermark_config(
        self, part_number: int, total_parts: int
    ) -> Dict[str, Any]:
        """Create the watermark overlay configuration dict."""
        return {
            "text": self.wm_text_tpl.format(n=part_number, total=total_parts),
            "position": self.wm_position,
            "font_size": self.wm_font_size,
            "color": self.wm_color,
            "opacity": self.wm_opacity,
        }
