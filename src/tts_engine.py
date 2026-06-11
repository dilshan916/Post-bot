"""
RedditDaily-Bot — Neural TTS Synthesis Engine
==============================================
Edge TTS with free natural voices, cadence pre-processing, and streamed MP3 generation.

Component 2 of the RedditDaily-Bot pipeline.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import edge_tts

from src.utils import (
    BotLogger,
    load_config,
    resolve_path,
    timestamp_str,
)


class TTSEngine:
    """Edge TTS engine with cadence pre-processing.

    Args:
        config: Parsed configuration dictionary. If *None*, loads from the
            default ``config.yaml`` path.
        logger: Optional pre-built :class:`BotLogger`. A module-scoped
            logger is created when omitted.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[BotLogger] = None,
    ) -> None:
        self._cfg: Dict[str, Any] = config or load_config()
        self._log: BotLogger = logger or BotLogger(
            name="TTSEngine",
            log_dir=self._cfg.get("pipeline", {}).get("log_dir"),
            level=self._cfg.get("pipeline", {}).get("log_level", "INFO"),
        )

        # ---- TTS sub-config ----
        tts_cfg: Dict[str, Any] = self._cfg.get("tts", {})
        self._voice: str = tts_cfg.get("voice", "en-US-ChristopherNeural")

        # Resolved temp directory
        self._temp_dir: Path = resolve_path(
            self._cfg.get("pipeline", {}).get("temp_dir", "temp"), create=True
        )

        self._log.info("TTSEngine ready (voice=%s)", self._voice)

    # ------------------------------------------------------------------
    # Cadence Pre-processing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_cadence(text: str) -> str:
        """Restructure punctuation so Edge-TTS produces natural pauses.

        Transformations applied:

        * **Em/en-dashes & hyphens used as pauses** → replaced with
          ``' ... '`` (ellipsis), which the engine interprets as a suspense pause.
        * **Periods** → normalised to exactly one space after a period so
          the engine renders a consistent sentence break.
        * **Commas** → left as-is; the engine already produces a pause.

        Args:
            text: Raw narration script.

        Returns:
            Script with adjusted punctuation for optimal TTS cadence.
        """
        if not text:
            return text

        processed = text

        # 1. Replace em-dash (—), en-dash (–), and isolated hyphens used as
        #    pauses with ellipsis for suspense pauses.
        processed = re.sub(r"\s*[—–]\s*", " ... ", processed)
        processed = re.sub(r"(?<=\s)-(?=\s)", " ... ", processed)

        # 2. Normalise spacing after periods.
        processed = re.sub(r"([.!?])\s{2,}", r"\1 ", processed)
        processed = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", processed)

        # 3. Ensure commas are followed by a single space.
        processed = re.sub(r",\s{2,}", ", ", processed)
        processed = re.sub(r",(?=\S)", ", ", processed)

        # 4. Clean up any triple-or-more dots that aren't exactly three
        processed = re.sub(r"\.{4,}", "...", processed)

        # 5. Collapse multiple consecutive ellipses from dash replacements
        processed = re.sub(r"(\.\.\.\s*){2,}", "... ", processed)

        # 6. Strip leading/trailing whitespace; collapse internal runs
        processed = re.sub(r" {2,}", " ", processed).strip()

        return processed

    # ------------------------------------------------------------------
    # Async TTS Helper
    # ------------------------------------------------------------------
    async def _async_synthesize(self, text: str, output_path: Path, voice: Optional[str] = None) -> None:
        target_voice = voice or self._voice
        communicate = edge_tts.Communicate(text, target_voice)
        await communicate.save(str(output_path))

    # ------------------------------------------------------------------
    # TTS Synthesis
    # ------------------------------------------------------------------
    def synthesize(self, script: str, voice: Optional[str] = None) -> Path:
        """Synthesise narration audio from a text script using edge-tts.

        Args:
            script: The narration script text to synthesise.
            voice: Optional dynamic voice override.

        Returns:
            :class:`~pathlib.Path` to the generated MP3 audio file in the
            pipeline's ``temp_dir``.
        """
        if not script or not script.strip():
            raise ValueError("Cannot synthesise an empty script.")

        processed_text = self._preprocess_cadence(script)
        char_count = len(processed_text)
        target_voice = voice or self._voice
        self._log.info(
            "Starting Edge TTS synthesis (%d chars, %d words) using voice %s",
            char_count,
            len(processed_text.split()),
            target_voice,
        )

        filename = f"tts_{timestamp_str()}_{uuid.uuid4().hex[:8]}.mp3"
        output_path = self._temp_dir / filename

        try:
            asyncio.run(self._async_synthesize(processed_text, output_path, voice=voice))
        except Exception as exc:
            self._log.error("Edge TTS synthesis failed: %s", exc)
            raise RuntimeError(f"Edge TTS synthesis failed: {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Edge TTS output file is empty or was not created.")

        file_size = output_path.stat().st_size
        self._log.info(
            "TTS audio saved: %s (%d bytes)", output_path.name, file_size
        )
        return output_path
