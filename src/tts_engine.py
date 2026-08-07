"""
RedditDaily-Bot — Neural TTS Synthesis Engine
==============================================
Dual-engine Text-to-Speech synthesizer supporting Kokoro-82M (local ultra-realistic engine)
and Edge TTS (Microsoft cloud neural voices) with cadence pre-processing.

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
    """Neural TTS engine supporting Kokoro-82M and Edge-TTS synthesis.

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
        self._engine_type: str = tts_cfg.get("engine", "kokoro").lower()
        self._voice: str = tts_cfg.get("voice", "am_adam" if self._engine_type == "kokoro" else "en-US-ChristopherNeural")
        self._edge_default_voice: str = tts_cfg.get("edge_tts_voice", "en-US-ChristopherNeural")
        self._kokoro_default_voice: str = tts_cfg.get("male_voice", "am_adam")

        # Resolved temp directory
        self._temp_dir: Path = resolve_path(
            self._cfg.get("pipeline", {}).get("temp_dir", "temp"), create=True
        )

        # Kokoro engine instance
        self._kokoro_instance = None
        self._soundfile_module = None

        if self._engine_type == "kokoro":
            self._init_kokoro(tts_cfg)

        self._log.info(
            "TTSEngine ready (engine=%s, voice=%s)", self._engine_type, self._voice
        )

    def _init_kokoro(self, tts_cfg: Dict[str, Any]) -> None:
        """Initialize Kokoro-82M engine with asset check & fallback to Edge-TTS."""
        kokoro_cfg = tts_cfg.get("kokoro", {})
        model_rel_path = kokoro_cfg.get("model_path", "assets/kokoro/kokoro-v0_19.onnx")
        voices_rel_path = kokoro_cfg.get("voices_path", "assets/kokoro/voices.bin")

        model_path = resolve_path(model_rel_path, create=False)
        voices_path = resolve_path(voices_rel_path, create=False)
        if not voices_path.exists():
            # Check for alternative extension in same folder
            alt_bin = model_path.parent / "voices.bin"
            alt_json = model_path.parent / "voices.json"
            if alt_bin.exists():
                voices_path = alt_bin
            elif alt_json.exists():
                voices_path = alt_json

        # 1. Check if kokoro-onnx and soundfile packages are installed
        try:
            import kokoro_onnx
            import soundfile as sf
            self._soundfile_module = sf
        except ImportError:
            self._log.warning("[WARNING] kokoro-onnx package missing, falling back to Edge-TTS")
            self._engine_type = "edge-tts"
            self._voice = self._edge_default_voice
            return

        # 2. Check if model and voices files exist on disk
        if not model_path.exists() or not voices_path.exists():
            self._log.warning("[WARNING] Kokoro assets missing, falling back to Edge-TTS")
            self._engine_type = "edge-tts"
            self._voice = self._edge_default_voice
            return

        # 3. Instantiate Kokoro with CPUExecutionProvider preferred
        try:
            # Force CPUExecutionProvider for compatibility on older Kepler GPUs
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(
                    str(model_path),
                    providers=["CPUExecutionProvider"]
                )
                self._kokoro_instance = kokoro_onnx.Kokoro.from_session(session, str(voices_path))
            except Exception:
                # Fallback to standard Kokoro constructor if from_session is unavailable
                self._kokoro_instance = kokoro_onnx.Kokoro(str(model_path), str(voices_path))
                
            self._log.info("Kokoro-82M TTS engine successfully initialized (CPUExecutionProvider)")
        except Exception as exc:
            self._log.warning(
                "[WARNING] Kokoro initialization failed (%s), falling back to Edge-TTS", exc
            )
            self._engine_type = "edge-tts"
            self._voice = self._edge_default_voice

    # ------------------------------------------------------------------
    # Cadence Pre-processing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess_cadence(text: str) -> str:
        """Restructure punctuation for natural TTS pauses."""
        if not text:
            return text

        processed = text
        processed = re.sub(r"\s*[—–]\s*", " ... ", processed)
        processed = re.sub(r"(?<=\s)-(?=\s)", " ... ", processed)
        processed = re.sub(r"([.!?])\s{2,}", r"\1 ", processed)
        processed = re.sub(r"([.!?])(?=[A-Z])", r"\1 ", processed)
        processed = re.sub(r",\s{2,}", ", ", processed)
        processed = re.sub(r",(?=\S)", ", ", processed)
        processed = re.sub(r"\.{4,}", "...", processed)
        processed = re.sub(r"(\.\.\.\s*){2,}", "... ", processed)
        processed = re.sub(r" {2,}", " ", processed).strip()

        return processed

    # ------------------------------------------------------------------
    # Async Edge-TTS Helper
    # ------------------------------------------------------------------
    async def _async_synthesize_edge(
        self, text: str, output_path: Path, voice: Optional[str] = None
    ) -> None:
        target_voice = voice or self._voice
        communicate = edge_tts.Communicate(text, target_voice)
        await communicate.save(str(output_path))

    # ------------------------------------------------------------------
    # TTS Synthesis Public API
    # ------------------------------------------------------------------
    def synthesize(
        self,
        script: str,
        voice: Optional[str] = None,
    ) -> Path:
        """Synthesise narration audio from a text script using Kokoro or Edge-TTS.

        Args:
            script: The narration script text to synthesise.
            voice: Optional dynamic voice override.

        Returns:
            :class:`~pathlib.Path` to the generated audio file in the
            pipeline's ``temp_dir``.
        """
        if not script or not script.strip():
            raise ValueError("Cannot synthesise an empty script.")

        processed_text = self._preprocess_cadence(script)

        if self._engine_type == "kokoro":
            return self._synthesize_kokoro(processed_text, voice)
        else:
            return self._synthesize_edge_tts(processed_text, voice)

    def _synthesize_kokoro(self, processed_text: str, voice: Optional[str] = None) -> Path:
        """Synthesize audio using Kokoro-82M local engine."""
        char_count = len(processed_text)
        target_voice = voice or self._voice

        # Defensive check: if target_voice looks like an Edge-TTS voice (contains 'Neural'), map or fall back intelligently
        if target_voice and ("Neural" in target_voice or "-" in target_voice):
            lower_v = target_voice.lower()
            if any(k in lower_v for k in ("michelle", "jenny", "aria", "female", "woman", "girl")):
                mapped_voice = self._cfg.get("tts", {}).get("female_voice", "af_bella")
            elif any(k in lower_v for k in ("guy", "christopher", "male", "man", "boy")):
                mapped_voice = self._cfg.get("tts", {}).get("male_voice", "am_adam")
            elif any(k in lower_v for k in ("brian", "elder", "old")):
                mapped_voice = self._cfg.get("tts", {}).get("old_male_voice", "bm_george")
            elif any(k in lower_v for k in ("ana", "child")):
                mapped_voice = self._cfg.get("tts", {}).get("child_female_voice", "af_sky")
            else:
                mapped_voice = self._kokoro_default_voice

            self._log.warning(
                "Voice '%s' is an Edge-TTS voice. Mapped to Kokoro voice '%s'.",
                target_voice,
                mapped_voice,
            )
            target_voice = mapped_voice

        self._log.info(
            "Starting Kokoro-82M TTS synthesis (%d chars, %d words) using voice %s",
            char_count,
            len(processed_text.split()),
            target_voice,
        )

        filename = f"tts_{timestamp_str()}_{uuid.uuid4().hex[:8]}.mp3"
        output_path = self._temp_dir / filename

        try:
            samples, sample_rate = self._kokoro_instance.create(
                processed_text, voice=target_voice, speed=1.0, lang="en-us"
            )
            self._soundfile_module.write(str(output_path), samples, sample_rate)
        except Exception as exc:
            self._log.error("Kokoro-82M TTS synthesis failed: %s", exc)
            raise RuntimeError(f"Kokoro-82M TTS synthesis failed: {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Kokoro TTS output file is empty or was not created.")

        file_size = output_path.stat().st_size
        self._log.info("Kokoro audio saved: %s (%d bytes)", output_path.name, file_size)
        return output_path

    def _synthesize_edge_tts(self, processed_text: str, voice: Optional[str] = None) -> Path:
        """Synthesize audio using Edge-TTS cloud service."""
        char_count = len(processed_text)
        target_voice = voice or self._voice

        # Defensive check: if target_voice doesn't look like a valid Edge-TTS voice name, fall back.
        if target_voice and ("Neural" not in target_voice or "-" not in target_voice):
            fallback_voice = self._edge_default_voice
            self._log.warning(
                "Voice '%s' does not appear to be a valid Edge-TTS voice name. Falling back to default Edge-TTS voice '%s'.",
                target_voice,
                fallback_voice,
            )
            target_voice = fallback_voice

        self._log.info(
            "Starting Edge TTS synthesis (%d chars, %d words) using voice %s",
            char_count,
            len(processed_text.split()),
            target_voice,
        )

        filename = f"tts_{timestamp_str()}_{uuid.uuid4().hex[:8]}.mp3"
        output_path = self._temp_dir / filename

        try:
            asyncio.run(self._async_synthesize_edge(processed_text, output_path, voice=target_voice))
        except Exception as exc:
            self._log.error("Edge TTS synthesis failed: %s", exc)
            raise RuntimeError(f"Edge TTS synthesis failed: {exc}") from exc

        if not output_path.exists() or output_path.stat().st_size == 0:
            raise RuntimeError("Edge TTS output file is empty or was not created.")

        file_size = output_path.stat().st_size
        self._log.info("Edge TTS audio saved: %s (%d bytes)", output_path.name, file_size)
        return output_path
