"""
RedditDaily-Bot — Whisper Timestamps & Vosk Fallback
=====================================================
Extracts word-level timestamps from narration audio using
whisper-timestamped (GPU) with automatic Vosk (CPU) fallback
when CUDA VRAM is insufficient.

Public API
----------
    extractor = TimestampExtractor(config)
    result    = extractor.extract(audio_path)
    extractor.save_timestamps(result, output_path)
"""

from __future__ import annotations

import json
import re
import struct
import wave
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils import BotLogger, resolve_path, save_json

# Sentence-ending punctuation pattern
_SENTENCE_END_RE = re.compile(r"[.!?]$")


class TimestampExtractor:
    """Extract word-level timestamps from audio via Whisper or Vosk.

    The extractor automatically selects the best engine based on
    available GPU VRAM.  When VRAM is below the configured threshold
    (``whisper.min_vram_gb``, default 5 GB) it falls back to the
    Vosk CPU recogniser.

    Args:
        config: Parsed configuration dictionary (from ``load_config``).
        logger: Optional pre-configured ``BotLogger``.  A default one
            is created when not supplied.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[BotLogger] = None,
    ) -> None:
        self.config = config
        whisper_cfg = config.get("whisper", {})
        pipeline_cfg = config.get("pipeline", {})

        self.model_size: str = whisper_cfg.get("model_size", "medium")
        self.language: str = whisper_cfg.get("language", "en")
        self.min_vram_gb: float = float(whisper_cfg.get("min_vram_gb", 5.0))
        self.vosk_model_name: str = whisper_cfg.get(
            "vosk_model", "vosk-model-small-en-us-0.15"
        )
        self.vosk_model_path: Path = resolve_path(
            whisper_cfg.get("vosk_model_path", "assets/vosk_model"),
            create=True,
        )

        self.logger = logger or BotLogger(
            name="TimestampExtractor",
            log_dir=pipeline_cfg.get("log_dir", "output/logs"),
            level=pipeline_cfg.get("log_level", "INFO"),
        )

    # ------------------------------------------------------------------
    # VRAM detection
    # ------------------------------------------------------------------
    @staticmethod
    def _check_vram() -> float:
        """Return available GPU VRAM in gigabytes.

        Uses ``torch.cuda`` when PyTorch is installed and a CUDA device
        is present.  Returns ``0.0`` otherwise.

        Returns:
            Available VRAM in GB (float).
        """
        try:
            import torch  # type: ignore[import-untyped]

            if torch.cuda.is_available():
                free, _total = torch.cuda.mem_get_info(0)
                return free / (1024 ** 3)
        except (ImportError, RuntimeError, Exception):
            pass
        return 0.0

    # ------------------------------------------------------------------
    # Whisper GPU path
    # ------------------------------------------------------------------
    def _extract_with_whisper(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Transcribe *audio_path* with whisper-timestamped on GPU.

        Args:
            audio_path: Path to the audio file (WAV / MP3 / etc.).

        Returns:
            List of word dicts: ``[{word, start, end, confidence}, ...]``

        Raises:
            RuntimeError: If transcription fails.
        """
        try:
            import whisper_timestamped as whisper  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "whisper-timestamped is not installed. "
                "Install it with: pip install whisper-timestamped"
            ) from exc

        self.logger.info(
            "Loading Whisper model '%s' on CUDA …", self.model_size
        )
        model = whisper.load_model(self.model_size, device="cuda")

        self.logger.info("Loading audio: %s", audio_path.name)
        audio = whisper.load_audio(str(audio_path))

        self.logger.info("Transcribing with whisper-timestamped …")
        result = whisper.transcribe(
            model, audio, language=self.language
        )

        words: List[Dict[str, Any]] = []
        for segment in result.get("segments", []):
            for w in segment.get("words", []):
                words.append(
                    {
                        "word": w.get("text", "").strip(),
                        "start": float(w.get("start", 0.0)),
                        "end": float(w.get("end", 0.0)),
                        "confidence": float(w.get("confidence", 0.0)),
                    }
                )

        self.logger.info(
            "Whisper extracted %d words from '%s'.",
            len(words),
            audio_path.name,
        )
        return words

    # ------------------------------------------------------------------
    # Vosk CPU fallback
    # ------------------------------------------------------------------
    def _ensure_vosk_model(self) -> Path:
        """Download the Vosk model if it is not already present.

        The model archive is fetched from the official alphacephei
        mirror and extracted into ``vosk_model_path``.

        Returns:
            Path to the extracted model directory.

        Raises:
            RuntimeError: If download or extraction fails.
        """
        model_dir = self.vosk_model_path / self.vosk_model_name
        if model_dir.exists() and any(model_dir.iterdir()):
            self.logger.debug("Vosk model found at %s", model_dir)
            return model_dir

        import io
        import urllib.request
        import zipfile

        url = (
            f"https://alphacephei.com/vosk/models/"
            f"{self.vosk_model_name}.zip"
        )
        self.logger.info("Downloading Vosk model from %s …", url)
        try:
            resp = urllib.request.urlopen(url, timeout=300)  # noqa: S310
            data = resp.read()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download Vosk model from {url}: {exc}"
            ) from exc

        self.logger.info("Extracting Vosk model …")
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(str(self.vosk_model_path))
        except zipfile.BadZipFile as exc:
            raise RuntimeError(
                f"Downloaded Vosk archive is corrupt: {exc}"
            ) from exc

        if not model_dir.exists():
            # Some archives unzip to a slightly different name — find it
            candidates = [
                d
                for d in self.vosk_model_path.iterdir()
                if d.is_dir() and d.name.startswith("vosk-model")
            ]
            if candidates:
                model_dir = candidates[0]
            else:
                raise RuntimeError(
                    "Vosk model extraction succeeded but the model "
                    f"directory was not found in {self.vosk_model_path}"
                )

        self.logger.info("Vosk model ready at %s", model_dir)
        return model_dir

    def _convert_to_wav16k(self, audio_path: Path) -> Path:
        """Convert *audio_path* to 16 kHz mono WAV for Vosk.

        Uses **pydub** with the system's ffmpeg backend.

        Args:
            audio_path: Source audio file.

        Returns:
            Path to the 16 kHz mono WAV file (in the same directory).

        Raises:
            RuntimeError: If pydub / ffmpeg conversion fails.
        """
        try:
            from pydub import AudioSegment  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "pydub is required for Vosk audio conversion. "
                "Install it with: pip install pydub"
            ) from exc

        wav_path = audio_path.with_suffix(".vosk.wav")
        if wav_path.exists():
            self.logger.debug("Reusing existing 16 kHz WAV: %s", wav_path.name)
            return wav_path

        self.logger.info(
            "Converting '%s' → 16 kHz mono WAV for Vosk …",
            audio_path.name,
        )
        try:
            audio = AudioSegment.from_file(str(audio_path))
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            audio.export(str(wav_path), format="wav")
        except Exception as exc:
            raise RuntimeError(
                f"Audio conversion to 16 kHz WAV failed: {exc}"
            ) from exc

        self.logger.debug("WAV written to %s", wav_path)
        return wav_path

    def _extract_with_vosk(self, audio_path: Path) -> List[Dict[str, Any]]:
        """Transcribe *audio_path* with the Vosk offline recogniser.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of word dicts: ``[{word, start, end, confidence}, ...]``

        Raises:
            RuntimeError: If Vosk recognition fails.
        """
        try:
            import vosk  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "vosk is not installed. Install it with: pip install vosk"
            ) from exc

        # Ensure model is available
        model_dir = self._ensure_vosk_model()
        vosk.SetLogLevel(-1)  # suppress Vosk internal logging
        model = vosk.Model(str(model_dir))

        # Ensure audio is 16 kHz mono WAV
        wav_path = self._convert_to_wav16k(audio_path)

        self.logger.info("Running Vosk recogniser on '%s' …", audio_path.name)

        recognizer = vosk.KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)

        words: List[Dict[str, Any]] = []

        try:
            wf = wave.open(str(wav_path), "rb")
        except wave.Error as exc:
            raise RuntimeError(
                f"Failed to open WAV file for Vosk: {exc}"
            ) from exc

        try:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                raise RuntimeError(
                    "Vosk requires 16 kHz mono 16-bit PCM WAV. "
                    "The converted file does not meet this requirement."
                )

            chunk_size = 4000  # bytes (~0.125 s at 16 kHz / 16-bit)
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data)

        finally:
            wf.close()

        # Collect final result
        final_json = json.loads(recognizer.FinalResult())
        self._collect_vosk_words(final_json, words)

        self.logger.info(
            "Vosk extracted %d words from '%s'.",
            len(words),
            audio_path.name,
        )
        return words

    @staticmethod
    def _collect_vosk_words(
        result_dict: Dict[str, Any],
        accumulator: List[Dict[str, Any]],
    ) -> None:
        """Parse a Vosk JSON result block into normalised word dicts.

        Args:
            result_dict: Single Vosk result (from ``AcceptWaveform``
                or ``FinalResult``).
            accumulator: List to append word dicts to.
        """
        for entry in result_dict.get("result", []):
            accumulator.append(
                {
                    "word": entry.get("word", "").strip(),
                    "start": float(entry.get("start", 0.0)),
                    "end": float(entry.get("end", 0.0)),
                    "confidence": float(entry.get("conf", 0.0)),
                }
            )

    # ------------------------------------------------------------------
    # Vosk streaming (process all partial results too)
    # ------------------------------------------------------------------
    def _extract_with_vosk_streaming(
        self, audio_path: Path
    ) -> List[Dict[str, Any]]:
        """Full streaming Vosk extraction that collects partial results.

        This is more accurate for long audio because it processes
        intermediate ``AcceptWaveform`` results, not just the final one.

        Args:
            audio_path: Path to the audio file.

        Returns:
            List of word dicts.
        """
        try:
            import vosk  # type: ignore[import-untyped]
        except ImportError as exc:
            raise RuntimeError(
                "vosk is not installed. Install it with: pip install vosk"
            ) from exc

        model_dir = self._ensure_vosk_model()
        vosk.SetLogLevel(-1)
        model = vosk.Model(str(model_dir))

        wav_path = self._convert_to_wav16k(audio_path)

        self.logger.info(
            "Running Vosk streaming recogniser on '%s' …", audio_path.name
        )

        recognizer = vosk.KaldiRecognizer(model, 16000)
        recognizer.SetWords(True)

        words: List[Dict[str, Any]] = []

        try:
            wf = wave.open(str(wav_path), "rb")
        except wave.Error as exc:
            raise RuntimeError(
                f"Failed to open WAV file for Vosk: {exc}"
            ) from exc

        try:
            chunk_size = 4000
            while True:
                data = wf.readframes(chunk_size)
                if len(data) == 0:
                    break
                if recognizer.AcceptWaveform(data):
                    partial = json.loads(recognizer.Result())
                    self._collect_vosk_words(partial, words)

        finally:
            wf.close()

        # Final utterance
        final_json = json.loads(recognizer.FinalResult())
        self._collect_vosk_words(final_json, words)

        self.logger.info(
            "Vosk (streaming) extracted %d words from '%s'.",
            len(words),
            audio_path.name,
        )
        return words

    # ------------------------------------------------------------------
    # Sentence grouping
    # ------------------------------------------------------------------
    @staticmethod
    def _group_into_sentences(
        words: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Group a flat word list into sentences.

        Sentence boundaries are determined by trailing punctuation
        (``'.'``, ``'!'``, ``'?'``) on a word's text.  If no
        punctuation is found after the final word, the remaining words
        form the last sentence.

        Args:
            words: Flat list of ``{word, start, end, confidence}`` dicts.

        Returns:
            List of sentence dicts::

                {
                    'text': str,
                    'start': float,
                    'end': float,
                    'words': [...]
                }
        """
        if not words:
            return []

        sentences: List[Dict[str, Any]] = []
        current_words: List[Dict[str, Any]] = []

        for w in words:
            current_words.append(w)
            if _SENTENCE_END_RE.search(w["word"]):
                sentences.append(_build_sentence(current_words))
                current_words = []

        # Remaining words that didn't end with punctuation
        if current_words:
            sentences.append(_build_sentence(current_words))

        return sentences

    # ------------------------------------------------------------------
    # Audio duration helper
    # ------------------------------------------------------------------
    @staticmethod
    def _get_audio_duration(audio_path: Path) -> float:
        """Return the duration of an audio file in seconds.

        Attempts to use the last word's end-time if available, then
        falls back to ``pydub`` or ``wave`` for measurement.

        Args:
            audio_path: Path to the audio file.

        Returns:
            Duration in seconds.
        """
        # WAV fast-path
        if audio_path.suffix.lower() == ".wav":
            try:
                with wave.open(str(audio_path), "rb") as wf:
                    frames = wf.getnframes()
                    rate = wf.getframerate()
                    if rate > 0:
                        return frames / rate
            except Exception:
                pass

        # pydub fallback (supports MP3, OGG, etc.)
        try:
            from pydub import AudioSegment  # type: ignore[import-untyped]

            audio = AudioSegment.from_file(str(audio_path))
            return len(audio) / 1000.0
        except Exception:
            pass

        return 0.0

    # ------------------------------------------------------------------
    # Main public interface
    # ------------------------------------------------------------------
    def extract(self, audio_path: Path) -> Dict[str, Any]:
        """Extract word-level timestamps from *audio_path*.

        Automatically selects Whisper (GPU) or Vosk (CPU) based on
        available VRAM.

        Args:
            audio_path: Path to the narration audio file.

        Returns:
            Dictionary with keys:

            - ``words``: flat list of word dicts.
            - ``sentences``: list of sentence dicts.
            - ``engine``: ``'whisper'`` or ``'vosk'``.
            - ``duration``: total audio duration in seconds.

        Raises:
            FileNotFoundError: If *audio_path* does not exist.
            RuntimeError: If both engines fail.
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        vram = self._check_vram()
        self.logger.info("Available GPU VRAM: %.2f GB", vram)

        engine: str = ""
        words: List[Dict[str, Any]] = []

        # ---- Try Whisper (GPU) first ---------------------------------
        if vram >= self.min_vram_gb:
            self.logger.info(
                "VRAM (%.2f GB) ≥ threshold (%.1f GB) → using Whisper.",
                vram,
                self.min_vram_gb,
            )
            try:
                words = self._extract_with_whisper(audio_path)
                engine = "whisper"
            except Exception as exc:
                self.logger.warning(
                    "Whisper extraction failed (%s). Falling back to Vosk.",
                    exc,
                )
        else:
            self.logger.info(
                "VRAM (%.2f GB) < threshold (%.1f GB) → using Vosk CPU.",
                vram,
                self.min_vram_gb,
            )

        # ---- Vosk fallback -------------------------------------------
        if not words:
            try:
                words = self._extract_with_vosk_streaming(audio_path)
                engine = "vosk"
            except Exception as exc:
                self.logger.error("Vosk extraction also failed: %s", exc)
                raise RuntimeError(
                    f"Both Whisper and Vosk failed for '{audio_path}': {exc}"
                ) from exc

        # ---- Post-process --------------------------------------------
        sentences = self._group_into_sentences(words)

        # Duration: prefer last word end, fall back to file measurement
        duration = 0.0
        if words:
            duration = max(w["end"] for w in words)
        if duration <= 0.0:
            duration = self._get_audio_duration(audio_path)

        result: Dict[str, Any] = {
            "words": words,
            "sentences": sentences,
            "engine": engine,
            "duration": duration,
        }

        self.logger.info(
            "Timestamp extraction complete — engine=%s, words=%d, "
            "sentences=%d, duration=%.2fs",
            engine,
            len(words),
            len(sentences),
            duration,
        )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_timestamps(
        self, data: Dict[str, Any], output_path: Path
    ) -> None:
        """Persist timestamp data to a JSON file.

        Args:
            data: The dictionary returned by :meth:`extract`.
            output_path: Destination file path.
        """
        output_path = Path(output_path)
        save_json(data, str(output_path))
        self.logger.info("Timestamps saved to %s", output_path)


# ======================================================================
# Module-level helpers
# ======================================================================

def _build_sentence(words: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build a sentence dict from a list of word dicts.

    Args:
        words: Non-empty list of word dicts belonging to one sentence.

    Returns:
        Sentence dict with ``text``, ``start``, ``end``, and ``words``.
    """
    text = " ".join(w["word"] for w in words)
    return {
        "text": text,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "words": list(words),
    }
