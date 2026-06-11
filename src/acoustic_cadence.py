"""
RedditDaily-Bot — Acoustic Cadence Processor
==============================================
Implements the "Acoustic Cadence Hack":
  1.  Appends a tail phrase ("THE END.") to the script before TTS.
  2.  After TTS + Whisper, locates the tail phrase via word timestamps.
  3.  Slices out the tail phrase audio and replaces it with low-amplitude
      room-tone noise to simulate a natural human pause at the end.

Also provides a duration-based fallback for when Whisper timestamps
are not yet available.
"""

import struct
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from pydub import AudioSegment

from src.utils import BotLogger, estimate_duration_sec, resolve_path


class AcousticCadenceProcessor:
    """Process TTS audio to implement natural vocal decay and room tone.

    Parameters
    ----------
    config : dict
        Full application config (uses ``cadence`` and ``pipeline`` sections).
    logger : BotLogger
        Structured logger instance.
    """

    def __init__(self, config: Dict[str, Any], logger: BotLogger) -> None:
        self.logger = logger
        self.config = config

        cadence_cfg = config.get("cadence", {})
        self.tail_phrase: str = cadence_cfg.get("tail_phrase", "THE END.")
        self.room_tone_duration_sec: float = cadence_cfg.get(
            "room_tone_duration_sec", 1.2
        )
        self.room_tone_amplitude: float = cadence_cfg.get(
            "room_tone_amplitude", 0.005
        )

        self.temp_dir = resolve_path(
            config.get("pipeline", {}).get("temp_dir", "temp"), create=True
        )

    # ------------------------------------------------------------------
    # Public: Prepare script with tail phrase
    # ------------------------------------------------------------------
    def prepare_script_with_tail(self, script: str) -> str:
        """Append the tail phrase to the end of the script.

        The tail phrase (default ``THE END.``) forces the neural TTS
        engine to produce a natural emotional vocal decay.  After
        generation the tail segment is sliced out and replaced with
        room tone.

        Parameters
        ----------
        script : str
            The narration script text.

        Returns
        -------
        str
            Script with appended tail phrase.
        """
        script = script.rstrip()
        if not script.endswith((".", "!", "?")):
            script += "."

        # Separate with a clear sentence break for the TTS engine
        result = f"{script} {self.tail_phrase}"
        self.logger.info(
            f"Appended tail phrase '{self.tail_phrase}' to script"
        )
        return result

    # ------------------------------------------------------------------
    # Public: Process audio with Whisper timestamps (primary path)
    # ------------------------------------------------------------------
    def process_audio(
        self,
        audio_path: Path,
        word_timestamps: List[Dict[str, Any]],
    ) -> Path:
        """Trim the tail phrase from TTS audio using word-level timestamps
        and replace it with room-tone noise.

        Parameters
        ----------
        audio_path : Path
            Path to the raw TTS audio file (MP3 or WAV).
        word_timestamps : list[dict]
            Whisper / Vosk word-level timestamps.  Each dict must have
            ``word`` (str), ``start`` (float, seconds), ``end`` (float).

        Returns
        -------
        Path
            Path to the processed audio file with tail replaced by room tone.
        """
        self.logger.info("Acoustic cadence: locating tail phrase in timestamps")

        # ── Find tail phrase words ───────────────────────────────────
        tail_words = self.tail_phrase.replace(".", "").strip().upper().split()
        tail_start_sec = self._find_tail_start(word_timestamps, tail_words)

        if tail_start_sec is None:
            self.logger.warning(
                "Could not locate tail phrase in timestamps — "
                "falling back to duration-based trim"
            )
            return self.trim_tail_by_duration(
                audio_path, self.tail_phrase
            )

        self.logger.info(
            f"Tail phrase starts at {tail_start_sec:.2f}s — slicing"
        )

        # ── Load audio ───────────────────────────────────────────────
        audio = AudioSegment.from_file(str(audio_path))
        tail_start_ms = int(tail_start_sec * 1000)

        # Slice: keep everything before the tail
        main_audio = audio[:tail_start_ms]

        # ── Generate room tone ───────────────────────────────────────
        room_tone_ms = int(self.room_tone_duration_sec * 1000)
        room_tone = self._generate_room_tone(
            duration_ms=room_tone_ms,
            sample_rate=audio.frame_rate,
            amplitude=self.room_tone_amplitude,
        )

        # ── Concatenate ──────────────────────────────────────────────
        final_audio = main_audio + room_tone

        # ── Export ────────────────────────────────────────────────────
        output_path = self.temp_dir / f"cadence_{audio_path.stem}.mp3"
        final_audio.export(str(output_path), format="mp3", bitrate="128k")

        orig_dur = len(audio) / 1000.0
        new_dur = len(final_audio) / 1000.0
        self.logger.info(
            f"Acoustic cadence applied: {orig_dur:.1f}s → {new_dur:.1f}s "
            f"(tail trimmed, {self.room_tone_duration_sec}s room tone added)"
        )

        return output_path

    # ------------------------------------------------------------------
    # Public: Duration-based fallback trim
    # ------------------------------------------------------------------
    def trim_tail_by_duration(
        self,
        audio_path: Path,
        tail_phrase: Optional[str] = None,
        wpm: int = 155,
    ) -> Path:
        """Trim the tail phrase by estimating its spoken duration.

        Used as a fallback when word timestamps are not available.

        Parameters
        ----------
        audio_path : Path
            Path to the TTS audio file.
        tail_phrase : str | None
            The tail phrase appended to the script.  Defaults to
            the configured ``cadence.tail_phrase``.
        wpm : int
            Words per minute for duration estimation.

        Returns
        -------
        Path
            Path to the processed audio file.
        """
        phrase = tail_phrase or self.tail_phrase
        est_dur_sec = estimate_duration_sec(phrase, wpm)
        # Add a small buffer (pauses around the phrase)
        trim_sec = est_dur_sec + 0.5

        self.logger.info(
            f"Duration-based trim: removing last {trim_sec:.2f}s "
            f"(estimated for '{phrase}')"
        )

        audio = AudioSegment.from_file(str(audio_path))
        trim_ms = int(trim_sec * 1000)

        if trim_ms >= len(audio):
            self.logger.warning(
                "Trim duration exceeds audio length — skipping trim"
            )
            return audio_path

        main_audio = audio[: len(audio) - trim_ms]

        # Append room tone
        room_tone_ms = int(self.room_tone_duration_sec * 1000)
        room_tone = self._generate_room_tone(
            duration_ms=room_tone_ms,
            sample_rate=audio.frame_rate,
            amplitude=self.room_tone_amplitude,
        )
        final_audio = main_audio + room_tone

        output_path = self.temp_dir / f"cadence_fb_{audio_path.stem}.mp3"
        final_audio.export(str(output_path), format="mp3", bitrate="128k")

        self.logger.info(
            f"Fallback cadence applied: "
            f"{len(audio)/1000:.1f}s → {len(final_audio)/1000:.1f}s"
        )
        return output_path

    # ------------------------------------------------------------------
    # Private: Find tail phrase start time in timestamps
    # ------------------------------------------------------------------
    @staticmethod
    def _find_tail_start(
        words: List[Dict[str, Any]],
        tail_words: List[str],
    ) -> Optional[float]:
        """Search the word timestamp list from the end to find where the
        tail phrase begins.

        Parameters
        ----------
        words : list[dict]
            Word-level timestamps with ``word`` and ``start`` keys.
        tail_words : list[str]
            Uppercased words of the tail phrase (e.g. ["THE", "END"]).

        Returns
        -------
        float | None
            Start time in seconds of the first tail word, or None if
            the phrase was not found.
        """
        if not words or not tail_words:
            return None

        num_tail = len(tail_words)
        # Search backwards for a match
        for i in range(len(words) - num_tail, -1, -1):
            candidate = [
                w["word"].strip().strip(".,!?;:").upper()
                for w in words[i : i + num_tail]
            ]
            if candidate == tail_words:
                return words[i]["start"]

        return None

    # ------------------------------------------------------------------
    # Private: Generate room-tone noise
    # ------------------------------------------------------------------
    @staticmethod
    def _generate_room_tone(
        duration_ms: int,
        sample_rate: int = 44100,
        amplitude: float = 0.005,
    ) -> AudioSegment:
        """Create a low-amplitude white-noise audio segment that simulates
        natural room tone / recording silence.

        Parameters
        ----------
        duration_ms : int
            Duration in milliseconds.
        sample_rate : int
            Audio sample rate (Hz).
        amplitude : float
            Peak amplitude of the noise (0.0–1.0 scale).

        Returns
        -------
        AudioSegment
            Mono 16-bit audio segment containing room-tone noise.
        """
        num_samples = int(sample_rate * duration_ms / 1000)

        # Generate Gaussian noise scaled to 16-bit range
        noise = np.random.normal(0, amplitude, num_samples)
        noise = np.clip(noise, -1.0, 1.0)

        # Convert to 16-bit PCM
        pcm_data = (noise * 32767).astype(np.int16)
        raw_bytes = pcm_data.tobytes()

        # Build AudioSegment from raw PCM
        room_tone = AudioSegment(
            data=raw_bytes,
            sample_width=2,       # 16-bit
            frame_rate=sample_rate,
            channels=1,           # mono
        )

        return room_tone
