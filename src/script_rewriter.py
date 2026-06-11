"""
RedditDaily-Bot — Script Rewriter
==================================
Transforms raw Reddit self-text posts into clean, spoken-word narration
scripts suitable for TTS.  Uses a regex-based cleanup pipeline (no LLM
required) while keeping the class interface extensible for future LLM
integration.
"""

from __future__ import annotations

import random
import re
import time
import unicodedata
from typing import Any, Dict, List, Optional

from src.utils import BotLogger, validate_api_key


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_INTRO_VARIATIONS: List[str] = [
    "Here's a story from Reddit.",
    "Listen to this story from Reddit.",
    "Check out this Reddit story.",
    "Here's an interesting post from Reddit.",
    "Someone shared this story on Reddit.",
    "This story was posted on Reddit.",
    "Here's what someone shared on Reddit.",
]

# Reddit artefact patterns (case-insensitive)
_ARTIFACT_PATTERNS: List[re.Pattern[str]] = [
    re.compile(
        r"^[\s]*(?:EDIT|UPDATE|EDITED|EDITING)[\s]*\d*[\s]*:.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"^[\s]*(?:TL;?\s*DR|TLDR)[\s:;\-—]*.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"(?:throwaway\s+(?:account|because)|using\s+a?\s*throwaway)",
        re.IGNORECASE,
    ),
    re.compile(
        r"sorry\s+for\s+(?:the\s+)?(?:formatting|grammar|English)[^.]*\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:i'?m|I am)\s+on\s+mobile[^.]*\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:obligatory\s+)?not\s+(?:a\s+)?native\s+(?:English\s+)?speaker[^.]*\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:first\s+time\s+poster|long\s+time\s+lurker)[^.]*\.?",
        re.IGNORECASE,
    ),
    re.compile(
        r"^[\s]*(?:ETA|FYI|PSA|NTA|YTA|ESH|NAH|INFO)[\s]*:.*$",
        re.IGNORECASE | re.MULTILINE,
    ),
    re.compile(
        r"don'?t\s+(?:repost|share|post)\s+(?:this|my\s+(?:story|post))[^.]*\.?",
        re.IGNORECASE,
    ),
]

# Profanity / abbreviation map for TTS clarity
_TTS_WORD_MAP: Dict[str, str] = {
    "AITA": "Am I the jerk",
    "aita": "Am I the jerk",
    "WIBTA": "Would I be the jerk",
    "wibta": "Would I be the jerk",
    "NTA": "not the jerk",
    "YTA": "you're the jerk",
    "ESH": "everyone's wrong here",
    "NAH": "no one's wrong here",
    "OP": "the original poster",
    "SO": "significant other",
    "SO's": "significant other's",
    "MIL": "mother-in-law",
    "FIL": "father-in-law",
    "SIL": "sister-in-law",
    "BIL": "brother-in-law",
    "BF": "boyfriend",
    "GF": "girlfriend",
    "DH": "dear husband",
    "DW": "dear wife",
    "DD": "dear daughter",
    "DS": "dear son",
    "LDR": "long-distance relationship",
    "IMO": "in my opinion",
    "IMHO": "in my humble opinion",
    "IIRC": "if I recall correctly",
    "AFAIK": "as far as I know",
    "TBH": "to be honest",
    "tbh": "to be honest",
    "IDK": "I don't know",
    "idk": "I don't know",
    "SMH": "shaking my head",
    "IRL": "in real life",
    "STFU": "shut up",
    "LMAO": "laughing out loud",
    "lmao": "laughing out loud",
    "LMFAO": "laughing out loud",
    "LOL": "laughing out loud",
    "lol": "laughing out loud",
    "ROFL": "laughing out loud",
    "BTW": "by the way",
    "btw": "by the way",
    "FYI": "for your information",
    "fyi": "for your information",
    "JK": "just kidding",
    "jk": "just kidding",
    "FWIW": "for what it's worth",
    "fwiw": "for what it's worth",
    "WTF": "what the heck",
    "wtf": "what the heck",
    "GTFO": "get out",
    "AF": "extremely",
    "af": "extremely",
    "POS": "piece of work",
    "pos": "piece of work",
    "ETA": "edited to add",
    "MF": "person",
}

# Unicode normalization table
_UNICODE_MAP: Dict[str, str] = {
    "\u2018": "'",   # left single quote
    "\u2019": "'",   # right single quote
    "\u201c": '"',   # left double quote
    "\u201d": '"',   # right double quote
    "\u2013": "-",   # en-dash
    "\u2014": " - ", # em-dash
    "\u2026": "...", # ellipsis character
    "\u00a0": " ",   # non-breaking space
    "\u200b": "",    # zero-width space
    "\u200c": "",    # zero-width non-joiner
    "\u200d": "",    # zero-width joiner
    "\ufeff": "",    # BOM / zero-width no-break space
    "\u00ad": "",    # soft hyphen
}


class ScriptRewriter:
    """Cleans a raw Reddit post into a TTS-ready narration script.

    The default pipeline uses only regex and string operations — no
    external LLM calls.  The class is structured so an LLM-based
    ``rewrite`` method can be added via subclassing or strategy injection
    in the future.

    Args:
        config: Parsed configuration dictionary (from ``load_config``).
            The ``llm`` section is checked but only the regex fallback
            is implemented in this version.
        logger: Optional pre-built ``BotLogger``.

    Example::

        rewriter = ScriptRewriter(cfg)
        clean = rewriter.rewrite(post_dict)
        tts_ready = rewriter.format_for_tts(clean)
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[BotLogger] = None,
    ) -> None:
        self.cfg = config
        self.log = logger or BotLogger(
            name="ScriptRewriter",
            log_dir=config.get("pipeline", {}).get("log_dir"),
            level=config.get("pipeline", {}).get("log_level", "INFO"),
        )

        groq_cfg = config.get("groq", {})
        self._groq_key: str = groq_cfg.get("api_key", "").strip()
        self._groq_model: str = groq_cfg.get("model", "llama-3.1-70b-versatile").strip()
        self._use_groq: bool = bool(self._groq_key and "YOUR_" not in self._groq_key and self._groq_key != "")

        if self._use_groq:
            self.log.info(
                "LLM provider 'groq' configured with model '%s'",
                self._groq_model,
            )
        else:
            self.log.info("Using regex cleanup pipeline (no LLM)")

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def rewrite(self, post: Dict[str, Any]) -> str:
        """Clean a raw Reddit post dict into a narration script.

        The full pipeline:
        1. Extract and join title + body.
        2. Strip Reddit markdown.
        3. Remove Reddit artefacts (EDIT, TLDR, etc.).
        4. Remove URLs.
        5. Normalise Unicode characters.
        6. Expand abbreviations / sanitise profanity for TTS.
        7. Clean whitespace and punctuation.
        8. Normalise punctuation cadence for spoken word.
        9. Prepend a natural intro line.

        If Gemini LLM is enabled and configured, it rewrites the script using Gemini,
        falling back to the regex clean-up pipeline on any API errors or missing credentials.

        Args:
            post: A post dict with at least ``title`` and ``body`` keys.

        Returns:
            The cleaned narration script as a single string.
        """
        title: str = post.get("title", "").strip()
        body: str = post.get("body", "").strip()

        if not body:
            self.log.warning("Post %s has empty body — returning title only", post.get("id"))
            return title

        self.log.info(
            "Rewriting post %s (%d chars) …",
            post.get("id", "?"),
            len(body),
        )

        # Build raw text: title as opening sentence, then body
        raw = self._merge_title_body(title, body)

        # Try Groq LLM rewrite if active
        if self._use_groq:
            max_retries = 3
            backoff_seconds = [15, 30, 60]
            for attempt in range(max_retries + 1):
                try:
                    self.log.info(
                        "Using Groq LLM rewriter model: %s (attempt %d/%d)",
                        self._groq_model,
                        attempt + 1,
                        max_retries + 1,
                    )
                    
                    headers = {
                        "Authorization": f"Bearer {self._groq_key}",
                        "Content-Type": "application/json",
                    }
                    
                    system_prompt = (
                        "You are a voice actor recording an urgent, deeply personal voice memo. You are NOT an AI assistant, and you are NOT editing a post. You are speaking your raw, unedited personal reality directly into the microphone.\n\n"
                        "Strictly enforce these narrative audio rules:\n"
                        "1. Speak exclusively in the first person (\"I\", \"my\", \"me\").\n"
                        "2. The very first syllable out of your mouth MUST be the dramatic structural hook. Absolutely ZERO introductory padding, greetings, meta-commentary, or setup transitions. Do not say \"So,\" \"Okay,\" \"Hey guys,\" or mention the internet.\n"
                        "3. Jump directly into the high-stakes conflict within the first 5 words. (e.g., \"My mother is running away with my ex-boyfriend.\")\n"
                        "4. Translate all text shorthand into full, natural spoken-word phrases (e.g., convert \"AITA\" to \"Am I the jerk\", \"ex\" to \"ex-partner\", \"MIL\" to \"mother-in-law\").\n"
                        "5. Write in brief, punchy, human sentences optimized for rapid, continuous Text-to-Speech breathing patterns and word-by-word dynamic subtitles. Keep the tension scaling upward.\n"
                        "6. Strip out all editorial structural markers like \"EDIT:\", \"TL;DR:\", or chronological bullet points. Merge into a seamless, gripping, continuous narrative flow.\n\n"
                        "CRITICAL CONTRAINT: Output ONLY the raw spoken script text. No titles, no introduction, no concluding notes, no meta-tags, no conversational filler. If you output a single word of commentary outside the character's direct spoken script, the pipeline will break. Begin speaking NOW:"
                    )
                    
                    payload = {
                        "model": self._groq_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"Reddit post text:\n{raw}"}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2048,
                    }
                    
                    import requests
                    response = requests.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=30,
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        text = data["choices"][0]["message"]["content"]
                        if text and text.strip():
                            text = text.strip()
                            self.log.info(
                                "Groq LLM Rewrite complete — %d chars → %d chars (No intro prepended)",
                                len(raw),
                                len(text),
                            )
                            return text
                        else:
                            self.log.warning("Groq API returned an empty response. Falling back to regex pipeline.")
                            break
                    else:
                        self.log.warning(
                            "Groq API call returned HTTP status %d: %s",
                            response.status_code,
                            response.text[:300],
                        )
                        raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
                except Exception as exc:
                    if attempt < max_retries:
                        sleep_time = backoff_seconds[attempt]
                        # Check if the error message contains a specific retry delay
                        parsed_delay = 0.0
                        exc_str = str(exc)
                        match = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s(?:econds)?", exc_str, re.IGNORECASE)
                        if match:
                            parsed_delay = float(match.group(1))
                        
                        if parsed_delay > 0:
                            sleep_time = max(sleep_time, parsed_delay + 2.0)
                            self.log.warning(
                                "Groq API quota rate-limit detected. Waiting for dynamic delay: %.1f seconds...",
                                sleep_time
                            )
                        else:
                            self.log.warning(
                                "Groq API call failed (attempt %d/%d): %s. Retrying in %d seconds...",
                                attempt + 1,
                                max_retries + 1,
                                exc,
                                sleep_time,
                            )
                        time.sleep(sleep_time)
                    else:
                        self.log.error(
                            "Groq API call failed after %d retries: %s. Falling back to regex pipeline.",
                            max_retries,
                            exc,
                        )

        # Fallback to local regex cleanup pipeline
        self.log.info("Running regex cleanup pipeline")
        text = self._strip_markdown(raw)
        text = self._remove_artifacts(text)
        text = self._remove_urls(text)
        text = self._normalize_unicode(text)
        text = self._expand_abbreviations(text)
        text = self._clean_whitespace(text)
        text = self._normalize_punctuation(text)
        text = self._add_intro(text)

        self.log.info(
            "Rewrite complete (regex) — %d chars → %d chars",
            len(raw),
            len(text),
        )
        return text

    def format_for_tts(self, text: str) -> str:
        """Apply final TTS-specific formatting to a clean script.

        This method is intended to be called after :meth:`rewrite` and
        makes adjustments that improve synthesised speech quality:

        - Ensures the script ends with a sentence terminator.
        - Adds breathing-room spacing around long dashes.
        - Converts numeric ranges to spoken form.
        - Normalises remaining edge-case whitespace.

        Args:
            text: A cleaned script string.

        Returns:
            The TTS-optimised script.
        """
        if not text:
            return text

        result = text

        # Ensure trailing sentence terminator
        result = result.rstrip()
        if result and result[-1] not in ".!?":
            result += "."

        # Add spacing around dashes used for dramatic pauses
        result = re.sub(r"\s*—\s*", " — ", result)
        result = re.sub(r"\s+-\s+", " — ", result)

        # Convert common numeric patterns to TTS-friendly form
        result = re.sub(r"(\d+)\s*-\s*(\d+)", r"\1 to \2", result)

        # Clean up any double-spaces introduced
        result = re.sub(r" {2,}", " ", result)

        # Ensure no leading/trailing whitespace on lines
        lines = [line.strip() for line in result.split("\n")]
        result = "\n".join(line for line in lines if line)

        self.log.debug("TTS formatting applied (%d chars)", len(result))
        return result

    # ------------------------------------------------------------------
    # Pipeline steps (private)
    # ------------------------------------------------------------------

    @staticmethod
    def _merge_title_body(title: str, body: str) -> str:
        """Combine the post title and body into a single text block.

        The title is treated as the opening sentence.  If the title
        doesn't end with punctuation it gets a period appended.

        Args:
            title: Post title.
            body: Post self-text body.

        Returns:
            Combined text with title as first sentence.
        """
        if title and title[-1] not in ".!?":
            title += "."
        return f"{title}\n\n{body}" if title else body

    @staticmethod
    def _strip_markdown(text: str) -> str:
        """Remove Reddit-flavour Markdown formatting.

        Handles: bold, italic, strikethrough, headers, blockquotes,
        links ``[text](url)``, inline code, and fenced code blocks.

        Args:
            text: Raw text potentially containing Markdown.

        Returns:
            Plain text with Markdown syntax removed.
        """
        # Fenced code blocks (``` ... ```)
        text = re.sub(r"```[\s\S]*?```", " ", text)

        # Inline code (`...`)
        text = re.sub(r"`([^`]+)`", r"\1", text)

        # Images ![alt](url)
        text = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", text)

        # Links [text](url)
        text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)

        # Headers (# ... ######)
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

        # Blockquotes
        text = re.sub(r"^>\s?", "", text, flags=re.MULTILINE)

        # Bold + italic (***text*** or ___text___)
        text = re.sub(r"(\*{3}|_{3})(.+?)\1", r"\2", text)

        # Bold (**text** or __text__)
        text = re.sub(r"(\*{2}|_{2})(.+?)\1", r"\2", text)

        # Italic (*text* or _text_) — avoid matching mid-word underscores
        text = re.sub(r"(?<!\w)(\*|_)(?!\s)(.+?)(?<!\s)\1(?!\w)", r"\2", text)

        # Strikethrough (~~text~~)
        text = re.sub(r"~~(.+?)~~", r"\1", text)

        # Horizontal rules (---, ***, ___)
        text = re.sub(r"^[\s]*[-*_]{3,}[\s]*$", "", text, flags=re.MULTILINE)

        # Superscript (^text or ^(text))
        text = re.sub(r"\^\(([^)]+)\)", r"\1", text)
        text = re.sub(r"\^(\S+)", r"\1", text)

        # Reddit spoiler tags >!text!<
        text = re.sub(r">!(.+?)!<", r"\1", text)

        return text

    @staticmethod
    def _remove_artifacts(text: str) -> str:
        """Remove common Reddit artefacts that break narration flow.

        Args:
            text: Text potentially containing Reddit-isms.

        Returns:
            Cleaned text with artefacts removed.
        """
        for pattern in _ARTIFACT_PATTERNS:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _remove_urls(text: str) -> str:
        """Strip all URLs from the text.

        Handles http(s), www, and bare domain patterns.

        Args:
            text: Text potentially containing URLs.

        Returns:
            Text with URLs removed.
        """
        # Full URLs
        text = re.sub(
            r"https?://[^\s)\]]+",
            "",
            text,
        )
        # www. URLs without protocol
        text = re.sub(
            r"www\.[^\s)\]]+",
            "",
            text,
        )
        # Subreddit / user references → spoken form
        text = re.sub(r"r/(\w+)", r"the \1 subreddit", text)
        text = re.sub(r"u/(\w+)", r"a Reddit user", text)

        return text

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Replace Unicode quotes, dashes, and special chars with ASCII.

        Args:
            text: Text potentially containing Unicode characters.

        Returns:
            ASCII-normalised text.
        """
        for uchar, replacement in _UNICODE_MAP.items():
            text = text.replace(uchar, replacement)

        # NFD → strip combining characters for accented letters
        # (keep the base letter)
        normalised = unicodedata.normalize("NFKD", text)
        # Only strip truly invisible combining marks, keep accented chars
        # that are common in English loan-words
        result: List[str] = []
        for ch in normalised:
            cat = unicodedata.category(ch)
            if cat.startswith("M"):  # combining marks
                continue
            result.append(ch)
        return "".join(result)

    @staticmethod
    def _expand_abbreviations(text: str) -> str:
        """Replace Reddit abbreviations and profanity with TTS-safe text.

        Uses whole-word matching to avoid corrupting normal words.

        Args:
            text: Text potentially containing abbreviations.

        Returns:
            Text with abbreviations expanded.
        """
        for abbr, expansion in _TTS_WORD_MAP.items():
            # Whole-word match (word boundary)
            pattern = re.compile(r"\b" + re.escape(abbr) + r"\b")
            text = pattern.sub(expansion, text)
        return text

    @staticmethod
    def _clean_whitespace(text: str) -> str:
        """Normalise whitespace, newlines, and blank lines.

        Args:
            text: Text with potentially messy spacing.

        Returns:
            Text with clean, single-spaced paragraphs.
        """
        # Replace carriage returns
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Collapse 3+ newlines to 2 (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Collapse multiple spaces to one
        text = re.sub(r"[ \t]{2,}", " ", text)

        # Remove spaces before punctuation
        text = re.sub(r"\s+([.,!?;:])", r"\1", text)

        # Remove leading/trailing whitespace per line
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Remove leading/trailing whitespace overall
        return text.strip()

    @staticmethod
    def _normalize_punctuation(text: str) -> str:
        """Normalise punctuation for natural TTS cadence.

        Adjustments:
        - Ellipsis ``...`` → period + pause indicator.
        - Ensure sentences end with proper terminators.
        - Fix comma spacing.
        - Remove repeated punctuation.

        Args:
            text: Text with potentially inconsistent punctuation.

        Returns:
            Punctuation-normalised text.
        """
        # Ellipsis → period (TTS handles pauses better on periods)
        text = re.sub(r"\.{3,}", ".", text)

        # Multiple exclamation/question marks → single
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)

        # Multiple periods → single
        text = re.sub(r"\.{2,}", ".", text)

        # Ensure comma is followed by a space
        text = re.sub(r",(?!\s)", ", ", text)

        # Ensure sentence terminators are followed by a space (if not end)
        text = re.sub(r"([.!?])(?=[A-Za-z])", r"\1 ", text)

        # Remove spaces before sentence terminators
        text = re.sub(r"\s+([.!?])", r"\1", text)

        # Ensure sentences that end in a letter get a period
        # (applied per-line to catch broken sentences)
        lines = text.split("\n")
        fixed_lines: List[str] = []
        for line in lines:
            stripped = line.rstrip()
            if stripped and stripped[-1].isalpha():
                stripped += "."
            fixed_lines.append(stripped)
        text = "\n".join(fixed_lines)

        return text

    @staticmethod
    def _add_intro(text: str) -> str:
        """Prepend a natural spoken-word intro to the script.

        A random variation is chosen to avoid repetition across videos.

        Args:
            text: The cleaned narration body.

        Returns:
            Text with intro prepended.
        """
        intro = random.choice(_INTRO_VARIATIONS)
        return f"{intro}\n\n{text}"
