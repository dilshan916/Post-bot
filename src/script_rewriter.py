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

        import os

        def _clean_key(k: Any) -> str:
            s = str(k or "").strip()
            if not s or s.startswith("YOUR_") or s.endswith("_HERE"):
                return ""
            return s

        groq_cfg = config.get("groq", {})
        env_groq = _clean_key(os.environ.get("GROQ_API_KEY", ""))
        raw_groq_key = _clean_key(groq_cfg.get("api_key", ""))
        raw_groq_keys = [_clean_key(k) for k in groq_cfg.get("api_keys", []) if _clean_key(k)]

        if env_groq:
            self._groq_api_keys = [env_groq]
        elif raw_groq_keys:
            self._groq_api_keys = raw_groq_keys
        elif raw_groq_key:
            self._groq_api_keys = [raw_groq_key]
        else:
            self._groq_api_keys = []

        self._groq_api_key: str = self._groq_api_keys[0] if self._groq_api_keys else ""
        self._groq_key: str = self._groq_api_key
        self._groq_model: str = groq_cfg.get("model", "llama-3.3-70b-versatile").strip()
        self._use_groq: bool = bool(self._groq_api_keys)

        llm_cfg = config.get("llm", {})
        env_gemini = _clean_key(os.environ.get("GEMINI_API_KEY", ""))
        raw_gem_key = _clean_key(llm_cfg.get("api_key", ""))
        raw_gem_keys = [_clean_key(k) for k in llm_cfg.get("api_keys", []) if _clean_key(k)]

        if env_gemini:
            self._gemini_api_keys = [env_gemini]
        elif raw_gem_keys:
            self._gemini_api_keys = raw_gem_keys
        elif raw_gem_key:
            self._gemini_api_keys = [raw_gem_key]
        else:
            self._gemini_api_keys = []

        self._gemini_api_key: str = self._gemini_api_keys[0] if self._gemini_api_keys else ""
        self._gemini_model: str = llm_cfg.get("model", "gemini-2.5-flash").strip()

        if self._use_groq or self._gemini_api_keys:
            self.log.info(
                "LLM providers configured: Groq model='%s', Gemini model='%s'",
                self._groq_model,
                self._gemini_model,
            )
        else:
            self.log.info("Using regex cleanup pipeline (no LLM)")

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def rewrite(self, post: Dict[str, Any], feedback: Optional[str] = None) -> str:
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
        pipeline_mode = self.cfg.get("pipeline", {}).get("pipeline_mode", "monologue")

        used_topics = []
        from pathlib import Path
        riddle_history_path = Path("riddle_history.json")
        if pipeline_mode == "riddle" and riddle_history_path.exists():
            import json
            try:
                with open(riddle_history_path, "r", encoding="utf-8") as f:
                    used_topics = json.load(f)
            except Exception as read_err:
                self.log.warning("Failed to read riddle history: %s", read_err)

        if not body and pipeline_mode not in ("shower", "riddle"):
            self.log.warning("Post %s has empty body — returning title only", post.get("id"))
            return title

        self.log.info(
            "Rewriting post %s …",
            post.get("id", "?"),
        )

        # Build raw text: title as opening sentence, then body
        raw = self._merge_title_body(title, body)

        # Try Gemini rewrite if pipeline_mode == "riddle" and we have Gemini keys
        if pipeline_mode == "riddle" and self._gemini_api_keys:
            max_retries = 3
            backoff_seconds = [15, 30, 60]
            
            system_prompt = self.cfg.get("riddle", {}).get("system_prompt")
            if not system_prompt:
                system_prompt = (
                    "You are a riddle compilation generator. You must strictly output a single, well-formatted JSON object containing a riddle script in Block 1-5 format.\n"
                    "You must strictly output a single, well-formatted JSON object with absolutely nothing else. Do not include any markdown wrap around headers, intro, or outro text outside the JSON.\n\n"
                    "JSON STRUCTURE REQUIRED:\n"
                    "{\n"
                    "  \"answer\": \"The actual answer to the riddle\",\n"
                    "  \"caption\": \"A short, viral description under 10 words followed by EXACTLY 5 relevant hashtags\",\n"
                    "  \"pinned_comment\": \"A teaser question asking viewers to guess the answer\",\n"
                    "  \"script\": [\n"
                    "    {\"speaker\": \"MALE\", \"text\": \"Hook sentence...\"},\n"
                    "    {\"speaker\": \"FEMALE\", \"text\": \"Clue 1 sentence...\"}\n"
                    "  ]\n"
                    "}\n"
                )
            if used_topics:
                system_prompt += f"\n\nDo NOT write a riddle about any of these subjects/answers: {', '.join(used_topics)}"
                
            user_content = f"Riddle concept:\n{raw}"
            if feedback:
                user_content += f"\n\nUser feedback for rewrite: {feedback}"
                
            for attempt in range(max_retries + 1):
                try:
                    current_key = self._gemini_api_keys[attempt % len(self._gemini_api_keys)]
                    self.log.info(
                        "Using Gemini LLM rewriter model: %s (attempt %d/%d) using key ...%s...",
                        self._gemini_model,
                        attempt + 1,
                        max_retries + 1,
                        current_key[-6:] if current_key else ""
                    )
                    
                    from google import genai
                    client = genai.Client(api_key=current_key)
                    
                    response = client.models.generate_content(
                        model=self._gemini_model,
                        contents=f"{system_prompt}\n\n{user_content}"
                    )
                    
                    text = (response.text or "").strip()
                    
                    # Strip markdown code block wrappers if present
                    if text.startswith("```"):
                        lines = text.splitlines()
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].startswith("```"):
                            lines = lines[:-1]
                        text = "\n".join(lines).strip()
                        
                    if not text:
                        self.log.warning("Gemini API returned an empty response. Retrying...")
                        continue
                        
                    # Validate JSON structure
                    import json
                    try:
                        parsed = json.loads(text)
                        answer_val = parsed.get("answer")
                        if isinstance(parsed, dict) and ("script" in parsed or "answer" in parsed):
                            # Try to extract the riddle answer for the exclusions list
                            if not answer_val:
                                ans_match = re.search(r'["\']answer["\']\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
                                if ans_match:
                                    answer_val = ans_match.group(1)
                                    
                            if answer_val:
                                answer_clean = str(answer_val).strip().lower()
                                if answer_clean and answer_clean not in used_topics:
                                    used_topics.append(answer_clean)
                                    used_topics = used_topics[-50:]
                                    try:
                                        with open(riddle_history_path, "w", encoding="utf-8") as f:
                                            json.dump(used_topics, f, indent=2)
                                    except Exception as write_err:
                                        self.log.warning("Failed to save riddle history: %s", write_err)
                                        
                            self.log.info(
                                "Gemini LLM Rewrite complete — %d chars → %d chars (No intro prepended)",
                                len(raw),
                                len(text),
                            )
                            return text
                    except Exception as json_err:
                        self.log.warning("Gemini JSON parsing failed for riddle mode: %s. Output: %s", json_err, text)
                        
                except Exception as exc:
                    self.log.warning(
                        "Gemini API call failed (attempt %d/%d): %s. Retrying in %d seconds...",
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        backoff_seconds[attempt] if attempt < len(backoff_seconds) else 10
                    )
                    if attempt < max_retries:
                        time.sleep(backoff_seconds[attempt] if attempt < len(backoff_seconds) else 10)
                        
            self.log.warning("Gemini Riddle Mode rewriting failed. Falling back to Groq/Regex.")

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
                    
                    current_key = self._groq_api_keys[attempt % len(self._groq_api_keys)] if self._groq_api_keys else self._groq_key
                    headers = {
                        "Authorization": f"Bearer {current_key}",
                        "Content-Type": "application/json",
                    }
                    
                    pipeline_mode = self.cfg.get("pipeline", {}).get("pipeline_mode", "monologue")
                    if pipeline_mode == "conversational":
                        system_prompt = self.cfg.get("conversational", {}).get("system_prompt")
                        if not system_prompt:
                            system_prompt = (
                                "You are a dramatic script writer. Convert the following story into a strict structured dialogue format.\n"
                                "You must strictly output a single, well-formatted JSON object with absolutely nothing else. Do not include any markdown wrap around headers, intro, or outro text outside the JSON.\n\n"
                                "JSON STRUCTURE REQUIRED:\n"
                                "{\n"
                                "  \"caption\": \"A short, viral description under 10 words followed by EXACTLY 5 relevant hashtags\",\n"
                                "  \"pinned_comment\": \"A value-driven, organic question based directly on the story's conflict to engage viewers in debate\",\n"
                                "  \"script\": [\n"
                                "    {\"speaker\": \"MALE\", \"text\": \"Sentence spoken by male character...\"},\n"
                                "    {\"speaker\": \"FEMALE\", \"text\": \"Sentence spoken by female character...\"}\n"
                                "  ]\n"
                                "}\n\n"
                                "Rules:\n"
                                "1. Split the narrative into a sequence of conversational turns between a MALE and a FEMALE speaker.\n"
                                "2. Make it dramatic, engaging, and flow naturally like a text message drama or conversation.\n"
                                "3. The first block must start immediately with the hook. No intro or explanation.\n"
                                "4. Translate all text shorthand into full, natural spoken-word phrases (e.g., convert \"AITA\" to \"Am I the jerk\", \"ex\" to \"ex-partner\", \"MIL\" to \"mother-in-law\").\n"
                                "5. CAPTION AND HASHTAG HYGIENE: The caption field must be extremely brief (under 10 words) and end with exactly 5 hashtags. Use 2 genre tags (e.g., #relationship, #storytime) and 3 specific contextual tags defining the story's subject.\n"
                                "6. UTIS COMPLIANCE COMMENT: The pinned_comment field must contain a unique, thought-provoking question derived organically from the story's conflict to stimulate genuine interest-driven debate.\n"
                                "7. Do NOT include markdown formatting like ```json or ```. Output ONLY the raw JSON string."
                            )
                        user_content = f"Reddit post text:\n{raw}"
                    elif pipeline_mode == "thread":
                        system_prompt = self.cfg.get("thread", {}).get("system_prompt")
                        if not system_prompt:
                            system_prompt = (
                                "You are a professional script compiler. Convert the following Reddit post question and its comment answers into a strict structured compilation format.\n"
                                "You must output the script in a strict, parsable JSON Array format, mapping roles cleanly:\n"
                                "[\n"
                                "  {\"speaker\": \"NARRATOR\", \"voice\": \"en-US-ChristopherNeural\", \"author\": \"reddit_question\", \"score\": 9500, \"text\": \"What is the thread question text here?\"},\n"
                                "  {\"speaker\": \"USER_1\", \"voice\": \"en-US-GuyNeural\", \"author\": \"username1\", \"score\": 1200, \"text\": \"Answer 1...\"}\n"
                                "]\n\n"
                                "Rules:\n"
                                "1. The first block must represent the thread question (speaker: \"NARRATOR\").\n"
                                "2. Each subsequent block represents a clean, rewritten user comment.\n"
                                "3. Choose a voice from monologue/conversational voices dynamically for each comment (assigning it to the \"voice\" field).\n"
                                "4. Clean all comments of Reddit-specific meta elements (like \"EDIT\", \"TL;DR\", \"Update\").\n"
                                "5. Translate all text shorthand into full spoken-word phrases (e.g. \"AITA\" -> \"Am I the jerk\", \"ex\" -> \"ex-partner\").\n"
                                "6. Ensure the text ends with appropriate sentence punctuation (?, !, or .) so the TTS engine pauses naturally.\n"
                                "7. Do NOT include markdown formatting like ```json or ```. Output ONLY the raw JSON array string."
                            )
                        comments_list = post.get("comments", [])
                        comments_str = ""
                        for c_idx, c in enumerate(comments_list):
                            comments_str += f"Comment {c_idx+1} by u/{c['author']}:\n{c['body']}\n\n"
                        user_content = (
                            f"Reddit Thread Question: {title}\n"
                            f"Description/Body: {body}\n\n"
                            f"Top Comments:\n{comments_str}"
                        )
                    elif pipeline_mode in ("shower", "riddle"):
                        system_prompt = self.cfg.get("riddle", {}).get("system_prompt")
                        if not system_prompt:
                            system_prompt = (
                                "You are an elite, viral short-form scriptwriter specialized in creating high-retention Facebook Reels for a premium US/UK audience in \"Fun Riddle Mode\". Your primary objective is to output a flawless JSON object that satisfies 2026 Meta Algorithmic Compliance (UTIS and Anti-Engagement Bait models).\n\n"
                                "You must strictly output a single, well-formatted JSON object with absolutely nothing else. Do not include any markdown wrap around headers, intro, or outro text outside the JSON.\n\n"
                                "JSON STRUCTURE REQUIRED:\n"
                                "{\n"
                                "  \"answer\": \"The single-word or short-phrase answer to the riddle\",\n"
                                "  \"caption\": \"A punchy, brief description under 10 words followed by EXACTLY 5 relevant hashtags\",\n"
                                "  \"pinned_comment\": \"A value-driven, highly specific question based directly on the riddle content to engage viewers organically\",\n"
                                "  \"script\": [\n"
                                "    {\"speaker\": \"MALE\", \"emotion\": \"excited\", \"text\": \"The opening hook line stating the riddle's stakes or twist setup\"},\n"
                                "    {\"speaker\": \"FEMALE\", \"emotion\": \"curious\", \"text\": \"Reaction or prompt for clues\"},\n"
                                "    {\"speaker\": \"MALE\", \"emotion\": \"explaining\", \"text\": \"The riddle clues delivered clearly\"},\n"
                                "    {\"speaker\": \"FEMALE\", \"emotion\": \"thinking\", \"text\": \"A relatable wrong guess or expression of confusion\"},\n"
                                "    {\"speaker\": \"MALE\", \"emotion\": \"talking\", \"text\": \"The closing call to action challenge pushing for shares\"}\n"
                                "  ]\n"
                                "}\n\n"
                                "STRICT CONTENT RULES (META 2026 UTIS COMPLIANCE):\n"
                                "1. CRITICAL BAN ON ENGAGEMENT-BAIT: You are absolutely FORBIDDEN from using generic hype phrases like \"Only 1% can solve this\", \"Only geniuses know this\", \"I bet $100 you will fail\", or \"Comment below to get pinned\". The algorithm downranks this phrasing immediately.\n"
                                "2. THE HOOK: The opening dialogue must state the riddle's actual high stakes, an intriguing logical claim, or a surprising visual setup directly tied to the specific twist of the riddle. Keep it under 3 seconds of pacing.\n"
                                "3. MIX OF CASUAL & MATH RIDDLES: You must alternate between famous classic word riddles (e.g., candle, shadow, map) and clever mathematical/numerical logic riddles (e.g., divisibility constraints like \"greater than 5, less than 15, only divisible by 3\", sequence patterns, or numeric wordplay like \"What is between 4 and 6 but is not 5? Answer: the word 'and'\"). Keep them simple, engaging, and mind-bending.\n"
                                "4. CAPTION AND HASHTAG HYGIENE: The caption field must be extremely brief and end with exactly 5 hashtags. Avoid spammy tags like #viral, #trending, #fbreels, #explore. Instead, use 2 riddle-centric tags (e.g., #riddles, #brainteasers) and 3 specific contextual tags defining the riddle's subject (e.g., if the riddle is about a river, use #river, #nature, #puzzle).\n"
                                "5. UTIS COMPLIANCE COMMENT: The pinned_comment field must contain a unique, thought-provoking question or teaser derived organically from the riddle's logic. It must stimulate a genuine interest-driven response from the viewer (e.g., \"If it has hands but can't clap, what else can it do? Lock your honest guess below!\"). No templated or lazy CTAs allowed.\n"
                                "6. SCRIPT PACING: Keep the overall script short and fast-paced (between 15 to 25 seconds estimated duration). Ensure word transitions are seamless for high-retention text-to-speech generation."
                            )
                        exclude_clause = ""
                        if used_topics:
                            exclude_clause = f"Do NOT write a riddle about any of these subjects or words: {', '.join(used_topics)}.\n"
                        salt = f"{random.randint(1000, 9999)}"
                        user_content = (
                            "Generate either a famous, widely-known word riddle (e.g. riddles about an egg, a candle, a shadow, footsteps, a map, a coffin, etc.) OR a clever mathematical/numerical/logical riddle (e.g. greater than 5 and less than 15 but divisible by only 3, pattern sequences, or clever math wordplay/logic puzzles like 'What is between 4 and 6 but is not 5?').\n"
                            "Do NOT invent weird, obscure, or awkward riddles. Keep them universally recognized, clever, and highly engaging.\n"
                            f"{exclude_clause}"
                            "Strictly adhere to the alternating 5-block structure: Hook sentence (Block 1), response sentence (Block 2), riddle clues sentence (Block 3), confused guess reaction sentence (Block 4), and Double CTA ending sentence (Block 5).\n"
                            "The opening dialogue block must state the riddle's actual high stakes, an intriguing logical claim, or a surprising visual setup directly tied to the specific twist of the riddle.\n"
                            "The riddle clues block (Block 3) MUST state the COMPLETE famous riddle clues in a single compound sentence (e.g. 'What has a head but never weeps, a bed but never sleeps, and a bank but no money?'). Do not truncate it to a single brief clue.\n"
                            "Every single block MUST contain strictly ONE short sentence. Do NOT name the object/answer itself anywhere in the script. Describe it using mysterious, clever clues.\n"
                            "You MUST also generate a very short, catchy caption with exactly 5 hashtags under the 'caption' field, and an engaging comment teaser question under the 'pinned_comment' field.\n"
                            f"Focus on a random creative object, concept, word, or mathematical riddle. (Seed/Salt: {salt})"
                        )
                    else:
                        is_mode_5 = self.cfg.get("pipeline", {}).get("mode_5_active", False)
                        system_prompt = self.cfg.get("monologue", {}).get("system_prompt")
                        if not system_prompt:
                            system_prompt = (
                                "You are a voice actor recording an urgent, deeply personal voice memo. You are NOT an AI assistant, and you are NOT editing a post. You are speaking your raw, unedited personal reality directly into the microphone.\n\n"
                                "Strictly enforce these narrative audio rules:\n"
                                "1. Speak exclusively in the first person (\"I\", \"my\", \"me\").\n"
                                "2. CRITICAL CLICKBAIT HOOK: The very first sentence MUST be a hyper-dramatic, shocking, or deeply emotional confession. Start at the absolute climax, betrayal, or conflict of the story. Do NOT build up slowly.\n"
                                "   * Bad (too slow): \"My brother-in-law called my wife a bad parent at dinner.\"\n"
                                "   * Good (dramatic hook): \"I stood there silently while my own family tore my wife to shreds.\"\n"
                                "   * Bad (too slow): \"My best friend said a derogatory slur at our wedding.\"\n"
                                "   * Good (dramatic hook): \"My wedding ended in pure chaos when my childhood best friend insulted my wife.\"\n"
                                "3. Jump directly into the emotional core within the first 5 words. Absolutely ZERO introductory padding, greetings, meta-commentary, or setup transitions. Do not say \"So,\" \"Okay,\" \"Hey guys,\" \"Reddit,\" or \"Today.\"\n"
                                "4. Translate all text shorthand into full, natural spoken-word phrases (e.g., convert \"AITA\" to \"Am I the jerk\", \"ex\" to \"ex-partner\", \"MIL\" to \"mother-in-law\").\n"
                                "5. Write in brief, punchy, human sentences optimized for rapid, continuous Text-to-Speech breathing patterns and word-by-word dynamic subtitles. Keep the tension scaling upward.\n"
                                "6. Strip out all editorial structural markers like \"EDIT:\", \"TL;DR:\", or chronological bullet points. Merge into a seamless, gripping, continuous narrative flow.\n\n"
                                "CRITICAL CONSTRAINT: Output ONLY the raw spoken script text. No titles, no introduction, no concluding notes, no meta-tags, no conversational filler. If you output a single word of commentary outside the character's direct spoken script, the pipeline will break. Begin speaking NOW:"
                            )
                        if is_mode_5:
                            system_prompt = (
                                "You are a voice actor and social media writer. Convert the following story into a structured monologue format.\n"
                                "You must strictly output a single, well-formatted JSON object with absolutely nothing else. Do not include any markdown wrap around headers, intro, or outro text outside the JSON.\n\n"
                                "JSON STRUCTURE REQUIRED:\n"
                                "{\n"
                                "  \"caption\": \"A short, viral description under 10 words followed by EXACTLY 5 relevant hashtags\",\n"
                                "  \"pinned_comment\": \"A value-driven, organic question based directly on the story's conflict to engage viewers in debate\",\n"
                                "  \"script\": \"The complete raw spoken monologue text (written strictly in first person 'I', 'my', 'me', and containing the entire story in one continuous block of text)\"\n"
                                "}\n\n"
                                "You MUST follow these strict narrative audio rules for the 'script' field:\n"
                                f"{system_prompt}\n\n"
                                "CRITICAL CONSTRAINT: Do NOT include markdown formatting like ```json or ```. Output ONLY the raw JSON string containing 'caption', 'pinned_comment', and 'script' keys."
                            )
                        user_content = f"Reddit post text:\n{raw}"
                    if feedback:
                        user_content += f"\n\nAdditional feedback instructions for rewriting: {feedback}"

                    payload = {
                        "model": self._groq_model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 2048,
                    }
                    
                    import requests
                    from requests.exceptions import Timeout
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
                            is_mode_5 = self.cfg.get("pipeline", {}).get("mode_5_active", False)
                            if pipeline_mode in ("conversational", "thread", "shower", "riddle") or (pipeline_mode == "monologue" and is_mode_5):
                                # Clean markdown code wrappers if present
                                text_clean = text
                                if text_clean.startswith("```"):
                                    text_clean = re.sub(r"^```(?:json)?\n?", "", text_clean, flags=re.IGNORECASE)
                                    text_clean = re.sub(r"\n?```$", "", text_clean)
                                text_clean = text_clean.strip()
                                
                                # Robust JSON extraction:
                                if pipeline_mode in ("riddle", "conversational") or (pipeline_mode == "monologue" and is_mode_5):
                                    if text_clean.startswith("{"):
                                        match_json = re.search(r"\{.*\}", text_clean, re.DOTALL)
                                    elif text_clean.startswith("["):
                                        match_json = re.search(r"\[.*\]", text_clean, re.DOTALL)
                                    else:
                                        match_json = re.search(r"\{.*\}|\[.*\]", text_clean, re.DOTALL)
                                else:
                                    match_json = re.search(r"\[.*\]", text_clean, re.DOTALL)
                                if match_json:
                                    text_clean = match_json.group(0)
                                    
                                # Clean up raw literal newlines inside double quotes so json.loads doesn't fail
                                def replace_newlines(match_obj):
                                    return match_obj.group(0).replace('\n', ' ').replace('\r', '')
                                text_clean = re.sub(r'"(?:[^"\\]|\\.)*"', replace_newlines, text_clean)
                                
                                # Strip trailing commas in JSON arrays/objects
                                text_clean = re.sub(r',\s*([\]}])', r'\1', text_clean)
                                
                                import json
                                try:
                                    parsed = json.loads(text_clean)
                                    answer_val = None
                                    
                                    if isinstance(parsed, dict) and pipeline_mode in ("riddle", "conversational") and "script" in parsed:
                                        script_blocks = parsed.get("script", [])
                                        if not isinstance(script_blocks, list):
                                            raise ValueError("JSON script output is not an array/list")
                                        if pipeline_mode == "riddle":
                                            answer_val = parsed.get("answer")
                                        # Normalize whole dictionary JSON formatting
                                        text = json.dumps(parsed)
                                    elif isinstance(parsed, list):
                                        script_blocks = parsed
                                        if not isinstance(script_blocks, list):
                                            raise ValueError("JSON script output is not an array/list")
                                        # Normalize list JSON formatting
                                        text = json.dumps(script_blocks)
                                    else:
                                        raise ValueError("JSON output format is invalid")
                                    
                                    # Try to find "answer": "..." in raw response via regex if not found in parsed dict
                                    if not answer_val and pipeline_mode == "riddle":
                                        ans_match = re.search(r'["\']answer["\']\s*:\s*["\']([^"\']+)["\']', text_clean, re.IGNORECASE)
                                        if ans_match:
                                            answer_val = ans_match.group(1)
                                            
                                    # Save to riddle history if we got an answer
                                    if answer_val and pipeline_mode == "riddle":
                                        answer_clean = str(answer_val).strip().lower()
                                        if answer_clean and answer_clean not in used_topics:
                                            used_topics.append(answer_clean)
                                            used_topics = used_topics[-50:]
                                            try:
                                                with open(riddle_history_path, "w", encoding="utf-8") as f:
                                                    json.dump(used_topics, f, indent=2)
                                            except Exception as write_err:
                                                self.log.warning("Failed to save riddle history: %s", write_err)
                                except Exception as exc:
                                    self.log.warning("Groq JSON parsing failed for conversational mode: %s. Output: %s", exc, text)
                                    # Attempt robust fallback recovery of dialogue blocks to avoid reading raw JSON syntax aloud
                                    recovered_blocks = []
                                    # Find all complete or incomplete {...} structures
                                    obj_matches = re.findall(r'\{([^{}]+)(?:\}|$)', text_clean, re.DOTALL)
                                    keys = ["speaker", "emotion", "text", "voice", "author", "score"]
                                    
                                    for obj_str in obj_matches:
                                        key_positions = []
                                        for key in keys:
                                            match = re.search(r'["\']' + key + r'["\']\s*:', obj_str, re.IGNORECASE)
                                            if match:
                                                key_positions.append((match.start(), match.end(), key))
                                        
                                        if not key_positions:
                                            continue
                                            
                                        key_positions.sort()
                                        item = {}
                                        
                                        for i, (start, end, key) in enumerate(key_positions):
                                            val_start = end
                                            if i + 1 < len(key_positions):
                                                val_end = key_positions[i+1][0]
                                            else:
                                                val_end = len(obj_str)
                                                
                                            val_str = obj_str[val_start:val_end].strip()
                                            
                                            # Clean trailing commas/whitespace/quotes
                                            if val_str.endswith(','):
                                                val_str = val_str[:-1].strip()
                                            if val_str.startswith('"') and val_str.endswith('"'):
                                                val_str = val_str[1:-1]
                                            elif val_str.startswith("'") and val_str.endswith("'"):
                                                val_str = val_str[1:-1]
                                            elif val_str.startswith('"'):
                                                val_str = val_str[1:]
                                                if val_str.endswith('"'):
                                                    val_str = val_str[:-1]
                                            elif val_str.startswith("'"):
                                                val_str = val_str[1:]
                                                if val_str.endswith("'"):
                                                    val_str = val_str[:-1]
                                            val_str = val_str.strip()
                                            
                                            item[key] = val_str.replace('\\"', '"').replace("\\'", "'")
                                            
                                        if "speaker" in item and "text" in item:
                                            if pipeline_mode == "shower":
                                                recovered_blocks.append({
                                                    "speaker": item["speaker"].strip().upper(),
                                                    "voice": item.get("voice", "en-US-ChristopherNeural").strip(),
                                                    "author": item.get("author", "reddit_user").strip(),
                                                    "score": int(item["score"]) if "score" in item and str(item["score"]).isdigit() else random.randint(200, 4500),
                                                    "option_a": item.get("option_a", "Option A").strip(),
                                                    "option_b": item.get("option_b", "Option B").strip(),
                                                    "percentage_a": int(item["percentage_a"]) if "percentage_a" in item and str(item["percentage_a"]).isdigit() else random.randint(40, 60),
                                                    "text": item["text"].strip()
                                                })
                                            elif pipeline_mode == "thread":
                                                recovered_blocks.append({
                                                    "speaker": item["speaker"].strip().upper(),
                                                    "voice": item.get("voice", "en-US-ChristopherNeural").strip(),
                                                    "author": item.get("author", "reddit_user").strip(),
                                                    "score": int(item["score"]) if "score" in item and str(item["score"]).isdigit() else random.randint(200, 4500),
                                                    "text": item["text"].strip()
                                                })
                                            else:
                                                recovered_blocks.append({
                                                    "speaker": item["speaker"].strip().upper(),
                                                    "emotion": item.get("emotion", "talking").strip(),
                                                    "text": item["text"].strip()
                                                })
                                    
                                    if recovered_blocks:
                                        self.log.info("Successfully recovered %d dialogue turns using robust regex parsing.", len(recovered_blocks))
                                        text = json.dumps(recovered_blocks)
                                    else:
                                        # Fallback: strip any residual JSON-like syntax to avoid spoken syntax
                                        cleaned_text = re.sub(r'[{}\[\]"\'\\]', '', text_clean)
                                        cleaned_text = re.sub(r'\s*(?:speaker|text)\s*:', '', cleaned_text, flags=re.IGNORECASE)
                                        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
                                        fallback = [{"speaker": "MALE", "text": cleaned_text}]
                                        text = json.dumps(fallback)
                            
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
        # If Groq failed or is not available, try Gemini fallback for conversational mode
        pipeline_mode = self.cfg.get("pipeline", {}).get("pipeline_mode", "monologue")
        if pipeline_mode == "conversational" and self._gemini_api_keys:
            self.log.info("Attempting Conversational script rewriting with Gemini fallback...")
            system_prompt = self.cfg.get("conversational", {}).get("system_prompt")
            user_content = f"Reddit post text:\n{raw}"
            if feedback:
                user_content += f"\n\nAdditional feedback instructions for rewriting: {feedback}"
            for attempt, key in enumerate(self._gemini_api_keys):
                try:
                    from google import genai
                    client = genai.Client(api_key=key)
                    response = client.models.generate_content(
                        model=self._gemini_model,
                        contents=f"{system_prompt}\n\n{user_content}"
                    )
                    g_text = (response.text or "").strip()
                    if g_text.startswith("```"):
                        g_text = re.sub(r"^```(?:json)?\n?", "", g_text, flags=re.IGNORECASE)
                        g_text = re.sub(r"\n?```$", "", g_text)
                    g_text = g_text.strip()
                    match_json = re.search(r"\{.*\}", g_text, re.DOTALL)
                    if match_json:
                        g_text = match_json.group(0)
                    import json
                    parsed = json.loads(g_text)
                    if isinstance(parsed, dict) and "script" in parsed:
                        self.log.info("Gemini Conversational Rewrite complete — %d chars → %d chars", len(raw), len(g_text))
                        return json.dumps(parsed)
                except Exception as g_exc:
                    self.log.warning("Gemini Conversational fallback attempt %d failed: %s", attempt + 1, g_exc)

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
        
        pipeline_mode = self.cfg.get("pipeline", {}).get("pipeline_mode", "monologue")
        if pipeline_mode == "conversational":
            import json
            fallback = [{"speaker": "MALE", "text": text}]
            return json.dumps(fallback)
        elif pipeline_mode == "thread":
            import json
            fallback = [{"speaker": "NARRATOR", "voice": self.cfg.get("tts", {}).get("voice", "en-US-ChristopherNeural"), "author": post.get("author", "reddit_question"), "score": post.get("score", 9500), "text": text}]
            return json.dumps(fallback)
        elif pipeline_mode == "riddle":
            import json
            fallback = [
                {"speaker": "MALE", "emotion": "surprised", "text": "I bet you one hundred dollars you can't solve this in five seconds! I have a face but no eyes, and hands but no arms. What am I?"},
                {"speaker": "FEMALE", "emotion": "thinking", "text": "Wait, a face and hands? Is it a clock? No, that's way too simple!"},
                {"speaker": "MALE", "emotion": "talking", "text": "Do you know the answer? Comment below to get PINNED and follow for more mind-melting riddles!"}
            ]
            return json.dumps(fallback)
        elif pipeline_mode == "shower":
            import json
            max_thoughts = self.cfg.get("shower", {}).get("max_thoughts", 3)
            candidates = post.get("candidates", [])
            fallback = []
            # Prepend high-impact warning intro block
            fallback.append({
                "speaker": "INTRO",
                "voice": self.cfg.get("shower", {}).get("default_voice", "en-US-ChristopherNeural"),
                "author": "system",
                "score": 99999,
                "text": "Would you rather? Only geniuses can answer the last one!"
            })
            for i, c in enumerate(candidates[:max_thoughts]):
                c_title = c.get("title", "").strip()
                # Try to parse "Would you rather A or B?"
                match = re.search(r"would you rather\s+(.*?)\s+or\s+(.*)", c_title, re.IGNORECASE)
                if match:
                    opt_a = match.group(1).strip()
                    opt_b = match.group(2).strip()
                    # Clean trailing question marks
                    if opt_b.endswith("?"):
                        opt_b = opt_b[:-1].strip()
                else:
                    # Fallback split by ' or '
                    parts_split = re.split(r"\s+or\s+", c_title, flags=re.IGNORECASE)
                    if len(parts_split) >= 2:
                        opt_a = parts_split[0].strip()
                        opt_b = " or ".join(parts_split[1:]).strip()
                        if opt_b.endswith("?"):
                            opt_b = opt_b[:-1].strip()
                    else:
                        opt_a = c_title
                        opt_b = "Something else"
                
                # Truncate options to keep them clean
                opt_a = opt_a[:50]
                opt_b = opt_b[:50]
                
                # Synthesise text narration
                spoken_text = f"Would you rather {opt_a}, or, {opt_b}?"
                
                fallback.append({
                    "speaker": f"Q{i+1}",
                    "voice": self.cfg.get("shower", {}).get("default_voice", "en-US-ChristopherNeural"),
                    "author": c.get("author", "anonymous"),
                    "score": c.get("score", 1000 - i * 100),
                    "option_a": opt_a,
                    "option_b": opt_b,
                    "percentage_a": random.randint(38, 62),
                    "text": spoken_text
                })
            if len(fallback) == 1:
                # If no candidates, fallback to a single custom message
                fallback = [
                    {"speaker": "INTRO", "voice": self.cfg.get("shower", {}).get("default_voice", "en-US-ChristopherNeural"), "author": "system", "score": 99999, "text": "Would you rather? Only geniuses can answer the last one!"},
                    {"speaker": "Q1", "voice": self.cfg.get("shower", {}).get("default_voice", "en-US-ChristopherNeural"), "author": "system", "score": 2500, "option_a": "Have a personal chef", "option_b": "Have a private jet", "text": "Would you rather have a personal chef, or, have a private jet?", "percentage_a": 54}
                ]
            return json.dumps(fallback)
            
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
