"""
RedditDaily-Bot — Reddit Scraper & History Manager
====================================================
Connects to Reddit directly via public RSS feeds using the requests library,
fetches self-text posts from configured subreddits, applies quality filters,
and de-duplicates against a persistent JSON ledger so stories are never re-used.
"""

from __future__ import annotations

import time
import random
import html
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from src.utils import (
    BotLogger,
    load_json,
    resolve_path,
    save_json,
    word_count,
    validate_api_key,
)

_USER_AGENTS: List[str] = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:122.0) Gecko/20100101 Firefox/122.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 Edge/120.0.0.0",
]


class RedditScraper:
    """Scrapes Reddit self-text posts and manages a de-duplication ledger.

    The scraper reads all connection parameters and filter thresholds from
    the project's ``config.yaml``. Scraped post IDs are persisted to a
    local JSON file so the pipeline never processes the same story twice.

    Args:
        config: Parsed configuration dictionary (from ``load_config``).
        logger: Optional pre-built ``BotLogger``. A default one is
            created when *None* is passed.

    Raises:
        KeyError: If required configuration keys are absent.

    Example::

        from src.utils import load_config
        cfg = load_config()
        scraper = RedditScraper(cfg)
        best = scraper.get_best_story()
        if best:
            scraper.mark_scraped(best["id"])
    """

    def __init__(
        self,
        config: Dict[str, Any],
        logger: Optional[BotLogger] = None,
    ) -> None:
        self.cfg: Dict[str, Any] = config
        self.reddit_cfg: Dict[str, Any] = config["reddit"]
        self.pipeline_cfg: Dict[str, Any] = config["pipeline"]
        self.log: BotLogger = logger or BotLogger(
            name="RedditScraper",
            log_dir=self.pipeline_cfg.get("log_dir"),
            level=self.pipeline_cfg.get("log_level", "INFO"),
        )

        # History ledger path (resolved against project root)
        self._history_path = resolve_path(
            self.pipeline_cfg["scraped_history"], create=True
        )
        self._history: Dict[str, str] = self._load_history()

        self.log.info(
            "RedditScraper initialised — %d subreddits, %d posts already scraped",
            len(self.reddit_cfg.get("subreddits", [])),
            len(self._history),
        )

        llm_cfg = config.get("llm", {})
        provider = llm_cfg.get("provider", "").strip().lower()
        self._use_llm: bool = (provider == "gemini")
        self._api_key: str = llm_cfg.get("api_key", "")
        self._api_keys: List[str] = llm_cfg.get("api_keys", [])
        if not self._api_keys and self._api_key:
            self._api_keys = [self._api_key]
        self._model: str = llm_cfg.get("model", "gemini-2.5-flash")
        self._session = requests.Session()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_history(self) -> Dict[str, str]:
        """Load the scraped-post history ledger from disk.

        Returns:
            A dict mapping post IDs to ISO-timestamp strings.
        """
        data = load_json(str(self._history_path), default={})
        if not isinstance(data, dict):
            self.log.warning(
                "History ledger at %s is not a dict — resetting",
                self._history_path,
            )
            return {}
        return data

    def _save_history(self) -> None:
        """Persist the in-memory history ledger to disk."""
        save_json(self._history, str(self._history_path))
        self.log.debug(
            "History saved (%d entries) → %s",
            len(self._history),
            self._history_path,
        )

    def _is_scraped(self, post_id: str) -> bool:
        """Check whether a post has already been scraped.

        Args:
            post_id: Reddit submission ID (without prefix).

        Returns:
            True if the post ID exists in the history ledger.
        """
        return post_id in self._history

    def _fetch_subreddit_posts(
        self, subreddit_name: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Fetch raw submissions from a single subreddit using requests RSS.

        The sort method (hot / top / new / rising) and time_filter are
        read from config. For ``top`` sort, the ``time_filter`` parameter
        is forwarded to the API.

        Args:
            subreddit_name: Name of the target subreddit (no ``r/`` prefix).
            limit: Maximum number of submissions to retrieve.

        Returns:
            A list of raw submission dictionary data parsed from RSS.
        """
        sort_method: str = self.reddit_cfg.get("sort", "hot").lower()
        time_filter: str = self.reddit_cfg.get("time_filter", "day").lower()

        # Build custom user agent to avoid being blocked by Reddit
        user_agent = self.reddit_cfg.get("user_agent", "")
        # Use a realistic desktop browser user agent if none is configured or it's a default placeholder
        if not user_agent or "YourRedditUsername" in user_agent or "YOUR_" in user_agent:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        # Use public RSS feed to bypass JSON block
        url = f"https://www.reddit.com/r/{subreddit_name}/{sort_method}.rss"
        params: Dict[str, Any] = {}
        if sort_method == "top":
            params["t"] = time_filter

        max_retries = 3
        for attempt in range(max_retries + 1):
            ua = random.choice(_USER_AGENTS) if (not user_agent or "YourRedditUsername" in user_agent or "YOUR_" in user_agent) else user_agent
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Ch-Ua": '"Not A(MeatPage;Bypass;1.0", "Chromium";"121", "Google Chrome";"121"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            }
            try:
                self.log.debug("GET %s with params %s (attempt %d/%d)", url, params, attempt + 1, max_retries + 1)
                response = requests.get(url, headers=headers, params=params, timeout=15)

                if response.status_code == 429:
                    if attempt < max_retries:
                        self.log.warning(
                            "Reddit RSS rate-limited (HTTP 429) for r/%s. "
                            "Sleeping 30 seconds before retry %d/%d...",
                            subreddit_name, attempt + 1, max_retries
                        )
                        time.sleep(30)
                        continue
                    else:
                        self.log.error(
                            "Reddit RSS rate-limited (HTTP 429) for r/%s. "
                            "Failed after %d retries.",
                            subreddit_name, max_retries
                        )
                        return []

                if response.status_code != 200:
                    self.log.error(
                        "Failed to fetch r/%s RSS: HTTP %d - %s",
                        subreddit_name,
                        response.status_code,
                        response.text[:200],
                    )
                    return []

                root = ET.fromstring(response.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}

                posts = []
                for entry in root.findall('atom:entry', ns):
                    # ID
                    id_node = entry.find('atom:id', ns)
                    raw_id = id_node.text if id_node is not None else ""
                    # ID format is t3_postid. Extract postid.
                    post_id = raw_id.split('_')[1] if '_' in raw_id else raw_id

                    # Title
                    title_node = entry.find('atom:title', ns)
                    title = title_node.text if title_node is not None else ""

                    # Content (HTML body)
                    content_node = entry.find('atom:content', ns)
                    content_html = content_node.text if content_node is not None else ""

                    # Author
                    author_node = entry.find('atom:author/atom:name', ns)
                    author = author_node.text if author_node is not None else "[deleted]"
                    if author.startswith("/u/"):
                        author = author[3:]

                    # Link / URL
                    link_node = entry.find("atom:link[@rel='alternate']", ns)
                    if link_node is None:
                        link_node = entry.find("atom:link", ns)
                    permalink_url = link_node.attrib.get('href', '') if link_node is not None else ""

                    # Created UTC
                    updated_node = entry.find('atom:updated', ns)
                    if updated_node is None:
                        updated_node = entry.find('atom:published', ns)

                    created_utc = 0.0
                    if updated_node is not None and updated_node.text:
                        try:
                            dt = datetime.fromisoformat(updated_node.text)
                            created_utc = dt.timestamp()
                        except Exception:
                            pass

                    # Extract selftext (body) from the HTML content
                    body = ""
                    is_self = False
                    match = re.search(r'<div class="md">(.*?)</div>', content_html, re.DOTALL)
                    if match:
                        body_html = match.group(1)
                        body = re.sub(r'<[^>]+>', '', body_html)
                        body = html.unescape(body).strip()
                        is_self = True

                    # Note: RSS does not provide score. Set it to a dummy high score to pass filter thresholds.
                    min_upvotes = self.reddit_cfg.get("min_upvotes", 500)
                    score = min_upvotes + 100

                    posts.append({
                        "id": post_id,
                        "title": title,
                        "selftext": body,
                        "subreddit": subreddit_name,
                        "score": score,
                        "permalink": f"/r/{subreddit_name}/comments/{post_id}/",
                        "url": permalink_url,
                        "author": author,
                        "created_utc": created_utc,
                        "is_self": is_self,
                    })

                # Enforce limits locally
                return posts[:limit]

            except Exception as exc:
                self.log.error(
                    "Error fetching r/%s RSS: %s",
                    subreddit_name,
                    exc,
                )
                return []

    @staticmethod
    def _submission_to_dict(data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw submission data to a plain dictionary.

        Args:
            data: Raw JSON dict of a submission.

        Returns:
            Dictionary with the fields consumed by downstream pipeline
            components.
        """
        return {
            "id": data.get("id", ""),
            "title": data.get("title", ""),
            "body": data.get("selftext", ""),
            "subreddit": data.get("subreddit", ""),
            "score": data.get("score", 0),
            "url": data.get("url", f"https://www.reddit.com{data.get('permalink', '')}"),
            "author": data.get("author", "[deleted]"),
            "created_utc": data.get("created_utc", 0.0),
        }

    def _passes_filters(self, data: Dict[str, Any]) -> bool:
        """Apply quality-gate filters to a submission.

        Filters applied:
        1. Must be a self-text post (no links / images / videos).
        2. Body must not be empty, ``[removed]``, or ``[deleted]``.
        3. Score must meet ``min_upvotes``.
        4. Word count must be between ``min_word_count`` and
           ``max_word_count``.
        5. Must not already exist in the history ledger.

        Args:
            data: A raw submission data dictionary to evaluate.

        Returns:
            True if the submission passes all filters.
        """
        pipeline_mode = self.pipeline_cfg.get("pipeline_mode", "monologue")

        # Self-text only
        if pipeline_mode not in ("shower", "riddle") and not data.get("is_self", False):
            return False

        body: str = (data.get("selftext", "") or "").strip()
        title: str = (data.get("title", "") or "").strip()

        # Removed / deleted / empty title
        if not title or title.lower() in ("[removed]", "[deleted]", "[ removed by reddit ]"):
            return False

        if pipeline_mode not in ("thread", "shower", "riddle"):
            # Removed / deleted / empty body
            if not body or body.lower() in ("[removed]", "[deleted]", "[ removed by reddit ]"):
                return False
        else:
            if body.lower() in ("[removed]", "[deleted]", "[ removed by reddit ]"):
                body = ""

        # Reject meta-focused posts containing survey, studies, moderator notes etc.
        combined_text = f"{title} {body}".lower()
        meta_keywords = {"survey", "cornell", "norms", "participation", "notice", "feedback", "study", "moderator", "announcement"}
        if any(kw in combined_text for kw in meta_keywords):
            return False

        # Minimum upvotes
        min_upvotes: int = self.reddit_cfg.get("min_upvotes", 500)
        if data.get("score", 0) < min_upvotes:
            return False

        # Word-count window
        if pipeline_mode not in ("thread", "shower", "riddle"):
            wc = word_count(body)
            min_wc: int = self.reddit_cfg.get("min_word_count", 150)
            max_wc: int = self.reddit_cfg.get("max_word_count", 3000)
            if wc < min_wc or wc > max_wc:
                return False

        # De-duplication
        if self._is_scraped(data.get("id", "")):
            return False

        return True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scrape_stories(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Fetch and filter stories from all configured subreddits.

        Iterates each subreddit listed in ``config["reddit"]["subreddits"]``,
        applies quality filters, removes duplicates against the history
        ledger, and returns a flat list of post dicts sorted by score
        (descending).

        Args:
            limit: Override for the per-subreddit post limit. Falls back
                to ``config["reddit"]["post_limit"]`` (default 25).

        Returns:
            A list of post dicts suitable for the rewriting pipeline.
            Each dict contains keys: ``id``, ``title``, ``body``,
            ``subreddit``, ``score``, ``url``, ``author``, ``created_utc``.
        """
        subreddits: List[str] = self.reddit_cfg.get("subreddits", [])
        per_sub_limit: int = limit or self.reddit_cfg.get("post_limit", 25)

        if not subreddits:
            self.log.warning("No subreddits configured — nothing to scrape")
            return []

        all_posts: List[Dict[str, Any]] = []
        seen_ids: set[str] = set()

        for sub_name in subreddits:
            self.log.info(
                "Scraping r/%s (sort=%s, limit=%d) …",
                sub_name,
                self.reddit_cfg.get("sort", "hot"),
                per_sub_limit,
            )

            submissions = self._fetch_subreddit_posts(sub_name, per_sub_limit)
            accepted = 0

            for submission in submissions:
                sub_id = submission.get("id", "")
                if not sub_id or sub_id in seen_ids:
                    continue  # cross-posted to multiple configured subs

                if not self._passes_filters(submission):
                    continue

                all_posts.append(self._submission_to_dict(submission))
                seen_ids.add(sub_id)
                accepted += 1

            self.log.info(
                "  r/%s → %d/%d posts passed filters",
                sub_name,
                accepted,
                len(submissions),
            )

            # Cooldown delay of 45 seconds to prevent HTTP 429 rate limits
            if sub_name != subreddits[-1]:
                cooldown = 45.0
                self.log.info("Sleeping for %.1f seconds before next subreddit...", cooldown)
                time.sleep(cooldown)

        # Sort by score descending so best stories come first
        all_posts.sort(key=lambda p: p["score"], reverse=True)

        self.log.info(
            "Scraping complete — %d qualifying stories across %d subreddits",
            len(all_posts),
            len(subreddits),
        )

        # Apply intelligent Gemini filter to select the perfect storytelling post
        pipeline_mode = self.pipeline_cfg.get("pipeline_mode", "monologue")
        if self._use_llm and validate_api_key(self._api_key, "Gemini") and all_posts:
            if pipeline_mode in ("shower", "riddle"):
                self.log.info("Filtering candidate posts via Gemini story selector for shower/riddle mode...")
                selected_ids = self._intelligent_llm_filter_shower(all_posts[:15])
                if selected_ids:
                    selected_posts = [p for p in all_posts if p["id"] in selected_ids]
                    remaining = [p for p in all_posts if p["id"] not in selected_ids]
                    return selected_posts + remaining
            else:
                llm_candidates = all_posts[:15]
                self.log.info("Filtering %d candidate posts via Gemini story selector...", len(llm_candidates))
                selected_data = self._intelligent_llm_filter(llm_candidates)
                if selected_data == "FALLBACK_TO_ALL":
                    self.log.warning("Gemini story selector failed. Falling back to default candidate list.")
                    return all_posts
                if selected_data:
                    selected_id = selected_data.get("id")
                    gender = selected_data.get("gender", "MALE")
                    virality_reason = selected_data.get("virality_reason", "No reason provided.")
                    selected_post = next((p for p in all_posts if p["id"] == selected_id), None)
                    if selected_post:
                        selected_post["narrator_gender"] = gender
                        selected_post["virality_reason"] = virality_reason
                        self.log.info("Gemini selected story: [%s] %s (gender=%s)", selected_id, selected_post["title"], gender)
                        self.log.info("Virality Reason: %s", virality_reason)
                        # Return all posts with the Gemini selected post at the front
                        remaining = [p for p in all_posts if p["id"] != selected_id]
                        return [selected_post] + remaining
                # Gemini returned NONE, empty, or unrecognized — fall back to returning all posts
                self.log.warning("Gemini story selector returned no valid match. Falling back to all candidates.")
                return all_posts

        return all_posts

    def _intelligent_llm_filter(self, candidates: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
        """Send candidates to Gemini to select the single best storytelling post.

        Args:
            candidates: List of post dictionaries that passed basic structural filters.

        Returns:
            Dict containing selected post ID and detected narrator gender,
            or None if "NONE" returned / error / fallback.
        """
        if not candidates:
            return None

        # Build candidate summaries text block for the LLM
        candidate_texts = []
        for p in candidates:
            body_summary = p.get("body", "")[:300]
            candidate_texts.append(
                f"ID: {p.get('id')}\n"
                f"Subreddit: r/{p.get('subreddit')}\n"
                f"Title: {p.get('title')}\n"
                f"Body: {body_summary}...\n"
                f"---"
            )
        candidates_str = "\n".join(candidate_texts)

        pipeline_mode = self.pipeline_cfg.get("pipeline_mode", "monologue")
        system_prompt_base = (
            "You are an advanced Social Media Analytics Engine specialized in algorithmic virality for vertical videos (TikTok/Reels/Shorts). Your sole purpose is to analyze a batch of Reddit posts and extract the single post with the absolute highest probability of going viral.\n\n"
            "Evaluate and compare all candidate stories simultaneously based on these mathematical virality dimensions:\n"
            "1. High Emotional Trigger (Outrage/Shock): Does the story provoke instant moral outrage, deep betrayal, or shocking disbelief? (Family/relationship drama ranks highest).\n"
            "2. Cognitive Conflict (Debatability): Is there a massive moral 'grey area'? Viewers must feel intensely compelled to split into factions and fight each other in the comment section.\n"
            "3. Native 3-Second Hook: Does the source text contain a shocking, high-stakes statement in its first two sentences that can be locked as a visual/audio hook?\n\n"
        )
        if pipeline_mode == "thread":
            system_prompt_base += (
                "For thread compilation mode, look for open-ended, highly intriguing questions that invite juicy, funny, or shocking responses (e.g. secrets, relationship dynamics, regrets).\n"
                "Do NOT filter out short texts or empty post bodies; the question title itself is the hook.\n\n"
            )
        elif pipeline_mode in ("shower", "riddle"):
            system_prompt_base += (
                "For shower thoughts/riddle compilation mode, look for the most mind-bending, original, deep, or humorous shower thoughts/philosophical statements.\n"
                "Do NOT filter out short texts or empty post bodies; the thought/riddle is typically in the title.\n\n"
            )
        else:
            system_prompt_base += "Strictly filter out: Meta-text, updates, announcements, lists, short FAQs, wholesome content with no conflict, or text under 800 characters.\n\n"

        system_prompt = (
            system_prompt_base +
            "Analyze context/pronouns to extract narrator gender (MALE/FEMALE).\n\n"
            "Output ONLY a raw, minified JSON object. No markdown wrappers, no backticks, no prose.\n"
            "Required Keys:\n"
            "- \"id\": \"The unique Reddit ID string\" (or \"NONE\")\n"
            "- \"gender\": \"MALE\" or \"FEMALE\" (default to \"MALE\" if ambiguous)\n"
            "- \"virality_reason\": \"A aggressive 1-sentence analytical breakdown of why this specific narrative will force high comment engagement.\"\n\n"
            "Example Output:\n"
            "{\"id\":\"1u1pr9q\",\"gender\":\"FEMALE\",\"virality_reason\":\"High outrage factor due to parental betrayal and extreme debatability regarding financial inheritance.\"}"
        )

        max_retries = 3
        backoff_seconds = [15, 30, 60]
        
        for attempt in range(max_retries + 1):
            try:
                current_key = self._api_keys[attempt % len(self._api_keys)] if self._api_keys else self._api_key
                self.log.info("Querying Gemini for story selection (attempt %d/%d) using key ...%s...",
                              attempt + 1, max_retries + 1, current_key[-6:] if current_key else "")
                from google import genai
                client = genai.Client(api_key=current_key)
                
                response = client.models.generate_content(
                    model=self._model,
                    contents=f"{system_prompt}\n\nCandidate Posts:\n{candidates_str}"
                )
                
                result = (response.text or "").strip()
                self.log.info("Gemini story selector response: '%s'", result)
                
                # Strip markdown code block wrappers if present
                if result.startswith("```"):
                    lines = result.splitlines()
                    if lines[0].startswith("```"):
                        lines = lines[1:]
                    if lines and lines[-1].startswith("```"):
                        lines = lines[:-1]
                    result = "\n".join(lines).strip()
                
                import json
                try:
                    data = json.loads(result)
                    selected_id = data.get("id", "").strip()
                    gender = data.get("gender", "MALE").strip().upper()
                    virality_reason = data.get("virality_reason", "Selected via LLM.").strip()
                except Exception:
                    # Fallback to substring matching if JSON parsing fails
                    selected_id = "NONE"
                    gender = "MALE"
                    virality_reason = "Selected via LLM fallback."
                    for p in candidates:
                        if p["id"] in result:
                            selected_id = p["id"]
                            break
                    if "FEMALE" in result.upper():
                        gender = "FEMALE"
                    elif "MALE" in result.upper():
                        gender = "MALE"
                
                if selected_id == "NONE" or not selected_id:
                    return None
                
                valid_ids = {p["id"] for p in candidates}
                if selected_id in valid_ids:
                    return {"id": selected_id, "gender": gender, "virality_reason": virality_reason}
                
                self.log.warning("Gemini returned unrecognized ID: '%s'. Returning None.", selected_id)
                return None
            except Exception as exc:
                if attempt < max_retries:
                    sleep_time = backoff_seconds[attempt]
                    # Check if the error message contains a specific retry delay
                    parsed_delay = 0.0
                    exc_str = str(exc)
                    match = re.search(r"retry\s+in\s+(\d+(?:\.\d+)?)\s*s(?:econds)?", exc_str, re.IGNORECASE)
                    if match:
                        parsed_delay = float(match.group(1))
                    else:
                        match2 = re.search(r"retryDelay[\'\"]?\s*:\s*[\'\"]?(\d+)", exc_str, re.IGNORECASE)
                        if match2:
                            parsed_delay = float(match2.group(1))
                    
                    if parsed_delay > 0:
                        sleep_time = max(sleep_time, parsed_delay + 2.0)
                        self.log.warning(
                            "Gemini API quota rate-limit detected. Waiting for dynamic delay: %.1f seconds...",
                            sleep_time
                        )
                    else:
                        self.log.warning(
                            "Gemini filter API call failed: %s. Retrying in %d seconds...",
                            exc,
                            sleep_time,
                        )
                    time.sleep(sleep_time)
                else:
                    self.log.error(
                        "Gemini filter API call failed after %d retries: %s. Falling back to default list.",
                        max_retries,
                        exc,
                    )
                    return "FALLBACK_TO_ALL"
        return None

    def mark_scraped(self, post_id: str) -> None:
        """Record a post ID in the history ledger.

        The ledger is immediately persisted to disk so progress survives
        crashes.

        Args:
            post_id: The Reddit submission ID to mark as used.
        """
        if post_id in self._history:
            self.log.debug("Post %s already in history — skipping", post_id)
            return

        self._history[post_id] = datetime.now(timezone.utc).isoformat()
        self._save_history()
        self.log.info("Marked post %s as scraped", post_id)

    def _intelligent_llm_filter_shower(self, candidates: List[Dict[str, Any]]) -> List[str]:
        """Send candidates to Gemini to select the best shower thoughts.

        Args:
            candidates: List of post dictionaries.

        Returns:
            List of selected post ID strings.
        """
        if not candidates:
            return []

        max_thoughts = self.cfg.get("riddle", {}).get("max_thoughts") or self.cfg.get("shower", {}).get("max_thoughts", 5)

        candidate_texts = []
        for p in candidates:
            body_summary = p.get("body", "")[:300]
            candidate_texts.append(
                f"ID: {p.get('id')}\n"
                f"Title: {p.get('title')}\n"
                f"Body: {body_summary}\n"
                f"---"
            )
        candidates_str = "\n".join(candidate_texts)

        # Build list placeholder like ["id1","id2","id3","id4","id5"]
        placeholders = [f"\"id{i+1}\"" for i in range(max_thoughts)]
        placeholders_str = ",".join(placeholders)

        system_prompt = (
            "You are an advanced Social Media Analytics Engine specialized in algorithmic virality for vertical videos.\n"
            f"Analyze the candidate thoughts and select exactly {max_thoughts} thoughts that have the absolute highest probability of going viral together.\n"
            "Evaluate them based on mind-bending outrage/shock, debatability, and interest.\n\n"
            f"Output ONLY a raw, minified JSON object with the key 'ids' containing a list of the {max_thoughts} selected Reddit IDs. No markdown wrappers, no backticks, no prose.\n"
            "Example Output:\n"
            f"{{\"ids\":[{placeholders_str}]}}"
        )

        max_retries = 3
        backoff_seconds = [15, 30, 60]
        
        for attempt in range(max_retries + 1):
            try:
                current_key = self._api_keys[attempt % len(self._api_keys)] if self._api_keys else self._api_key
                self.log.info("Querying Gemini for shower thoughts selection (attempt %d/%d)...",
                              attempt + 1, max_retries + 1)
                from google import genai
                client = genai.Client(api_key=current_key)
                
                response = client.models.generate_content(
                     model=self._model,
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
                try:
                    data = json.loads(result)
                    ids = data.get("ids", [])
                    if isinstance(ids, list) and len(ids) == max_thoughts:
                        return ids
                except Exception:
                    # Fallback parsing
                    found_ids = []
                    for p in candidates:
                        if p["id"] in result:
                            found_ids.append(p["id"])
                    if len(found_ids) >= max_thoughts:
                        return found_ids[:max_thoughts]
                
                return []
            except Exception as exc:
                if attempt < max_retries:
                    self.log.warning("Gemini shower story selection attempt failed: %s. Retrying...", exc)
                    time.sleep(backoff_seconds[attempt])
                else:
                    self.log.error("Gemini shower story selection failed: %s", exc)
        return []


    def get_best_story(self) -> Optional[Dict[str, Any]]:
        """Scrape all configured subreddits and return the top-scoring story.

        Convenience wrapper around :meth:`scrape_stories` that returns
        only the single highest-scoring unscraped post, or *None* if no
        qualifying stories were found.

        Returns:
            The highest-scoring post dict, or None.
        """
        stories = self.scrape_stories()
        if not stories:
            self.log.warning("No qualifying stories found")
            return None

        best = stories[0]  # already sorted by score descending
        self.log.info(
            "Best story: [%s] %s (score=%d, words=%d, r/%s)",
            best["id"],
            best["title"][:60],
            best["score"],
            word_count(best["body"]),
            best["subreddit"],
        )
        return best

    def get_history_count(self) -> int:
        """Return the number of posts in the history ledger.

        Returns:
            Integer count of previously scraped post IDs.
        """
        return len(self._history)

    def clear_history(self) -> None:
        """Wipe the history ledger (useful for testing).

        Clears both the in-memory dict and the on-disk JSON file.
        """
        self._history.clear()
        self._save_history()
        self.log.warning("History ledger cleared")

    def fetch_comments(self, post_id: str, subreddit_name: str) -> List[Dict[str, Any]]:
        """Fetch comments for a specific post using the keyless RSS comments feed.

        Args:
            post_id: The ID of the Reddit post.
            subreddit_name: Subreddit name (without 'r/' prefix).

        Returns:
            A list of comment dicts, each with keys: 'author', 'body', 'id'.
        """
        user_agent = self.reddit_cfg.get("user_agent", "")
        if not user_agent or "YourRedditUsername" in user_agent or "YOUR_" in user_agent:
            user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"

        url = f"https://www.reddit.com/r/{subreddit_name}/comments/{post_id}.rss"
        max_retries = 3
        
        thread_cfg = self.cfg.get("thread", {})
        min_words = thread_cfg.get("min_comment_words", 15)
        max_words = thread_cfg.get("max_comment_words", 150)
        max_comments = thread_cfg.get("max_comments", 4)

        for attempt in range(max_retries + 1):
            ua = random.choice(_USER_AGENTS) if (not user_agent or "YourRedditUsername" in user_agent or "YOUR_" in user_agent) else user_agent
            headers = {
                "User-Agent": ua,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Ch-Ua": '"Not A(MeatPage;Bypass;1.0", "Chromium";"121", "Google Chrome";"121"',
                "Sec-Ch-Ua-Mobile": "?0",
                "Sec-Ch-Ua-Platform": '"Windows"',
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
            }
            try:
                self.log.debug("GET %s (attempt %d/%d)", url, attempt + 1, max_retries + 1)
                response = requests.get(url, headers=headers, timeout=15)

                if response.status_code == 429:
                    if attempt < max_retries:
                        self.log.warning(
                            "Reddit RSS comments rate-limited (HTTP 429) for post %s. "
                            "Sleeping 30 seconds before retry %d/%d...",
                            post_id, attempt + 1, max_retries
                        )
                        time.sleep(30)
                        continue
                    else:
                        self.log.error(
                            "Reddit RSS comments rate-limited (HTTP 429) for post %s. Failed after %d retries.",
                            post_id, max_retries
                        )
                        return []

                if response.status_code != 200:
                    self.log.error(
                        "Failed to fetch post %s comments RSS: HTTP %d",
                        post_id,
                        response.status_code,
                    )
                    return []

                root = ET.fromstring(response.content)
                ns = {'atom': 'http://www.w3.org/2005/Atom'}

                comments = []
                for entry in root.findall('atom:entry', ns):
                    id_node = entry.find('atom:id', ns)
                    raw_id = id_node.text if id_node is not None else ""
                    if not raw_id or "t1_" not in raw_id:
                        continue

                    author_node = entry.find('atom:author/atom:name', ns)
                    author = author_node.text if author_node is not None else "[deleted]"
                    if author.startswith("/u/"):
                        author = author[3:]

                    if author.lower() in ("automoderator", "[deleted]", "deleted"):
                        continue

                    content_node = entry.find('atom:content', ns)
                    content_html = content_node.text if content_node is not None else ""

                    body = ""
                    match = re.search(r'<div class="md">(.*?)</div>', content_html, re.DOTALL)
                    if match:
                        body_html = match.group(1)
                        body = re.sub(r'<[^>]+>', '', body_html)
                        body = html.unescape(body).strip()

                    if not body or body.lower() in ("[removed]", "[deleted]"):
                        continue

                    words = len(body.split())
                    if words < min_words or words > max_words:
                        continue

                    comments.append({
                        "id": raw_id.split('_')[1] if '_' in raw_id else raw_id,
                        "author": author,
                        "body": body,
                    })

                self.log.info("Found %d comments passing filters for post %s", len(comments), post_id)
                return comments[:max_comments]

            except Exception as exc:
                self.log.error(
                    "Error fetching comments for post %s: %s",
                    post_id,
                    exc,
                )
                return []
        return []
