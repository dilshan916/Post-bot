"""
RedditDaily-Bot — Reddit Scraper & History Manager
====================================================
Connects to Reddit directly via public RSS feeds using the requests library,
fetches self-text posts from configured subreddits, applies quality filters,
and de-duplicates against a persistent JSON ledger so stories are never re-used.
"""

from __future__ import annotations

import time
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

        headers = {
            "User-Agent": user_agent,
            "Accept": "application/atom+xml,application/xml,text/xml",
        }

        # Use public RSS feed to bypass JSON block
        url = f"https://www.reddit.com/r/{subreddit_name}/{sort_method}.rss"
        params: Dict[str, Any] = {}
        if sort_method == "top":
            params["t"] = time_filter

        try:
            self.log.debug("GET %s with params %s", url, params)
            response = requests.get(url, headers=headers, params=params, timeout=15)

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
        # Self-text only
        if not data.get("is_self", False):
            return False

        body: str = (data.get("selftext", "") or "").strip()
        title: str = (data.get("title", "") or "").strip()

        # Removed / deleted / empty body or title
        if not body or body.lower() in ("[removed]", "[deleted]", "[ removed by reddit ]"):
            return False
        if not title or title.lower() in ("[removed]", "[deleted]", "[ removed by reddit ]"):
            return False

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

            # Small courtesy delay between subreddit requests to prevent HTTP 429 rate limits
            time.sleep(3.0)

        # Sort by score descending so best stories come first
        all_posts.sort(key=lambda p: p["score"], reverse=True)

        self.log.info(
            "Scraping complete — %d qualifying stories across %d subreddits",
            len(all_posts),
            len(subreddits),
        )

        # Apply intelligent Gemini filter to select the perfect storytelling post
        if self._use_llm and validate_api_key(self._api_key, "Gemini") and all_posts:
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
                    return [selected_post]
            # Gemini returned NONE, empty, or unrecognized — fall back to top candidates
            self.log.warning("Gemini story selector returned no valid match. Falling back to top candidate.")
            return all_posts[:1]

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

        system_prompt = (
            "You are an advanced Social Media Analytics Engine specialized in algorithmic virality for vertical videos (TikTok/Reels/Shorts). Your sole purpose is to analyze a batch of Reddit posts and extract the single post with the absolute highest probability of going viral.\n\n"
            "Evaluate and compare all candidate stories simultaneously based on these mathematical virality dimensions:\n"
            "1. High Emotional Trigger (Outrage/Shock): Does the story provoke instant moral outrage, deep betrayal, or shocking disbelief? (Family/relationship drama ranks highest).\n"
            "2. Cognitive Conflict (Debatability): Is there a massive moral 'grey area'? Viewers must feel intensely compelled to split into factions and fight each other in the comment section.\n"
            "3. Native 3-Second Hook: Does the source text contain a shocking, high-stakes statement in its first two sentences that can be locked as a visual/audio hook?\n\n"
            "Strictly filter out: Meta-text, updates, announcements, lists, short FAQs, wholesome content with no conflict, or text under 800 characters.\n\n"
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
