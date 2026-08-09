"""
RedditDaily-Bot — YouTube Shorts Publisher
==========================================
Publishes rendered Reels directly to YouTube Shorts via YouTube Data API v3.
Uses OAuth 2.0 Refresh Tokens for 100% headless, unattended automation.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import BotLogger


class YouTubeShortsPublisher:
    """Uploads finished vertical videos to YouTube Shorts.

    Attributes:
        config: Master application config dictionary.
        logger: BotLogger instance.
        enabled: Whether YouTube auto-posting is active.
    """

    def __init__(self, config: Dict[str, Any], logger: Optional[BotLogger] = None):
        self.config = config
        self.logger = logger or BotLogger(
            name="YouTubePublisher",
            log_dir=config.get("pipeline", {}).get("log_dir"),
            level=config.get("pipeline", {}).get("log_level", "INFO"),
        )

        yt_cfg = config.get("youtube", {})
        self.enabled: bool = bool(yt_cfg.get("enabled", False) or "YOUTUBE_REFRESH_TOKEN" in os.environ)
        self.privacy_status: str = yt_cfg.get("privacy_status", "public")
        self.category_id: str = str(yt_cfg.get("category_id", "24"))  # 24 = Entertainment

        # Resolve credentials from environment or config or local file
        def _clean(val: Any) -> str:
            s = str(val or "").strip()
            if s.startswith("YOUR_") or s.endswith("_HERE"):
                return ""
            return s

        self.client_id = _clean(os.environ.get("YOUTUBE_CLIENT_ID", "")) or _clean(yt_cfg.get("client_id", ""))
        self.client_secret = _clean(os.environ.get("YOUTUBE_CLIENT_SECRET", "")) or _clean(yt_cfg.get("client_secret", ""))
        self.refresh_token = _clean(os.environ.get("YOUTUBE_REFRESH_TOKEN", "")) or _clean(yt_cfg.get("refresh_token", ""))

        # Fallback to local youtube_credentials.json if exists
        if not (self.client_id and self.client_secret and self.refresh_token):
            local_creds_path = Path("youtube_credentials.json")
            if local_creds_path.exists():
                try:
                    import json
                    with open(local_creds_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.client_id = self.client_id or _clean(data.get("client_id", ""))
                        self.client_secret = self.client_secret or _clean(data.get("client_secret", ""))
                        self.refresh_token = self.refresh_token or _clean(data.get("refresh_token", ""))
                except Exception:
                    pass

        if self.enabled and not (self.client_id and self.client_secret and self.refresh_token):
            self.logger.warning("YouTube auto-posting enabled, but YOUTUBE credentials are missing or incomplete.")
            self.enabled = False

    def upload_short(
        self,
        video_path: Path,
        title: str,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Upload a video file as a YouTube Short.

        Args:
            video_path: Path to the local .mp4 video file.
            title: Title for the Short.
            description: Video description text.
            tags: List of keyword tags.

        Returns:
            The YouTube video ID on success, or None on failure.
        """
        if not self.enabled:
            self.logger.info("YouTube Shorts publishing is disabled in configuration.")
            return None

        if not video_path.exists():
            self.logger.error(f"Video file does not exist: {video_path}")
            return None

        file_size = video_path.stat().st_size
        self.logger.info(f"Initializing YouTube Shorts upload: '{video_path.name}' ({file_size} bytes)...")

        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaFileUpload

            creds = Credentials(
                token=None,
                refresh_token=self.refresh_token,
                token_uri="https://oauth2.googleapis.com/token",
                client_id=self.client_id,
                client_secret=self.client_secret,
                scopes=[
                    "https://www.googleapis.com/auth/youtube.upload",
                    "https://www.googleapis.com/auth/youtube",
                ],
            )

            youtube = build("youtube", "v3", credentials=creds)

            # Ensure title contains #Shorts and is under 100 characters
            clean_title = title.strip()
            if "#Shorts" not in clean_title and "#shorts" not in clean_title:
                clean_title = f"{clean_title} #Shorts"
            if len(clean_title) > 100:
                clean_title = clean_title[:92].strip() + "… #Shorts"

            # Prepare description
            clean_desc = description.strip()
            if not clean_desc:
                clean_desc = f"{clean_title}\n\n#Shorts #storytime #redditstories #reddit #truestory"
            elif "#Shorts" not in clean_desc and "#shorts" not in clean_desc:
                clean_desc = f"{clean_desc}\n\n#Shorts #storytime #redditstories #reddit"

            # Default tags
            default_tags = ["shorts", "reddit", "storytime", "redditstories", "truestory", "drama"]
            if tags:
                all_tags = list(set(tags + default_tags))
            else:
                all_tags = default_tags

            body = {
                "snippet": {
                    "title": clean_title,
                    "description": clean_desc,
                    "tags": all_tags,
                    "categoryId": self.category_id,
                },
                "status": {
                    "privacyStatus": self.privacy_status,
                    "selfDeclaredMadeForKids": False,
                },
            }

            media = MediaFileUpload(
                str(video_path),
                chunksize=1024 * 1024 * 5,  # 5MB chunks
                resumable=True,
                mimetype="video/mp4",
            )

            request = youtube.videos().insert(
                part="snippet,status",
                body=body,
                media_body=media,
            )

            response = None
            self.logger.info("Uploading video chunks to YouTube...")

            while response is None:
                status, response = request.next_chunk()
                if status:
                    progress = int(status.progress() * 100)
                    self.logger.info(f"  YouTube upload progress: {progress}%")

            video_id = response.get("id")
            if video_id:
                short_url = f"https://youtube.com/shorts/{video_id}"
                self.logger.info(f"✓ YouTube Short successfully published! URL: {short_url}")
                return video_id
            else:
                self.logger.warning(f"YouTube upload response missing video ID: {response}")
                return None

        except Exception as exc:
            self.logger.error(f"YouTube Shorts upload failed: {exc}")
            return None
