import os
import requests
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.utils import BotLogger

class FacebookReelsPublisher:
    """Handles publishing finalized video Reels to a Facebook Page using Meta's Video Reels API."""

    def __init__(self, config: Dict[str, Any], logger: Optional[BotLogger] = None, pipeline_mode: Optional[str] = None):
        self.config = config
        self.log = logger or BotLogger("FacebookPublisher")
        fb_cfg = self.config.get("facebook", {})
        self.page_id = ""
        self.access_token = ""

        # First check if facebook.pages array exists
        pages = fb_cfg.get("pages", [])
        if pages and isinstance(pages, list):
            page_obj = None
            if pipeline_mode == "monologue" and len(pages) >= 1:
                page_obj = pages[0]
            elif pipeline_mode == "conversational" and len(pages) >= 2:
                page_obj = pages[1]
            elif pipeline_mode == "riddle" and len(pages) >= 3:
                page_obj = pages[2]
            if page_obj and isinstance(page_obj, dict):
                self.page_id = str(page_obj.get("page_id", "")).strip()
                self.access_token = str(page_obj.get("access_token", "")).strip()

        # Fallback to direct key mappings if pages array is not defined or missing fields
        if not self.page_id or not self.access_token:
            if pipeline_mode == "conversational":
                self.page_id = str(fb_cfg.get("conversational_page_id", fb_cfg.get("page_id", ""))).strip()
                self.access_token = str(fb_cfg.get("conversational_page_access_token", fb_cfg.get("page_access_token", ""))).strip()
            elif pipeline_mode == "monologue":
                self.page_id = str(fb_cfg.get("monologue_page_id", fb_cfg.get("page_id", ""))).strip()
                self.access_token = str(fb_cfg.get("monologue_page_access_token", fb_cfg.get("page_access_token", ""))).strip()
            else:
                self.page_id = str(fb_cfg.get("page_id", "")).strip()
                self.access_token = str(fb_cfg.get("page_access_token", "")).strip()

        # Check for proxy configuration in page_obj or fb_cfg
        proxy_str = ""
        if 'page_obj' in locals() and page_obj and isinstance(page_obj, dict):
            proxy_str = str(page_obj.get("proxy", "")).strip()
        if not proxy_str:
            proxy_str = str(fb_cfg.get("proxy", "")).strip()

        if proxy_str:
            self.proxies = {"http": proxy_str, "https": proxy_str}
            self.log.info(f"Using proxy for Facebook Page upload: {proxy_str.split('@')[-1] if '@' in proxy_str else proxy_str}")
        else:
            self.proxies = None

        self.enabled = fb_cfg.get("enabled", False)

    def verify_proxy(self) -> Dict[str, Any]:
        """Verify if the configured proxy is working and return the resolved egress IP address."""
        if not self.proxies:
            return {"status": "NO_PROXY", "ip": "Direct Connection", "working": True}

        try:
            res = requests.get("https://api.ipify.org?format=json", proxies=self.proxies, timeout=12)
            if res.status_code == 200:
                ip = res.json().get("ip", "Unknown")
                self.log.info(f"✓ Proxy verification SUCCESS for Page {self.page_id} — Egress IP: {ip}")
                return {"status": "SUCCESS", "ip": ip, "working": True}
            else:
                self.log.error(f"❌ Proxy verification FAILED (HTTP {res.status_code}) for Page {self.page_id}")
                return {"status": f"HTTP_{res.status_code}", "ip": None, "working": False}
        except Exception as exc:
            self.log.error(f"❌ Proxy connection error for Page {self.page_id}: {exc}")
            return {"status": "ERROR", "error": str(exc), "working": False}

    @classmethod
    def verify_all_pages_proxies(cls, config: Dict[str, Any], logger: Optional[BotLogger] = None) -> List[Dict[str, Any]]:
        """Verify proxies across all configured Facebook pages."""
        log = logger or BotLogger("ProxyVerifier")
        fb_cfg = config.get("facebook", {})
        pages = fb_cfg.get("pages", [])
        results = []

        log.info("=" * 60)
        log.info("  Facebook Page Proxy Egress IP Verification Suite")
        log.info("=" * 60)

        if not pages or not isinstance(pages, list):
            log.warning("No facebook.pages array found in config.yaml.")
            return results

        for idx, page_info in enumerate(pages, start=1):
            page_name = page_info.get("page_name", f"Page #{idx}")
            page_id = page_info.get("page_id", "N/A")
            proxy_str = page_info.get("proxy", "").strip()

            if not proxy_str:
                log.info(f"[{idx}/{len(pages)}] Page '{page_name}' ({page_id}): Direct Connection (No proxy set)")
                results.append({"page_name": page_name, "page_id": page_id, "proxy": None, "ip": "Direct", "status": "NO_PROXY"})
                continue

            proxies = {"http": proxy_str, "https": proxy_str}
            try:
                res = requests.get("https://api.ipify.org?format=json", proxies=proxies, timeout=12)
                if res.status_code == 200:
                    ip = res.json().get("ip", "Unknown")
                    log.info(f"[{idx}/{len(pages)}] Page '{page_name}' ({page_id}): ✅ PROXY WORKING | Egress IP: {ip}")
                    results.append({"page_name": page_name, "page_id": page_id, "proxy": proxy_str, "ip": ip, "status": "WORKING"})
                else:
                    log.error(f"[{idx}/{len(pages)}] Page '{page_name}' ({page_id}): ❌ PROXY FAILED (HTTP {res.status_code})")
                    results.append({"page_name": page_name, "page_id": page_id, "proxy": proxy_str, "ip": None, "status": "FAILED"})
            except Exception as exc:
                log.error(f"[{idx}/{len(pages)}] Page '{page_name}' ({page_id}): ❌ PROXY CONNECTION ERROR ({exc})")
                results.append({"page_name": page_name, "page_id": page_id, "proxy": proxy_str, "ip": None, "status": "ERROR", "error": str(exc)})

        log.info("=" * 60)
        return results

    def publish_reel(self, video_path: Path, caption: str, pinned_comment: Optional[str] = None, scheduled_publish_time: Optional[int] = None) -> bool:
        """Uploads and publishes a vertical MP4 video as a Reel on the configured Facebook Page.

        Args:
            video_path: Absolute path to the finalized video file.
            caption: Reel text description and hashtags.
            pinned_comment: Optional first comment text to post under the Reel.
            scheduled_publish_time: Optional UNIX timestamp for scheduled publishing.

        Returns:
            True if the Reel was successfully processed and published/scheduled, False otherwise.
        """
        if not self.enabled:
            self.log.warning("Facebook Reels publishing is disabled in configuration.")
            return False

        if not self.page_id or not self.access_token:
            self.log.error("Facebook page_id or page_access_token is missing.")
            return False

        if not video_path.exists():
            self.log.error(f"Video file not found for upload: {video_path}")
            return False

        file_size = video_path.stat().st_size
        self.log.info(f"Initializing Reel upload to Page ID: {self.page_id} (file size: {file_size} bytes)...")

        # ----------------------------------------------------
        # Step 1: START phase (Register the Reel upload session)
        # ----------------------------------------------------
        start_url = f"https://graph.facebook.com/v19.0/{self.page_id}/video_reels"
        start_payload = {
            "upload_phase": "START",
            "access_token": self.access_token
        }

        try:
            res_start = requests.post(start_url, data=start_payload, timeout=30, proxies=self.proxies)
            res_start_json = res_start.json()
            if res_start.status_code != 200 or "video_id" not in res_start_json:
                self.log.error(f"START phase failed: {res_start.status_code} - {res_start.text}")
                return False

            video_id = res_start_json["video_id"]
            upload_url = res_start_json["upload_url"]
            self.log.info(f"START phase successful. video_id: {video_id}")
        except Exception as exc:
            self.log.error(f"Network error in START phase: {exc}")
            return False

        # ----------------------------------------------------
        # Step 2: UPLOAD phase (Transmit raw binary bytes)
        # ----------------------------------------------------
        self.log.info("Uploading binary video data to Facebook upload server...")
        upload_headers = {
            "Authorization": f"OAuth {self.access_token}",
            "offset": "0",
            "file_size": str(file_size),
            "Content-Type": "application/octet-stream"
        }

        try:
            with open(video_path, "rb") as video_file:
                res_upload = requests.post(upload_url, headers=upload_headers, data=video_file, timeout=180, proxies=self.proxies)

            if res_upload.status_code != 200:
                self.log.error(f"UPLOAD phase failed: {res_upload.status_code} - {res_upload.text}")
                return False

            self.log.info("UPLOAD phase successful. Binary data transferred.")
        except Exception as exc:
            self.log.error(f"Network error in UPLOAD phase: {exc}")
            return False

        # ----------------------------------------------------
        # Step 3: FINISH phase (Publish/Schedule the Reel with caption)
        # ----------------------------------------------------
        if scheduled_publish_time:
            self.log.info(f"Scheduling Reel for UNIX timestamp {scheduled_publish_time} with caption: '{caption}'...")
            finish_payload = {
                "upload_phase": "FINISH",
                "video_id": video_id,
                "video_state": "SCHEDULED",
                "scheduled_publish_time": str(scheduled_publish_time),
                "description": caption,
                "access_token": self.access_token
            }
        else:
            self.log.info(f"Publishing Reel immediately with caption: '{caption}'...")
            finish_payload = {
                "upload_phase": "FINISH",
                "video_id": video_id,
                "video_state": "PUBLISHED",
                "description": caption,
                "access_token": self.access_token
            }

        try:
            import time
            max_finish_retries = 4
            finish_success = False
            res_finish_json = {}
            
            for attempt in range(1, max_finish_retries + 1):
                try:
                    self.log.info(f"Sending FINISH request (attempt {attempt}/{max_finish_retries})...")
                    res_finish = requests.post(start_url, data=finish_payload, timeout=45, proxies=self.proxies)
                    res_finish_json = res_finish.json()
                    
                    if res_finish.status_code == 200 and res_finish_json.get("success", False):
                        finish_success = True
                        break
                    
                    err_msg = res_finish_json.get("error", {}).get("message", "")
                    err_code = res_finish_json.get("error", {}).get("code", 0)
                    self.log.warning(f"FINISH attempt {attempt} returned status {res_finish.status_code}: Code {err_code} - {err_msg}")
                    
                    # Check if it was actually published/scheduled regardless of the status code (some transient responses still succeed)
                    status_url = f"https://graph.facebook.com/v19.0/{video_id}"
                    status_params = {"fields": "status", "access_token": self.access_token}
                    try:
                        chk = requests.get(status_url, params=status_params, timeout=10, proxies=self.proxies)
                        if chk.status_code == 200 and chk.json().get("status", {}).get("video_status") in ("ready", "processing"):
                            self.log.info(f"Verified Reel exists and is processing on Facebook (video_status={chk.json().get('status', {}).get('video_status')}). Treating as success.")
                            finish_success = True
                            break
                    except Exception:
                        pass
                    
                    if attempt < max_finish_retries:
                        sleep_time = 15 * attempt
                        self.log.info(f"Waiting {sleep_time} seconds before retrying FINISH phase...")
                        time.sleep(sleep_time)
                except Exception as finish_exc:
                    self.log.warning(f"Exception during FINISH attempt {attempt}: {finish_exc}")
                    if attempt < max_finish_retries:
                        sleep_time = 15 * attempt
                        time.sleep(sleep_time)

            if not finish_success:
                self.log.error("All FINISH phase attempts failed.")
                return False

            if scheduled_publish_time:
                self.log.info(f"Reel successfully scheduled! Facebook Reel video ID: {video_id}")
            else:
                self.log.info(f"Reel successfully published! Facebook Reel video ID: {video_id}")
            
            # Post the teaser first comment if provided (only if not scheduled)
            if pinned_comment and not scheduled_publish_time:
                self.log.info(f"Posting teaser first comment: '{pinned_comment}'...")
                
                # Poll video status until ready or timeout (max 180s)
                status_url = f"https://graph.facebook.com/v19.0/{video_id}"
                status_params = {
                    "fields": "status",
                    "access_token": self.access_token
                }
                
                import time
                video_ready = False
                start_poll = time.time()
                self.log.info("Waiting for Facebook to process the Reel video before commenting...")
                
                while time.time() - start_poll < 180:
                    try:
                        res_status = requests.get(status_url, params=status_params, timeout=15, proxies=self.proxies)
                        if res_status.status_code == 200:
                            status_data = res_status.json().get("status", {})
                            video_status = status_data.get("video_status")
                            progress = status_data.get("processing_progress", 0)
                            
                            self.log.info(f"Reel status check: {video_status} ({progress}% processed)")
                            if video_status == "ready":
                                video_ready = True
                                break
                            elif video_status == "error":
                                self.log.error("Facebook backend failed to process the Reel video.")
                                break
                        else:
                            self.log.warning(f"Failed to check Reel status: {res_status.status_code} - {res_status.text}")
                    except Exception as poll_exc:
                        self.log.warning(f"Error checking Reel status: {poll_exc}")
                    
                    time.sleep(10)
                
                if not video_ready:
                    self.log.warning("Timeout or error waiting for video processing. Attempting comment posting anyway...")
                
                comment_url = f"https://graph.facebook.com/v19.0/{video_id}/comments"
                comment_payload = {
                    "message": pinned_comment,
                    "access_token": self.access_token
                }
                try:
                    res_comment = requests.post(comment_url, data=comment_payload, timeout=30, proxies=self.proxies)
                    if res_comment.status_code == 200:
                        self.log.info("Teaser first comment successfully posted!")
                    else:
                        self.log.warning(f"Failed to post first comment: {res_comment.status_code} - {res_comment.text}")
                except Exception as c_exc:
                    self.log.error(f"Network error posting first comment: {c_exc}")
                    
            return True
        except Exception as exc:
            self.log.error(f"Network error in FINISH phase: {exc}")
            return False
