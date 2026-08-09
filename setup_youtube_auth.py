#!/usr/bin/env python3
"""
Setup YouTube OAuth 2.0 Authentication
======================================
Interactive helper to generate your permanent YouTube Refresh Token
for automated video uploads to YouTube Shorts.
"""

import glob
import json
import os
import sys
from pathlib import Path
from colorama import Fore, Style, init

init(autoreset=True)

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]


def main():
    print(Fore.CYAN + Style.BRIGHT + "=" * 65)
    print(Fore.CYAN + Style.BRIGHT + "  YouTube Shorts OAuth 2.0 Setup Helper")
    print(Fore.CYAN + Style.BRIGHT + "=" * 65)
    print()

    try:
        from google_auth_oauthlib.flow import InstalledAppFlow
    except ImportError:
        print(Fore.RED + "Missing dependency. Installing google-auth-oauthlib...")
        os.system("pip install google-auth-oauthlib google-api-python-client")
        from google_auth_oauthlib.flow import InstalledAppFlow

    client_id = ""
    client_secret = ""

    # Check if a client_secret JSON file exists in current directory or Downloads
    json_candidates = glob.glob("client_secret*.json")
    if not json_candidates:
        downloads_dir = Path.home() / "Downloads"
        if downloads_dir.exists():
            json_candidates = glob.glob(str(downloads_dir / "client_secret*.json"))

    client_config = None

    if json_candidates:
        chosen_json = json_candidates[0]
        print(Fore.GREEN + f"✓ Found downloaded OAuth client file: {chosen_json}")
        try:
            with open(chosen_json, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "installed" in data:
                    client_id = data["installed"].get("client_id", "")
                    client_secret = data["installed"].get("client_secret", "")
                    client_config = data
                elif "web" in data:
                    client_id = data["web"].get("client_id", "")
                    client_secret = data["web"].get("client_secret", "")
                    client_config = data
        except Exception as e:
            print(Fore.RED + f"Failed to read JSON: {e}")

    if not client_config or not client_id or not client_secret:
        print(Fore.YELLOW + "Please paste your OAuth Client details from Google Cloud:")
        client_id = input(Fore.GREEN + "Enter Client ID: ").strip()
        client_secret = input(Fore.GREEN + "Enter Client Secret: ").strip()

        if not client_id or not client_secret:
            print(Fore.RED + "Client ID and Client Secret are required. Aborting.")
            return 1

        client_config = {
            "installed": {
                "client_id": client_id,
                "client_secret": client_secret,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
            }
        }

    print()
    print(Fore.CYAN + "Launching Google authorization in your default web browser...")
    print(Fore.YELLOW + "1. Log in with your YouTube channel Google account.")
    print(Fore.YELLOW + "2. Click 'Continue' / 'Allow' when prompted.")
    print()

    flow = InstalledAppFlow.from_client_config(
        client_config,
        scopes=SCOPES,
    )

    creds = flow.run_local_server(port=8080, prompt="consent", access_type="offline")

    refresh_token = creds.refresh_token

    if not refresh_token:
        print(Fore.RED + "Error: Could not retrieve Refresh Token. Please try again.")
        return 1

    print()
    print(Fore.GREEN + Style.BRIGHT + "=" * 65)
    print(Fore.GREEN + Style.BRIGHT + "  🎉 SUCCESS! YouTube Authentication Complete!")
    print(Fore.GREEN + Style.BRIGHT + "=" * 65)
    print()
    print(Fore.WHITE + "Add these 3 secrets to your GitHub repository secrets:")
    print(Fore.WHITE + "(GitHub -> Settings -> Secrets and variables -> Actions -> New secret)")
    print()
    print(Fore.CYAN + "1. Secret Name: " + Fore.YELLOW + Style.BRIGHT + "YOUTUBE_CLIENT_ID")
    print(Fore.WHITE + f"   Value: {client_id}")
    print()
    print(Fore.CYAN + "2. Secret Name: " + Fore.YELLOW + Style.BRIGHT + "YOUTUBE_CLIENT_SECRET")
    print(Fore.WHITE + f"   Value: {client_secret}")
    print()
    print(Fore.CYAN + "3. Secret Name: " + Fore.YELLOW + Style.BRIGHT + "YOUTUBE_REFRESH_TOKEN")
    print(Fore.GREEN + Style.BRIGHT + f"   Value: {refresh_token}")
    print()
    print(Fore.GREEN + Style.BRIGHT + "=" * 65)

    # Save to local youtube_credentials.json for local testing
    creds_data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
    }
    with open("youtube_credentials.json", "w", encoding="utf-8") as f:
        json.dump(creds_data, f, indent=2)
    print(Fore.GREEN + "✓ Also saved locally to 'youtube_credentials.json' for local posting.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
