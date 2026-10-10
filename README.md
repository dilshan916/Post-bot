# 🤖 Post-bot — Autonomous Viral Video Generator & Social Media Scheduler

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Local TTS Engine](https://img.shields.io/badge/TTS-Kokoro--82M%20ONNX-3b82f6?style=for-the-badge&logo=onnx&logoColor=white)](https://huggingface.co/hexgrad/Kokoro-82M)
[![LLM Reasoning](https://img.shields.io/badge/AI-Gemini%202.5%20Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Fast LLM Rewriter](https://img.shields.io/badge/LLM-Groq%20Llama%203.3%2070B-f55036?style=for-the-badge&logo=fastapi&logoColor=white)](https://groq.com/)
[![Video Pipeline](https://img.shields.io/badge/Video-FFmpeg%20%7C%20MoviePy-0078d4?style=for-the-badge&logo=ffmpeg&logoColor=white)](https://ffmpeg.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

### **End-to-End Vertical Video Generation, Neural Voice Narration & Facebook Reels Publishing Automation**
*Scrapes trending Reddit stories, rewrites high-retention scripts with dual LLMs, synthesizes offline Kokoro-82M neural voices, renders dynamic word-by-word subtitles, and auto-schedules directly to Facebook Pages.*

</div>

---

## 🌟 Overview

**Post-bot** is an enterprise-grade automated content generation engine built in Python. Designed for automated social media growth channels, it compiles Reddit stories, conversational text dramas, AskReddit threads, and viral riddles into broadcast-ready **9:16 vertical video Reels** complete with dynamic gameplay backgrounds and synchronized kinetic subtitles.

The bot employs a **Hybrid LLM Architecture** (Google Gemini 2.5 Flash for deep viral story analysis & ranking + Groq Llama 3.3 70B for high-retention script rewriting with automated multi-key rotation), a **Dual TTS Engine** (offline local Kokoro-82M ONNX neural speech with cloud Edge-TTS fallback), and direct **Facebook Graph API** auto-scheduling in Sri Lankan Time (SLT UTC+05:30) with automated hardware cooldown protection.

---

## ✨ Key Features

### 🎬 5 Production Pipeline Modes
* **Mode 1 — Monologue Mode**: Single-narrator viral narrative with screenshot hook title cards, dynamic background music ducking, and word-by-word active subtitles.
* **Mode 2 — Conversational Mode**: Multi-character text-message drama scripts with distinct neural character assignments (`MALE`, `FEMALE`, `OLD_FEMALE`, `OLD_MALE`, `CHILD_MALE`, `CHILD_FEMALE`).
* **Mode 3 — AskReddit Thread Mode**: Curated high-upvote thread discussions compiled into visual commentary reels with author avatars and upvote badges.
* **Mode 4 — Interactive Riddle Mode**: Brain-teaser riddles generated on the fly via Gemini Flash featuring animated suspense countdown timers.
* **Mode 5 — Batch Hybrid Scheduler (Full Automation)**: End-to-end autopilot that generates and schedules up to **42 Reels across 7 days** (2 daily drops at **09:30 AM** and **07:30 PM SLT**) with CLI script approval (`y`/`n`), zero comment clutter, and GPU/CPU thermal cooldown protection.

### 🎙️ Dual TTS Engine Architecture
* **Kokoro-82M (Local Offline ONNX)**: Ultra-realistic offline text-to-speech utilizing `CPUExecutionProvider` for 100% stability across all systems. Supports distinct voice assignments (`am_adam`, `af_bella`, `bm_george`, `af_nicole`, `am_puck`, `af_sky`).
* **Edge-TTS (Cloud Fallback)**: Zero-configuration cloud fallback engine that automatically engages if local Kokoro model assets are absent.

### 🧠 Hybrid LLM Engine with Key Rotation
* **Google Gemini 2.5 Flash**: Analytical engine that scans candidate Reddit submissions and scores them on emotional triggers, virality, relatability, and retention probability.
* **Groq Llama 3.3 70B**: High-speed creative script rewriter enforcing strict retention formulas (high-stakes openers, curiosity-gap teasers, delayed escalation reveals) with automatic multi-key rotation upon rate limits.

### 🎨 Visual Subtitle & Rendering Suite
* **Active Word Kinetic Subtitles**: Double-pass subtitle generation with colored word highlighting, custom typography, and drop shadows for maximum watch time.
* **Automated Background Randomizer**: Selects and trims high-bitrate vertical gameplay footage seamlessly.
* **Playwright Screenshot Engine**: Renders crisp browser-based Reddit UI title cards and comment headers with authentic fonts and dark mode themes.

---

## 🏗️ Technical Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Reddit Submission Scraper                 │
│         (r/relationship_advice, r/AmItheAsshole, etc.)      │
└──────────────────────────────┬──────────────────────────────┘
                               │ Raw Submission Text
┌──────────────────────────────▼──────────────────────────────┐
│                    Hybrid LLM Pipeline                      │
├──────────────────────────────┬──────────────────────────────┤
│     Gemini 2.5 Flash         │      Groq Llama 3.3 70B      │
│  (Story Scoring & Virality)  │ (Retention Script Rewriter)  │
└──────────────────────────────┬──────────────────────────────┘
                               │ Approved Production Script
┌──────────────────────────────▼──────────────────────────────┐
│                  Audio & Speech Synthesis                   │
├──────────────────────────────┬──────────────────────────────┤
│      Kokoro-82M ONNX         │      Edge-TTS (Fallback)     │
│   (Local CPU Multi-Voice)    │     (Cloud Neural Audio)     │
└──────────────────────────────┬──────────────────────────────┘
                               │ Master Audio + Word Timings
┌──────────────────────────────▼──────────────────────────────┐
│               Video Compositor & Rendering Engine           │
│   (MoviePy • FFmpeg • Kinetic Subtitles • Gameplay Cuts)    │
└──────────────────────────────┬──────────────────────────────┘
                               │ Final 1080x1920 MP4 Video
┌──────────────────────────────▼──────────────────────────────┐
│               Facebook Graph API Auto-Scheduler             │
│   (Daily Batches • Sri Lanka Time SLT • Cooldown Checks)    │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Getting Started

### 1. Prerequisites
* **Python**: `3.10` or higher installed.
* **FFmpeg**: Installed and configured in system `PATH`.
* **Git**: Installed.

### 2. Installation

```powershell
# Clone the repository
git clone https://github.com/dilshan916/Post-bot.git
cd Post-bot

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browser dependencies (for UI screenshots)
playwright install chromium
```

---

### 3. Local Kokoro-82M Model Setup (Optional, Recommended)

To enable 100% offline, studio-grade speech synthesis, download the Kokoro model files into `assets/kokoro/`:

```powershell
.venv\Scripts\python -c "
import requests, pathlib
d = pathlib.Path('assets/kokoro'); d.mkdir(parents=True, exist_ok=True)
for url, name in [
    ('https://huggingface.co/thewh1teagle/Kokoro/resolve/main/kokoro-v0_19.onnx', 'kokoro-v0_19.onnx'),
    ('https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin', 'voices.bin')
]:
    print('Downloading', name)
    resp = requests.get(url, stream=True)
    with open(d / name, 'wb') as f: f.write(resp.content)
print('Kokoro setup complete!')
"
```
*(If model files are omitted, `Post-bot` automatically logs a warning and falls back to Edge-TTS).*

---

## ⚙️ Configuration (`config.yaml`)

Copy `config.example.yaml` to `config.yaml`:
```powershell
cp config.example.yaml config.yaml
```

Configure your API keys, voices, and Facebook Page credentials:

```yaml
# Gemini API Key(s) for analytics & story selection
llm:
  api_keys:
    - "YOUR_GEMINI_API_KEY_1"
    - "YOUR_GEMINI_API_KEY_2"

# Groq API Key(s) for script rewriting (Rotated automatically on rate limits)
groq:
  api_key: "gsk_PRIMARY_GROQ_KEY"
  api_keys:
    - "gsk_PRIMARY_GROQ_KEY"
    - "gsk_BACKUP_GROQ_KEY"

# TTS Engine Choice
tts:
  engine: "kokoro" # "kokoro" | "edge-tts"
  male_voice: "am_adam"
  female_voice: "af_bella"
  old_male_voice: "bm_george"
  old_female_voice: "af_nicole"

# Facebook Page Access Tokens & Page IDs for Auto-Scheduling
facebook:
  pages:
    - page_name: "Daily Stories"
      page_id: "YOUR_PAGE_ID_1"
      access_token: "YOUR_PAGE_ACCESS_TOKEN_1"
    - page_name: "Reddit Stories"
      page_id: "YOUR_PAGE_ID_2"
      access_token: "YOUR_PAGE_ACCESS_TOKEN_2"
```

---

## 🎮 Running the Application

Launch the interactive console application:
```powershell
python main.py
```

### Interactive Menu:
* **`1` — Monologue Mode**: Renders a single-narrator video from top Reddit stories.
* **`2` — Conversational Mode**: Generates a multi-speaker text drama video.
* **`3` — AskReddit Thread Mode**: Compiles a curated multi-comment thread video.
* **`4` — Fun Riddle Mode**: Generates interactive riddles with animated countdown timers.
* **`5` — Batch Hybrid Scheduler (Recommended)**: Scrapes, rewrites, requests CLI verification (`y`/`n`), renders videos in sequence, and schedules them directly to Facebook Pages at **09:30 AM** and **07:30 PM SLT**.

---

## 🧪 Testing

Execute the comprehensive unit and integration test suite:
```powershell
python -m pytest
```
*Includes 54 unit tests covering TTS engines, scrapers, script splitters, speaker resolution, and Facebook publishers.*

---

## 📄 License & Credits

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

### Developed & Maintained by
* **Dilshan Chandrarathne** ([@dilshan916](https://github.com/dilshan916))
