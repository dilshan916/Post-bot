# 🤖 Reddit Daily Bot

An automated vertical Reel/Short generator that compiles engaging Reddit stories into high-retention, subtitled vertical videos overlayed on gameplay footage.

Designed with a modern, modular **Hybrid LLM Architecture** (Gemini for selector analytics + Groq for script-writing), Edge-TTS voice generation, dynamic double-pass subtitling, and metadata-scrambling video composite filters to ensure organic reach and prevent algorithmic deduplication.

---

## ✨ Features

- **🧠 Hybrid LLM Strategy**:
  - **Analytics (Gemini 2.5 Flash)**: Batch-evaluates the top 15 qualified candidate stories simultaneously based on strict engagement engagement metrics (High Emotional Trigger, Debatability, and Strong 3-second Openers) to select a single winner with the highest virality potential.
  - **Creative (Groq - Llama 3.3 70B)**: Rewrites the raw Reddit self-text into a high-retention first-person narration script with natural spoken pacing, removing Meta-elements (EDIT, TL;DR, formatting apologies) and resolving abbreviations (e.g., *AITA* → *"Am I the jerk"*).
- **🎙️ Acoustic Cadence Processing**:
  - Injects realistic, breathing-room pauses dynamically based on punctuation cadence (comma, period, long dash pauses).
  - Implements a **vocal decay hack** using tail-phrase rooms and noise floors to avoid sudden audio clip cuts.
  - **Dynamic Voice Selector**: Automatically detects the narrator's gender from the post context and switches voices dynamically (*MALE* → `en-US-ChristopherNeural`, *FEMALE* → `en-US-JennyNeural`).
- **💬 Double-Pass Active Subtitles**:
  - Renders stylized word-by-word active-word highlighted subtitles centered exactly on spoken timing.
  - Features glassmorphic/bordered typography options, background opacity control, and custom layouts.
- **📸 Screenshot Hook Overlay**:
  - Programmatically generates and overlays a Reddit post title card as a visual hook for the first 3.5 seconds, complete with a smooth fade-out.
- **🎮 Video Hash Scrambling & Compositing**:
  - Downloads gameplay background footages dynamically via `yt-dlp`.
  - Runs background videos through a custom **FFmpeg Hash Destruction Pipeline** (scales, minor shifts, temporal noise injection, 1% audio speedups) to bypass social media reuse detection algorithms.
  - Blends master audio with a background ambient music track mixed down to a balanced 7% (-23.1 dB) volume with a beautiful 1.5-second fade-out.
- **⚡ Smart Part Splitter**:
  - Automatically slices long-form posts crossing the length limits into multiple Reels, rendering them as part series (e.g., *"Part 1 of 3"*) with custom watermark indicators.

---

## 🛠️ Setup Instructions

### Prerequisites
- Python `3.10` or higher
- FFmpeg installed and added to your system's PATH variables

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_GITHUB_USERNAME/Post-bot.git
   cd Post-bot
   ```

2. Initialize virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up configuration:
   - Copy `config.example.yaml` to `config.yaml`:
     ```bash
     cp config.example.yaml config.yaml
     ```
   - Edit `config.yaml` to fill in your API keys (Reddit API, Gemini, and Groq).

---

## ⚙️ Configuration

The application is configured using a master `config.yaml` file. Key parameters include:

| Section | Parameter | Description |
| :--- | :--- | :--- |
| **reddit** | `subreddits` | List of target subreddits to scrape stories from. |
| | `min_upvotes` | Minimum score to consider a post. |
| **llm** | `api_keys` | Rotated pool of Gemini API keys for story selection. |
| **groq** | `api_key` | Groq API key used for script rewriting. |
| | `model` | Model name (default: `llama-3.3-70b-versatile`). |
| **video** | `bitrate` | Explicit video bitrate standard (e.g. `5000k` to prevent upload compression blur). |
| | `audio_bitrate` | High quality audio standard (`192k`). |
| | `preset` | H.264 compression speed (`medium` or `fast`). |

---

## 🚀 Usage

Run the full end-to-end pipeline:
```bash
python main.py
```

### Advanced Options

- **Batch Mode** (Process multiple stories):
  ```bash
  python main.py --batch 3
  ```

- **Dry-Run Validation** (Validates configuration, folders, dependencies without executing APIs or rendering):
  ```bash
  python main.py --dry-run
  ```

- **Run Single Components**:
  ```bash
  python main.py --component scraper
  python main.py --component tts --test-phrase "Testing TTS engine"
  python main.py --component whisper --test-audio path/to/audio.wav
  ```

---

## 🧪 Testing

Run the complete unit test suite containing 24 tests covering scrapers, splitters, screenshot card builders, and Groq mock validations:
```bash
python -m pytest
```

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
