# 🤖 Reddit Daily Bot (Post-bot) — Getting Started Guide & Documentation

An automated vertical video generator & Facebook Reel publisher that compiles Reddit stories, threads, and riddles into high-retention subtitled vertical Reels (9:16) overlayed on gameplay footage.

Features a **Hybrid LLM Architecture** (Gemini 2.5 Flash for analytics & story ranking + Groq Llama 3.3 70B for script rewriting with key rotation), **Dual TTS Engines** (Local Kokoro-82M + Edge-TTS fallback), dynamic double-pass active word subtitling, and direct **Facebook Page Reel Scheduling** in Sri Lankan Time (SLT UTC+05:30).

---

## ✨ Key Features & Pipeline Modes

### 🎬 Pipeline Modes
1. **Monologue Mode**: High-stakes narrative stories with a single narrator, screenshot hook title card, and active word subtitles.
2. **Conversational Mode**: Multi-character text-message drama scripts with distinct Kokoro voice roles (`MALE`, `FEMALE`, `OLD_FEMALE`, `OLD_MALE`, `CHILD_MALE`, `CHILD_FEMALE`).
3. **AskReddit Thread Mode**: Curated top comments compiled into thread-style video series with author headers and upvote counts.
4. **Fun Riddle Mode**: Interactive riddles generated directly via Gemini Flash with suspense countdown timers.
5. **Batch Hybrid Scheduler (Mode 5)**: End-to-end automation that generates and schedules up to **42 Reels over a 7-day period** (2 posts/day at **09:30 AM** and **07:30 PM SLT**) with upfront CLI script approval (`y`/`n`), zero comment clutter, and hardware cool-down protection.

### 🎙️ Dual TTS Engine Architecture
- **Kokoro-82M (Local, High Quality)**: Ultra-realistic offline text-to-speech utilizing `CPUExecutionProvider` for 100% stability across all GPUs. Supports distinct voice assignments (`am_adam`, `af_bella`, `bm_george`, `af_nicole`, `am_puck`, `af_sky`).
- **Edge-TTS (Cloud Fallback)**: Zero-setup cloud fallback engine that automatically engages if local Kokoro assets are absent.

### 🧠 Hybrid LLM Engine
- **Gemini 2.5 Flash**: Batch analytics engine that evaluates candidate stories and ranks the top viral concepts based on emotional triggers, debatability, and retention potential.
- **Groq (Llama 3.3 70B)**: Creative script rewriter enforcing strict retention constraints (high-stakes openers, curiosity-gap teasers, delayed escalation reveals) with automatic multi-key rotation on rate limits.

---

## 🛠️ Step-by-Step Getting Started Guide

### 1. Prerequisites
- **Python**: Version `3.10` or higher installed.
- **FFmpeg**: Installed and added to your system's PATH.

### 2. Environment & Dependency Setup

```powershell
# 1. Clone the repository
git clone https://github.com/dilshan916/Post-bot.git
cd Post-bot

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# 3. Install required Python packages
pip install -r requirements.txt

# 4. Install Playwright browser binaries (required for title card screenshot rendering)
playwright install chromium
```

---

### 3. Kokoro-82M Local TTS Setup (Recommended)

To use high-quality local offline TTS narration, download the two model assets into `assets/kokoro/`:

1. **`kokoro-v0_19.onnx`** (~310 MB): [Download Link](https://huggingface.co/thewh1teagle/Kokoro/resolve/main/kokoro-v0_19.onnx)
2. **`voices.bin`** (~5.5 MB): [Download Link](https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin)

*Or automatically download both files via Python:*
```powershell
.venv\Scripts\python -c "
import requests, pathlib
d = pathlib.Path('assets/kokoro'); d.mkdir(parents=True, exist_ok=True)
for url, name in [('https://huggingface.co/thewh1teagle/Kokoro/resolve/main/kokoro-v0_19.onnx', 'kokoro-v0_19.onnx'), ('https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin', 'voices.bin')]:
    print('Downloading', name)
    resp = requests.get(url, stream=True)
    with open(d / name, 'wb') as f: f.write(resp.content)
print('Kokoro setup complete!')
"
```
*(If model files are omitted, `Post-bot` automatically logs a warning and uses Edge-TTS smoothly).*

---

### 4. Configuration Setup (`config.yaml`)

Copy `config.example.yaml` to `config.yaml`:
```powershell
cp config.example.yaml config.yaml
```

Open `config.yaml` and configure your API keys and target pages:

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
    - page_name: "Pick your poison"
      page_id: "YOUR_PAGE_ID_3"
      access_token: "YOUR_PAGE_ACCESS_TOKEN_3"
```

---

### 5. Running the Bot

Launch the interactive console application:
```powershell
python main.py
```

#### Menu Options:
- **`1` — Monologue Mode**: Renders a single narration video from Reddit stories.
- **`2` — Conversational Mode**: Renders a multi-speaker text drama video.
- **`3` — AskReddit Thread Mode**: Renders a top-comments thread video.
- **`4` — Fun Riddle Mode**: Renders a riddle video with countdown timer.
- **`5` — Batch Hybrid Scheduler (Recommended for Full Automation)**:
  1. Select target page: `1. Daily Stories`, `2. Reddit Stories`, `3. Pick your poison`, or `4. All Pages`.
  2. The bot scrapes top posts and uses Gemini to rank up to 14 viral concepts.
  3. **Interactive Script Approval**: Review each rewritten Groq script in the CLI and enter `y` to approve or `n` to reject.
  4. **Automated Rendering & Scheduling**: Once approved, videos are rendered one by one and scheduled to your Facebook page at **09:30 AM** and **07:30 PM SLT**, with a 30-second hardware cooldown between videos.

---

## 🧪 Testing

Run the full automated unit test suite (54 unit tests covering TTS engines, scrapers, splitters, speaker resolution, and Facebook publisher):
```powershell
python -m pytest
```

---

## 📄 License
This project is licensed under the MIT License.
