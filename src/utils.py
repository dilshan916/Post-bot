"""
RedditDaily-Bot — Shared Utilities
===================================
Logging, config loading, retry logic, file helpers, and validation
used across all pipeline components.
"""

import json
import logging
import os
import sys
import time
import functools
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml
from colorama import Fore, Style, init as colorama_init

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"

# ---------------------------------------------------------------------------
# Colorama init (Windows ANSI support)
# ---------------------------------------------------------------------------
colorama_init(autoreset=True)


# ---------------------------------------------------------------------------
# Structured Logger
# ---------------------------------------------------------------------------
class BotLogger:
    """Custom logger with coloured console + file output."""

    _LOG_COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.MAGENTA + Style.BRIGHT,
    }

    def __init__(
        self,
        name: str = "RedditDailyBot",
        log_dir: Optional[str] = None,
        level: str = "INFO",
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self.logger.handlers.clear()

        # Console handler with colours
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(self._ColorFormatter())
        self.logger.addHandler(console)

        # File handler (if log_dir provided)
        if log_dir:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fh = logging.FileHandler(
                log_path / f"run_{timestamp}.log", encoding="utf-8"
            )
            fh.setFormatter(
                logging.Formatter(
                    "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
                )
            )
            self.logger.addHandler(fh)

    class _ColorFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            color = BotLogger._LOG_COLORS.get(record.levelname, "")
            ts = datetime.now().strftime("%H:%M:%S")
            msg = f"{Fore.WHITE}{ts} {color}[{record.levelname:<8}]{Style.RESET_ALL} {record.getMessage()}"
            return msg

    # Convenience pass-through methods
    def debug(self, msg: str, *a: Any, **kw: Any) -> None:
        self.logger.debug(msg, *a, **kw)

    def info(self, msg: str, *a: Any, **kw: Any) -> None:
        self.logger.info(msg, *a, **kw)

    def warning(self, msg: str, *a: Any, **kw: Any) -> None:
        self.logger.warning(msg, *a, **kw)

    def error(self, msg: str, *a: Any, **kw: Any) -> None:
        self.logger.error(msg, *a, **kw)

    def critical(self, msg: str, *a: Any, **kw: Any) -> None:
        self.logger.critical(msg, *a, **kw)


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------
def load_config(path: Optional[str] = None) -> Dict[str, Any]:
    """Load and validate the YAML configuration file.

    Parameters
    ----------
    path : str | None
        Path to config.yaml.  Falls back to ``PROJECT_ROOT/config.yaml``.

    Returns
    -------
    dict
        Parsed configuration dictionary.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If required top-level keys are missing.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.exists():
        fallback_path = PROJECT_ROOT / "config.example.yaml"
        if fallback_path.exists():
            config_path = fallback_path
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as fh:
        cfg: Dict[str, Any] = yaml.safe_load(fh)

    required_keys = ["reddit", "tts", "video", "pipeline"]
    missing = [k for k in required_keys if k not in cfg]
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")

    # Environment variable overrides for CI/CD and GitHub Actions
    if "GEMINI_API_KEY" in os.environ:
        gem_key = os.environ["GEMINI_API_KEY"].strip()
        if gem_key:
            cfg.setdefault("llm", {})["api_key"] = gem_key
            cfg.setdefault("llm", {})["api_keys"] = [gem_key]
    if "GROQ_API_KEY" in os.environ:
        groq_key = os.environ["GROQ_API_KEY"].strip()
        if groq_key:
            cfg.setdefault("groq", {})["api_key"] = groq_key
            cfg.setdefault("groq", {})["api_keys"] = [groq_key]
    # Mode-specific Facebook secret overrides
    fb_cfg = cfg.setdefault("facebook", {})
    if "FACEBOOK_CONVERSATIONAL_PAGE_ACCESS_TOKEN" in os.environ:
        c_token = os.environ["FACEBOOK_CONVERSATIONAL_PAGE_ACCESS_TOKEN"].strip()
        if c_token:
            fb_cfg["enabled"] = True
            fb_cfg["conversational_page_access_token"] = c_token
            if "FACEBOOK_CONVERSATIONAL_PAGE_ID" in os.environ and os.environ["FACEBOOK_CONVERSATIONAL_PAGE_ID"].strip():
                fb_cfg["conversational_page_id"] = os.environ["FACEBOOK_CONVERSATIONAL_PAGE_ID"].strip()

    if "FACEBOOK_MONOLOGUE_PAGE_ACCESS_TOKEN" in os.environ:
        m_token = os.environ["FACEBOOK_MONOLOGUE_PAGE_ACCESS_TOKEN"].strip()
        if m_token:
            fb_cfg["enabled"] = True
            fb_cfg["monologue_page_access_token"] = m_token
            if "FACEBOOK_MONOLOGUE_PAGE_ID" in os.environ and os.environ["FACEBOOK_MONOLOGUE_PAGE_ID"].strip():
                fb_cfg["monologue_page_id"] = os.environ["FACEBOOK_MONOLOGUE_PAGE_ID"].strip()

    if "FACEBOOK_PAGE_ACCESS_TOKEN" in os.environ:
        fb_token = os.environ["FACEBOOK_PAGE_ACCESS_TOKEN"].strip()
        if fb_token:
            fb_cfg["enabled"] = True
            fb_cfg["page_access_token"] = fb_token
            if "monologue_page_access_token" not in fb_cfg:
                fb_cfg["monologue_page_access_token"] = fb_token
            if "conversational_page_access_token" not in fb_cfg:
                fb_cfg["conversational_page_access_token"] = fb_token
            if "FACEBOOK_PAGE_ID" in os.environ and os.environ["FACEBOOK_PAGE_ID"].strip():
                page_id_val = os.environ["FACEBOOK_PAGE_ID"].strip()
                fb_cfg["page_id"] = page_id_val
                if "monologue_page_id" not in fb_cfg:
                    fb_cfg["monologue_page_id"] = page_id_val
                if "conversational_page_id" not in fb_cfg:
                    fb_cfg["conversational_page_id"] = page_id_val

    return cfg


def resolve_path(cfg_path: str, create: bool = False) -> Path:
    """Resolve a potentially relative path against the project root.

    Parameters
    ----------
    cfg_path : str
        Path string from config (may be relative).
    create : bool
        If True, create the directory (or parent for files).
    """
    p = Path(cfg_path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    if create:
        target = p if p.suffix == "" else p.parent
        target.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Retry Decorator
# ---------------------------------------------------------------------------
def retry(
    max_attempts: int = 3,
    delay_sec: float = 2.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    logger: Optional[BotLogger] = None,
) -> Callable:
    """Exponential-backoff retry decorator.

    Parameters
    ----------
    max_attempts : int
        Total number of tries (including the first).
    delay_sec : float
        Initial delay between retries in seconds.
    backoff : float
        Multiplier applied to delay after each failure.
    exceptions : tuple
        Exception types that trigger a retry.
    logger : BotLogger | None
        Logger for retry warnings.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            _delay = delay_sec
            last_exc: Optional[Exception] = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts:
                        break
                    msg = (
                        f"Retry {attempt}/{max_attempts} for "
                        f"{func.__name__}: {exc!r} — waiting {_delay:.1f}s"
                    )
                    if logger:
                        logger.warning(msg)
                    else:
                        print(f"[WARN] {msg}")
                    time.sleep(_delay)
                    _delay *= backoff
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# File & Data Helpers
# ---------------------------------------------------------------------------
def ensure_dirs(*dirs: str) -> None:
    """Create directories if they don't exist."""
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


def save_json(data: Any, path: str, indent: int = 2) -> None:
    """Write data as formatted JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)


def load_json(path: str, default: Any = None) -> Any:
    """Read a JSON file; return *default* if missing."""
    p = Path(path)
    if not p.exists():
        return default if default is not None else {}
    with open(p, "r", encoding="utf-8") as fh:
        return json.load(fh)


def file_hash(path: str, algo: str = "sha256") -> str:
    """Compute hex digest of a file for dedup checks."""
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Duration & Text Helpers
# ---------------------------------------------------------------------------
def estimate_duration_sec(text: str, wpm: int = 155) -> float:
    """Estimate spoken duration from word count and WPM. Supports JSON dialog arrays."""
    import json
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            total_words = 0
            for block in parsed:
                if isinstance(block, dict) and "text" in block:
                    total_words += len(str(block["text"]).split())
            return (total_words / wpm) * 60.0
    except Exception:
        pass
    words = len(text.split())
    return (words / wpm) * 60.0


def word_count(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def format_duration(seconds: float) -> str:
    """Format seconds into MM:SS string."""
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def sanitize_filename(name: str, max_len: int = 80) -> str:
    """Create a filesystem-safe filename from arbitrary text."""
    safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in name)
    safe = "_".join(safe.split())  # collapse whitespace
    return safe[:max_len]


def timestamp_str() -> str:
    """ISO-ish timestamp for filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------
def validate_api_key(key: str, name: str = "API Key") -> bool:
    """Check that an API key is a non-empty, non-placeholder string."""
    if not key or not isinstance(key, str):
        return False
    placeholders = {"YOUR_", "PLACEHOLDER", "xxx", "CHANGE_ME", ""}
    return not any(key.upper().startswith(p) for p in placeholders if p)


def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    import shutil
    return shutil.which("ffmpeg") is not None


def check_yt_dlp() -> bool:
    """Return True if yt-dlp is available on PATH."""
    import shutil
    return shutil.which("yt-dlp") is not None


def resolve_speaker_category(speaker_str: str) -> str:
    """Normalize custom character speaker names into 6 standard categories:
    MALE, FEMALE, OLD_MALE, OLD_FEMALE, CHILD_MALE, CHILD_FEMALE.

    This ensures that various custom names (e.g. 'Mom', 'Dave', 'Op', 'Brother')
    map cleanly to the correct voices, backgrounds, and sticker visuals.
    """
    import re
    s = speaker_str.strip().upper()
    
    # Direct match check for standard names first
    if s in ("MALE", "FEMALE", "OLD_MALE", "OLD_FEMALE", "CHILD_MALE", "CHILD_FEMALE", "CHIBI_MALE", "CHIBI_FEMALE"):
        if s == "CHIBI_MALE":
            return "CHILD_MALE"
        if s == "CHIBI_FEMALE":
            return "CHILD_FEMALE"
        return s

    if s in ("INTRO", "OUTRO", "NARRATOR"):
        return s
        
    # Split by non-alphanumeric characters to get individual tokens
    tokens = set(re.findall(r"[A-Z0-9]+", s))
    
    # Check for child indicators first (e.g., child, kid, boy, girl, son, daughter, baby, toddler, chibi)
    child_keywords = {"CHILD", "CHIBI", "KID", "BOY", "GIRL", "SON", "DAUGHTER", "BABY", "TODDLER", "KIDS", "BOYS", "GIRLS"}
    is_child = not tokens.isdisjoint(child_keywords)
    
    # Check for elder/parent indicators (e.g., mom, mother, dad, father, stepdad, stepmom, mil, fil, in-law, uncle, aunt, old, grand)
    old_keywords = {"OLD", "GRAND", "ELDER", "MOM", "MOTHER", "DAD", "FATHER", "STEPDAD", "STEPMOM", "MIL", "FIL", "IN-LAW", "INLAW", "UNCLE", "AUNT", "PARENT", "PARENTS", "GRANDPA", "GRANDMA", "GRANDFATHER", "GRANDMOTHER"}
    is_old = not tokens.isdisjoint(old_keywords)

    # Determine gender based on common male vs female indicator terms
    male_keywords = {"MALE", "BOY", "SON", "DAD", "FATHER", "STEPDAD", "HUSBAND", "BOYFRIEND", "BROTHER", "UNCLE", "NEPHEW", "GRANDPA", "GRANDFATHER", "GROOM", "HE", "HIM", "GUY", "MAN", "FIL", "BRO"}
    is_male = not tokens.isdisjoint(male_keywords)

    # If it has user numbers (like USER_1, USER_2), we can alternate based on the index/hash
    digits = re.findall(r"\d+", s)
    if digits:
        idx = int(digits[0])
        is_male = (idx % 2 != 0)

    if is_child:
        return "CHILD_MALE" if is_male else "CHILD_FEMALE"
    elif is_old:
        return "OLD_MALE" if is_male else "OLD_FEMALE"
    else:
        return "MALE" if is_male else "FEMALE"
