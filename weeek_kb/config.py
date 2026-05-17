import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# Paths (ROOT defined above for dotenv)
DATA_DIR = ROOT / "data"
CHROMA_DIR = ROOT / "chroma_db"
CONFIG_DIR = ROOT / "config"
ACTIVE_BOARD_IDS_FILE = CONFIG_DIR / "active-board-ids.json"

# Weeek workspace in task URLs: https://app.weeek.net/ws/<id>/task/<taskId>
WEEEK_WORKSPACE_ID = os.getenv("WEEEK_WORKSPACE_ID", "423726")
WEEEK_TASK_URL_TEMPLATE = os.getenv(
    "WEEEK_TASK_URL_TEMPLATE",
    "https://app.weeek.net/ws/{workspace_id}/task/{task_id}",
)
WEEEK_API_TOKEN = os.getenv("WEEEK_API_TOKEN") or os.getenv("WEEEK_TOKEN")
WEEEK_API_BASE_URL = os.getenv("WEEEK_API_BASE_URL", "https://api.weeek.net/public/v1").rstrip("/")

# Веб-вход для парсинга комментариев (Playwright)
WEEEK_EMAIL = (
    os.getenv("WEEEK_EMAIL")
    or os.getenv("WEEEK_LOGIN")
    or ""
).strip()
WEEEK_PASSWORD = os.getenv("WEEEK_PASSWORD", "").strip()
WEEEK_SESSION_FILE = Path(
    os.getenv("WEEEK_SESSION_FILE", str(DATA_DIR / "weeek-playwright-session.json"))
)
TASK_DEFAULT_COLUMN_NAME = os.getenv("TASK_DEFAULT_COLUMN_NAME", "На неделю")
TASK_DEFAULT_REQUESTER = os.getenv("TASK_DEFAULT_REQUESTER", "Мария")


def task_url(task_id: str | int) -> str:
    return WEEEK_TASK_URL_TEMPLATE.format(
        workspace_id=WEEEK_WORKSPACE_ID,
        task_id=task_id,
    )

# API keys (OPEN_APY_KEY / OPENAI_APY_KEY — частые опечатки в .env)
OPENAI_API_KEY = (
    os.getenv("OPENAI_API_KEY")
    or os.getenv("OPEN_APY_KEY")
    or os.getenv("OPENAI_APY_KEY")
)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_TOKEN")

OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
# Распознавание голосовых в боте: https://platform.openai.com/docs/guides/speech-to-text
OPENAI_TRANSCRIBE_MODEL = os.getenv("OPENAI_TRANSCRIBE_MODEL", "whisper-1")

VECTOR_SEARCH_PER_QUERY = int(os.getenv("VECTOR_SEARCH_PER_QUERY", "15"))
TOP_K_AFTER_MERGE = int(os.getenv("TOP_K_AFTER_MERGE", "10"))
TOP_PRESENT = 3
TOP_OVERFLOW = 7

INTENT_CONFIDENCE_THRESHOLD = float(os.getenv("INTENT_CONFIDENCE_THRESHOLD", "0.65"))
