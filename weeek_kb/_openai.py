from __future__ import annotations

from openai import OpenAI

from weeek_kb.config import OPENAI_API_KEY


def get_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Set OPENAI_API_KEY in .env (also accepted: OPEN_APY_KEY, OPENAI_APY_KEY)"
        )
    return OpenAI(api_key=OPENAI_API_KEY)
