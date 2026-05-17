"""Транскрибация аудио через OpenAI Speech-to-Text API (Whisper)."""

from __future__ import annotations

import io

from openai import OpenAI

from weeek_kb.config import OPENAI_API_KEY, OPENAI_TRANSCRIBE_MODEL


def transcribe_audio_bytes(data: bytes, filename: str = "audio.ogg") -> str:
    """
    Распознаёт речь в аудиофайле. Telegram voice — обычно OGG Opus, имя файла важно для формата.
    Возвращает текст без ведущих/хвостовых пробелов.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "Set OPENAI_API_KEY in .env (also accepted: OPEN_APY_KEY, OPENAI_APY_KEY)"
        )
    if not data:
        return ""
    client = OpenAI(api_key=OPENAI_API_KEY)
    buf = io.BytesIO(data)
    buf.name = filename
    tr = client.audio.transcriptions.create(
        model=OPENAI_TRANSCRIBE_MODEL,
        file=buf,
    )
    return (tr.text or "").strip()
