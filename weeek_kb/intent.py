from __future__ import annotations

import json
import re
from typing import Literal

from weeek_kb._openai import get_openai_client
from weeek_kb.config import OPENAI_CHAT_MODEL

MessageIntent = Literal["question", "task"]


def classify_message_intent(text: str) -> tuple[MessageIntent | None, float]:
    """
    Classify user message as question (search KB) or task (create new task).
    Returns (intent, confidence). intent is None if parsing failed or ambiguous.
    """
    system = (
        "Классифицируй сообщение пользователя в Telegram-боте Weeek.\n"
        "question — пользователь ищет информацию в уже существующих задачах: статус, что сделано, "
        "где описано, «как настроить…», «есть ли задача про…», «какие задачи по…».\n"
        "task — пользователь хочет создать или поставить новую задачу: поручение, "
        "«сделай задачу», «добавь в Weeek», «нужно сделать X к дате», «создай задачу» — "
        "без запроса к базе существующих задач.\n"
        'Ответь строго одним JSON: {"intent": "question" | "task", "confidence": 0..1}.'
    )
    resp = get_openai_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Сообщение:\n{text.strip()}"},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None, 0.0
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, 0.0
    intent_raw = data.get("intent")
    if intent_raw not in ("question", "task"):
        return None, 0.0
    conf = float(data.get("confidence") or 0)
    return intent_raw, conf
