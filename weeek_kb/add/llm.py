from __future__ import annotations

import html
import json
import re
from datetime import date
from typing import Any

from weeek_kb._openai import get_openai_client
from weeek_kb.config import OPENAI_CHAT_MODEL, TASK_DEFAULT_COLUMN_NAME
from weeek_kb.projects import Project


def _esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def _strip_json_fence(raw: str) -> str:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def extract_task_draft_fields(
    text: str,
    history: list[str],
    projects: list[Project],
    members: list[dict[str, Any]],
    column_names: list[str],
) -> dict[str, Any]:
    """Извлечь поля черновика задачи из последнего сообщения и истории диалога."""
    proj_lines = [
        f"- key={p.collection_name!r} label={p.label!r} projectId={p.project_id} boardId={p.board_id}"
        for p in projects
    ]
    mem_lines = [
        f"- id={m.get('id')!r} name={m.get('name') or m.get('fullName') or m.get('email')!r}"
        for m in members[:40]
    ]
    today = date.today()
    weekday_ru = ("понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье")[
        today.weekday()
    ]
    system = (
        "Ты извлекаешь поля для постановки задачи в Weeek из сообщений пользователя Telegram.\n"
        "Компания занимается техобслуживанием интернет-магазинов на 1С-Битрикс.\n"
        f"Сегодня: {today.strftime('%d.%m.%Y')} ({weekday_ru}). Все дедлайны считай от этой даты.\n"
        "Постановщик: если в тексте явно назван — укажи requester_name; если не упомянут — null и requester_unclear false "
        "(по умолчанию подставится Мария). requester_unclear true только при неоднозначном упоминании постановщика.\n"
        "due_date: только конкретная дата DD.MM.YYYY. Для «ближайшее воскресенье», «к пятнице» и т.п. "
        "посчитай календарную дату от сегодня; не выдумывай прошлые годы. Если дедлайн не указан — null.\n"
        f"Если колонка не указана явно — оставь column_name null (подставится «{TASK_DEFAULT_COLUMN_NAME}»).\n"
        "ВАЖНО — сохранность контекста (особенно для расшифровок голосовых):\n"
        "- task_text должен содержать ВСЕ конкретные данные из latest_message и conversation: "
        "номера телефонов, email, URL и домены, ФИО, названия компаний/контрагентов, артикулы и SKU, "
        "номера заказов/счетов/договоров, суммы, валюты, адреса, даты, время, версии, идентификаторы.\n"
        "- Цифры, телефоны, email, URL переноси ДОСЛОВНО, не пересказывай и не округляй. "
        "Телефоны нормализуй только в формат без пробелов и тире, цифры не теряй и не добавляй.\n"
        "- Если в голосовом распознавании цифры прописью («восемь девятьсот…») — собери их в число, "
        "но не пропускай ни одной произнесённой цифры.\n"
        "- «Своими словами» относится к формулировке сути, а НЕ к фактам: факты сохраняются полностью.\n"
        "- Перед ответом ПЕРЕПРОВЕРЬ: пройдись по latest_message и истории, выпиши мысленно все факты "
        "(контакты, числа, ссылки, имена, адреса) и убедись, что каждый из них присутствует в task_text. "
        "Если чего-то не хватает — допиши, прежде чем возвращать JSON.\n"
        "Ответ строго JSON:\n"
        "{\n"
        '  "task_text": "суть задачи + все конкретные факты из сообщения (контакты, числа, ссылки сохраняй дословно) или null",\n'
        '  "due_date": "DD.MM.YYYY или YYYY-MM-DD или null",\n'
        '  "project_key": "всегда null (доску выбирают кнопкой, если в тексте нет домена сайта); не угадывай по бренду товара",\n'
        '  "assignee_name": "имя исполнителя как в списке или null",\n'
        '  "requester_name": "имя постановщика (кто поставил задачу) или null",\n'
        '  "requester_unclear": true только если постановщик упомянут, но имя нельзя однозначно определить; иначе false,\n'
        '  "column_name": "название колонки или null"\n'
        "}"
    )
    user_payload = {
        "latest_message": text.strip(),
        "conversation": history[-8:],
        "projects": proj_lines,
        "members": mem_lines,
        "known_columns": column_names[:30],
    }
    resp = get_openai_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        temperature=0.2,
    )
    raw = resp.choices[0].message.content or ""
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def formulate_task_for_weeek(draft: dict[str, Any]) -> dict[str, str]:
    """Оформить title и description (HTML) как проджект-менеджер."""
    system = (
        "Ты проджект-менеджер команды по техническому и коммерческому сопровождению интернет-магазинов на 1С-Битрикс. "
        "Оформи задачу для Weeek: заголовок и описание в HTML (<p>, <ul><li>, <strong>).\n\n"
        "Масштаб оформления должен соответствовать объёму task_text:\n"
        "- Короткая/лаконичная постановка → ёмкий заголовок, описание без лишних разделов и длинных чек-листов.\n"
        "- Подробная постановка с контекстом, шагами, ограничениями → развёрнутое структурированное описание как у PM, "
        "только из фактов во входе.\n"
        "Не раздувай короткие задачи шаблонными блоками («Цель», «Контекст», «Риски»), выдуманными деталями, URL и доступами.\n\n"
        "ВАЖНО — сохранность контекста:\n"
        "- В description должны попасть ВСЕ конкретные факты из task_text: номера телефонов, email, URL и домены, "
        "ФИО, названия компаний и контрагентов, артикулы/SKU, номера заказов/счетов/договоров, суммы, валюты, "
        "адреса, даты, время, версии, идентификаторы.\n"
        "- Контакты, числа, ссылки и идентификаторы переноси ДОСЛОВНО (можно вынести в блок «Контакты» или "
        "«Данные» через <ul><li>), не пересказывай и не сокращай цифры.\n"
        "- Краткость оформления НЕ ОЗНАЧАЕТ выбрасывать факты: лучше короткий <p> с фактами, чем потерянный телефон.\n"
        "- Ничего не выдумывай: добавляй только то, что есть в task_text.\n"
        "- ПЕРЕД ОТВЕТОМ перепроверь: пройдись по task_text, выпиши все факты (телефоны, email, ссылки, имена, "
        "адреса, числа, артикулы, даты) и убедись, что каждый из них присутствует в description. "
        "Если чего-то не хватает — допиши, прежде чем возвращать JSON.\n\n"
        "Критерий выполнения всегда сформулируй сам из task_text (пользователь его отдельно не присылает). "
        "Заголовок блока ровно: <strong>Критерий выполнения</strong> (не «(сформулировано ботом)» и не другой текст), "
        "затем <ul> с пунктами; для простых задач — 1–2 пункта.\n\n"
        "Не включай в title и description постановщика, строку «Постановщик: …», дисклеймер бота и префикс [bot].\n"
        "Ответ строго JSON:\n"
        '{"title": "...", "description": "<p>...</p>"}'
    )
    resp = get_openai_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(draft, ensure_ascii=False)},
        ],
        temperature=0.35,
    )
    raw = _strip_json_fence(resp.choices[0].message.content or "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        title = str(draft.get("task_text") or "Новая задача")[:120]
        body = str(draft.get("task_text") or "")
        desc = (
            f"<p>{_esc(body)}</p>"
            "<p><strong>Критерий выполнения</strong></p>"
            "<ul><li>Задача выполнена согласно описанию выше.</li></ul>"
        )
        return {"title": title, "description": desc}
    return {
        "title": str(data.get("title") or "Новая задача").strip()[:200],
        "description": str(data.get("description") or "").strip(),
    }
