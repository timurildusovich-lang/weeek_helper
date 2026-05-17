from __future__ import annotations

import asyncio
import html
import logging
import re
from datetime import date, datetime, timedelta
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.ext import ContextTypes

from weeek_kb.config import (
    TASK_DEFAULT_COLUMN_NAME,
    TASK_DEFAULT_REQUESTER,
    WEEEK_API_TOKEN,
    task_url,
)
from weeek_kb.add.llm import extract_task_draft_fields, formulate_task_for_weeek
from weeek_kb.add.weeek_api import (
    WeeekApiError,
    column_by_name,
    create_task,
    get_board_columns,
    get_members,
    member_display_name,
    member_id,
)
from weeek_kb.projects import Project, load_projects, match_project_from_text, project_by_collection

logger = logging.getLogger(__name__)

BOT_TASK_TITLE_PREFIX = "[bot]"

USER_TASK_FLOW = "task_flow_active"
USER_TASK_DRAFT = "task_draft"
CACHE_MEMBERS = "weeek_members_cache"
CACHE_COLUMNS = "weeek_columns_cache"


def is_task_flow(context: ContextTypes.DEFAULT_TYPE) -> bool:
    return bool(context.user_data.get(USER_TASK_FLOW))


def clear_task_session(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop(USER_TASK_FLOW, None)
    context.user_data.pop(USER_TASK_DRAFT, None)


def weeek_api_configured() -> bool:
    return bool((WEEEK_API_TOKEN or "").strip())


_MISSING_TOKEN_MSG = (
    "Для постановки задач в Weeek не задан WEEEK_API_TOKEN.\n\n"
    "1. Weeek → настройки workspace → раздел API → создайте токен.\n"
    "2. Добавьте в файл .env в корне проекта строку:\n"
    "WEEEK_API_TOKEN=ваш_токен\n"
    "3. Перезапустите бота (python -m weeek_kb.bot)."
)


async def _require_weeek_token(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """False — токена нет, пользователю уже отправлено пояснение."""
    if weeek_api_configured():
        return True
    clear_task_session(context)
    await _reply(update, context, _MISSING_TOKEN_MSG)
    return False


def _new_draft() -> dict[str, Any]:
    return {
        "messages": [],
        "task_text": None,
        "acceptance_criteria": None,
        "due_date": None,
        "collection_name": None,
        "project_id": None,
        "board_id": None,
        "board_column_id": None,
        "column_name": None,
        "assignee_user_id": None,
        "assignee_name": None,
        "requester_name": None,
        "requester_unclear": False,
        "project_confirmed": False,
        "awaiting": None,
        "needs_criteria": False,
    }


def _get_draft(context: ContextTypes.DEFAULT_TYPE) -> dict[str, Any]:
    d = context.user_data.get(USER_TASK_DRAFT)
    if not isinstance(d, dict):
        d = _new_draft()
        context.user_data[USER_TASK_DRAFT] = d
    return d


def _members_cached(context: ContextTypes.DEFAULT_TYPE) -> list[dict[str, Any]]:
    m = context.bot_data.get(CACHE_MEMBERS)
    if isinstance(m, list):
        return m
    members = get_members()
    context.bot_data[CACHE_MEMBERS] = members
    return members


def _columns_cached(context: ContextTypes.DEFAULT_TYPE, board_id: int) -> list[dict[str, Any]]:
    key = f"{CACHE_COLUMNS}_{board_id}"
    c = context.bot_data.get(key)
    if isinstance(c, list):
        return c
    cols = get_board_columns(board_id)
    context.bot_data[key] = cols
    return cols


def _nearest_friday(today: date | None = None) -> str:
    """Ближайшая пятница (включая сегодня, если сегодня пятница)."""
    today = today or date.today()
    days_ahead = (4 - today.weekday()) % 7
    target = today if days_ahead == 0 else today + timedelta(days=days_ahead)
    return target.strftime("%d.%m.%Y")


_WEEKDAY_RE: list[tuple[int, str]] = [
    (0, r"понедельник(?:у|а)?|\bпн\b"),
    (1, r"вторник(?:у|а)?|\bвт\b"),
    (2, r"сред[аыу](?:у)?|\bср\b"),
    (3, r"четверг(?:у|а)?|\bчт\b"),
    (4, r"пятниц[аыу](?:у)?|\bпт\b"),
    (5, r"суббот[аыу](?:у)?|\bсб\b"),
    (6, r"воскресень[еяю]|\bвс\b"),
]


def _next_weekday(weekday: int, today: date | None = None) -> date:
    """Ближайший указанный день недели (включая сегодня, если совпадает)."""
    today = today or date.today()
    days_ahead = (weekday - today.weekday()) % 7
    return today if days_ahead == 0 else today + timedelta(days=days_ahead)


def _parse_relative_due_date(text: str, today: date | None = None) -> date | None:
    """«К воскресенью», «ближайшее воскресенье», «завтра» и т.п."""
    if not text or not text.strip():
        return None
    today = today or date.today()
    t = text.lower().replace("ё", "е")

    if re.search(r"\bзавтра\b", t):
        return today + timedelta(days=1)
    if re.search(r"\bпослезавтра\b", t):
        return today + timedelta(days=2)
    m_days = re.search(r"через\s+(\d+)\s+дн", t)
    if m_days:
        return today + timedelta(days=int(m_days.group(1)))

    for wd, pattern in _WEEKDAY_RE:
        if re.search(rf"(?:ближайш\w*|на\s+ближайш\w*)\s+(?:{pattern})", t):
            return _next_weekday(wd, today)
        if re.search(rf"(?:\bк\b|\bдо\b|\bна\b|\bв\b)\s+(?:{pattern})", t):
            return _next_weekday(wd, today)
    return None


def _extract_explicit_date(text: str) -> str | None:
    m = re.search(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{2,4}))?\b", text)
    if not m:
        return None
    return _normalize_due_date(m.group(0).replace("/", "."))


def _is_plausible_due_date(value: str) -> bool:
    try:
        dt = datetime.strptime(value, "%d.%m.%Y").date()
    except ValueError:
        return False
    today = date.today()
    if dt < today - timedelta(days=1):
        return False
    if dt > today + timedelta(days=400):
        return False
    return True


def _text_mentions_weekday(text: str) -> bool:
    t = text.lower().replace("ё", "е")
    for _, pattern in _WEEKDAY_RE:
        if re.search(pattern, t):
            return True
    return False


def _resolve_due_date(text: str, llm_value: str | None = None) -> str | None:
    """Сначала явная дата в тексте, затем относительная формулировка, затем LLM."""
    explicit = _extract_explicit_date(text)
    if explicit and _is_plausible_due_date(explicit):
        return explicit

    relative = _parse_relative_due_date(text)
    if relative:
        return relative.strftime("%d.%m.%Y")

    if llm_value:
        normalized = _normalize_due_date(str(llm_value))
        if normalized and _is_plausible_due_date(normalized):
            if _text_mentions_weekday(text):
                try:
                    dt = datetime.strptime(normalized, "%d.%m.%Y").date()
                except ValueError:
                    return None
                expected = _parse_relative_due_date(text)
                if expected and dt != expected:
                    return expected.strftime("%d.%m.%Y")
            return normalized
    return None


def _apply_due_date_from_context(draft: dict[str, Any]) -> None:
    messages = draft.get("messages") or []
    combined = " ".join(messages)
    if not combined.strip():
        return
    resolved = _resolve_due_date(combined, draft.get("due_date"))
    draft["due_date"] = resolved


def _ensure_due_date(draft: dict[str, Any]) -> None:
    if not draft.get("due_date"):
        draft["due_date"] = _nearest_friday()


def _ensure_requester(draft: dict[str, Any]) -> None:
    if not (draft.get("requester_name") or "").strip():
        draft["requester_name"] = TASK_DEFAULT_REQUESTER


def _coerce_future_date(dt: date) -> date:
    """Год не раньше текущего; если дата в прошлом — сдвиг вперёд (тот же день/месяц)."""
    today = date.today()
    year = dt.year
    if year < today.year:
        year = today.year
    for attempt in range(3):
        try:
            candidate = dt.replace(year=year + attempt)
        except ValueError:
            candidate = date(year + attempt, min(dt.month, 12), min(dt.day, 28))
        if candidate >= today:
            return candidate
    return today


def _normalize_due_date(raw: str | None) -> str | None:
    if not raw or not str(raw).strip():
        return None
    s = str(raw).strip()
    today = date.today()
    m = re.match(r"^(\d{1,2})\.(\d{1,2})$", s)
    if m:
        try:
            dt = date(today.year, int(m.group(2)), int(m.group(1)))
            return _coerce_future_date(dt).strftime("%d.%m.%Y")
        except ValueError:
            pass
    m2 = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{2,4})$", s)
    if m2:
        d, mo, y = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if y < 100:
            y += 2000
        try:
            dt = _coerce_future_date(date(y, mo, d))
            return dt.strftime("%d.%m.%Y")
        except ValueError:
            pass
    for fmt in ("%d.%m.%Y", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            dt = datetime.strptime(s[:10], fmt).date()
            return _coerce_future_date(dt).strftime("%d.%m.%Y")
        except ValueError:
            continue
    return None


def _sanitize_formulated_description(html_desc: str) -> str:
    """Правки типичных ошибок LLM в HTML описания."""
    s = (html_desc or "").strip()
    if not s:
        return s
    s = re.sub(
        r"<strong>\s*\(сформулировано ботом\)\s*</strong>",
        "<strong>Критерий выполнения</strong>",
        s,
        flags=re.IGNORECASE,
    )
    s = re.sub(
        r"<p>\s*\(сформулировано ботом\)\s*</p>",
        "<p><em>(сформулировано ботом)</em></p>",
        s,
        flags=re.IGNORECASE,
    )
    if re.search(r"сформулировано ботом", s, re.I) and not re.search(
        r"Критерий выполнения", s, re.I
    ):
        s += "<p><strong>Критерий выполнения</strong></p><p><em>(сформулировано ботом)</em></p>"
    return s


_ASSIGNEE_HINT_RE = re.compile(
    r"(?:исполнител\w*|ответственн\w*|назнач\w+\s+на|делает|сделает|выполнит)"
    r"[\s:—\-–]*([а-яёa-z]{2,30})",
    re.I,
)


def _match_member(name: str, members: list[dict[str, Any]]) -> dict[str, Any] | None:
    target = (name or "").strip().casefold()
    if not target:
        return None
    for m in members:
        if member_display_name(m).casefold() == target:
            return m
    target_first = target.split()[0] if target else ""
    for m in members:
        dn = member_display_name(m).casefold()
        if target in dn or dn in target:
            return m
        dn_first = dn.split()[0] if dn else ""
        if target_first and len(target_first) >= 3 and target_first == dn_first:
            return m
    return None


def _assignee_names_from_text(text: str) -> list[str]:
    names: list[str] = []
    t = text or ""
    for m in _ASSIGNEE_HINT_RE.finditer(t):
        names.append(m.group(1).strip())
    for m in re.finditer(r"@([а-яёa-z]{2,30})", t, re.I):
        names.append(m.group(1).strip())
    return names


def _find_member_in_text(text: str, members: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Имя исполнителя в тексте (Тимур, @Тимур, «исполнитель: …»)."""
    if not text or not members:
        return None

    for raw in _assignee_names_from_text(text):
        m = _match_member(raw, members)
        if m:
            return m

    t = text.lower().replace("ё", "е")
    found: list[dict[str, Any]] = []
    for m in members:
        dn = member_display_name(m).lower().replace("ё", "е")
        if not dn:
            continue
        parts = [dn] + [p for p in dn.split() if len(p) >= 3]
        for cand in parts:
            if re.search(rf"(?<![а-яa-z]){re.escape(cand)}(?![а-яa-z])", t):
                found.append(m)
                break

    if not found:
        return None
    if len(found) == 1:
        return found[0]
    # Несколько совпадений — берём самое длинное имя (полное ФИО точнее «Иван»)
    return max(found, key=lambda m: len(member_display_name(m)))


def _set_draft_project(draft: dict[str, Any], p: Project) -> None:
    draft["collection_name"] = p.collection_name
    draft["project_label"] = p.label
    draft["project_id"] = p.project_id
    draft["board_id"] = p.board_id
    draft["board_column_id"] = None
    draft["project_confirmed"] = True


def _apply_project_from_context(draft: dict[str, Any], projects: list[Project]) -> None:
    if draft.get("project_confirmed"):
        return
    messages = draft.get("messages") or []
    combined = " ".join(messages)
    p = match_project_from_text(combined, projects)
    if p:
        _set_draft_project(draft, p)


def _apply_assignee_from_context(
    draft: dict[str, Any], members: list[dict[str, Any]]
) -> None:
    if draft.get("assignee_user_id"):
        return
    messages = draft.get("messages") or []
    combined = " ".join(messages)

    if draft.get("assignee_name"):
        m = _match_member(str(draft["assignee_name"]), members)
        if m:
            draft["assignee_user_id"] = member_id(m)
            draft["assignee_name"] = member_display_name(m)
            return

    m = _find_member_in_text(combined, members)
    if m:
        draft["assignee_user_id"] = member_id(m)
        draft["assignee_name"] = member_display_name(m)


def _merge_extracted(draft: dict[str, Any], data: dict[str, Any]) -> None:
    if data.get("task_text"):
        draft["task_text"] = str(data["task_text"]).strip()
    if data.get("due_date"):
        draft["due_date"] = str(data["due_date"]).strip() or None
    if data.get("column_name"):
        draft["column_name"] = str(data["column_name"]).strip()
    an = data.get("assignee_name")
    if an:
        draft["assignee_name"] = str(an).strip()
    rn = data.get("requester_name")
    if rn:
        draft["requester_name"] = str(rn).strip()
        draft["requester_unclear"] = False
    if "requester_unclear" in data:
        draft["requester_unclear"] = bool(data["requester_unclear"])


async def _reply(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
    keyboard: InlineKeyboardMarkup | None = None,
    *,
    parse_mode: str | None = None,
) -> None:
    kwargs: dict[str, Any] = {"text": text, "reply_markup": keyboard}
    if parse_mode is not None:
        kwargs["parse_mode"] = parse_mode
    if update.callback_query and update.callback_query.message:
        await update.callback_query.message.reply_text(**kwargs)
    elif update.message:
        await update.message.reply_text(**kwargs)
    elif update.effective_chat:
        await context.bot.send_message(update.effective_chat.id, **kwargs)


def _assignee_label(draft: dict[str, Any], members: list[dict[str, Any]]) -> str:
    uid = draft.get("assignee_user_id")
    if uid:
        for m in members:
            if member_id(m) == str(uid):
                label = member_display_name(m)
                if label:
                    return label
                email = m.get("email")
                if email:
                    return str(email).strip()
    return (draft.get("assignee_name") or "").strip() or "—"


def _board_label(draft: dict[str, Any]) -> str:
    label = (draft.get("project_label") or "").strip()
    if label:
        return label
    cn = draft.get("collection_name")
    if cn:
        p = project_by_collection(load_projects(), str(cn))
        if p:
            return p.label
    return "—"


def _display_task_title(title: str) -> str:
    t = (title or "Задача").strip()
    if t.startswith(BOT_TASK_TITLE_PREFIX):
        t = t[len(BOT_TASK_TITLE_PREFIX) :].strip() or title.strip()
    return t or "Задача"


def _task_created_message_html(title: str, link: str, draft: dict[str, Any], members: list[dict[str, Any]]) -> str:
    display_title = html.escape(_display_task_title(title))
    link_esc = html.escape(link)
    assignee = html.escape(_assignee_label(draft, members))
    board = html.escape(_board_label(draft))
    due = html.escape((draft.get("due_date") or "").strip() or "—")
    return (
        "Задача создана\n"
        f"{display_title}\n"
        f'{link_esc}\n\n'
        f"<b>Ответственный:</b> {assignee}\n"
        f"<b>Доска:</b> {board}\n"
        f"<b>Крайний срок:</b> {due}"
    )


def _projects_keyboard_task(projects: list[Project]) -> InlineKeyboardMarkup:
    rows = []
    for p in projects:
        cb = f"tp:{p.collection_name}"[:64]
        btn = f"{p.label} · доска {p.board_id}"[:64]
        rows.append([InlineKeyboardButton(text=btn, callback_data=cb)])
    return InlineKeyboardMarkup(rows)


def _columns_keyboard(columns: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    rows = []
    for c in columns[:12]:
        cid = c.get("id")
        name = str(c.get("name") or cid)[:40]
        if cid is not None:
            rows.append([InlineKeyboardButton(text=name, callback_data=f"tc:{cid}")])
    return InlineKeyboardMarkup(rows)


def _assignees_keyboard(members: list[dict[str, Any]]) -> InlineKeyboardMarkup:
    rows = []
    for m in members[:8]:
        uid = member_id(m)
        label = member_display_name(m)[:40]
        if uid:
            rows.append([InlineKeyboardButton(text=label, callback_data=f"ta:{uid}")])
    return InlineKeyboardMarkup(rows)


async def _resolve_column(draft: dict[str, Any], context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Вернуть True, если колонка определена."""
    board_id = draft.get("board_id")
    if not board_id:
        return False
    cols = await asyncio.to_thread(_columns_cached, context, int(board_id))
    name = (draft.get("column_name") or "").strip() or TASK_DEFAULT_COLUMN_NAME
    col = column_by_name(cols, name)
    if col and col.get("id") is not None:
        draft["board_column_id"] = int(col["id"])
        draft["column_name"] = str(col.get("name") or name)
        return True
    if name != TASK_DEFAULT_COLUMN_NAME:
        col = column_by_name(cols, TASK_DEFAULT_COLUMN_NAME)
        if col and col.get("id") is not None:
            draft["board_column_id"] = int(col["id"])
            draft["column_name"] = TASK_DEFAULT_COLUMN_NAME
            return True
    draft["awaiting"] = "column"
    draft["_column_list"] = cols
    return False


def _missing_field(draft: dict[str, Any]) -> str | None:
    if not (draft.get("task_text") or "").strip():
        return "task_text"
    if not draft.get("project_confirmed"):
        return "project"
    if not draft.get("assignee_user_id"):
        return "assignee"
    if draft.get("requester_unclear") and not (draft.get("requester_name") or "").strip():
        return "requester"
    if not draft.get("board_column_id"):
        return "column"
    return None


async def _ask_missing(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    draft: dict[str, Any],
    field: str,
) -> None:
    if field == "task_text":
        await _reply(update, context, "Опишите, что нужно сделать (суть задачи).")
    elif field == "project":
        projects = load_projects()
        await _reply(
            update,
            context,
            "Выберите проект и доску (в Weeek у одного проекта может быть несколько досок):",
            _projects_keyboard_task(projects),
        )
    elif field == "assignee":
        members = await asyncio.to_thread(_members_cached, context)
        if len(members) <= 8:
            await _reply(
                update,
                context,
                "Кто исполнитель?",
                _assignees_keyboard(members),
            )
        else:
            names = ", ".join(member_display_name(m) for m in members[:15])
            await _reply(update, context, f"Укажите исполнителя (имя из списка): {names}")
    elif field == "requester":
        draft["_asked_requester"] = True
        await _reply(update, context, "Кто постановщик задачи?")
    elif field == "column":
        cols = draft.get("_column_list") or []
        if cols:
            await _reply(
                update,
                context,
                f"На доске нет колонки «{TASK_DEFAULT_COLUMN_NAME}». Выберите колонку:",
                _columns_keyboard(cols),
            )
        else:
            await _reply(update, context, "Укажите название колонки на доске.")


async def _submit_task(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    draft: dict[str, Any],
) -> None:
    chat = update.effective_chat
    if chat:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

    _ensure_due_date(draft)
    _ensure_requester(draft)

    formulated = await asyncio.to_thread(
        formulate_task_for_weeek,
        {
            "task_text": draft.get("task_text"),
            "project_label": draft.get("collection_name"),
            "due_date": draft.get("due_date"),
            "assignee": draft.get("assignee_name"),
        },
    )
    title = str(formulated.get("title") or "Новая задача").strip()
    if not title.startswith(BOT_TASK_TITLE_PREFIX):
        title = f"{BOT_TASK_TITLE_PREFIX} {title}"
    description = _sanitize_formulated_description(str(formulated.get("description") or ""))
    payload: dict[str, Any] = {
        "title": title[:200],
        "description": description,
        "projectId": int(draft["project_id"]),
        "boardId": int(draft["board_id"]),
        "boardColumnId": int(draft["board_column_id"]),
        "type": "action",
    }
    if draft.get("due_date"):
        payload["day"] = draft["due_date"]
    if draft.get("assignee_user_id"):
        payload["userId"] = str(draft["assignee_user_id"])

    try:
        created = await asyncio.to_thread(create_task, payload)
    except WeeekApiError as e:
        logger.exception("create_task failed")
        await _reply(update, context, f"Не удалось создать задачу в Weeek: {e}")
        return

    tid = created.get("id")
    if tid is None:
        await _reply(update, context, "Задача создана, но ID не получен из ответа API.")
        clear_task_session(context)
        return

    link = task_url(tid)
    title = title or "Задача"
    members = _members_cached(context)
    msg = _task_created_message_html(title, link, draft, members)
    clear_task_session(context)
    await _reply(update, context, msg, parse_mode=ParseMode.HTML)


async def process_task_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
) -> None:
    if not await _require_weeek_token(update, context):
        return

    try:
        await _process_task_message_impl(update, context, text)
    except WeeekApiError as e:
        logger.exception("task flow Weeek API error")
        await _reply(update, context, f"Ошибка Weeek API: {e}")


async def _process_task_message_impl(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
) -> None:
    if not is_task_flow(context):
        context.user_data[USER_TASK_FLOW] = True

    draft = _get_draft(context)
    prev_awaiting = draft.get("awaiting")
    draft["awaiting"] = None
    text = (text or "").strip()

    if draft.pop("_asked_requester", False) and text:
        draft["requester_name"] = text.strip()
        draft["requester_unclear"] = False
    elif prev_awaiting == "requester" and text:
        draft["requester_name"] = text.strip()
        draft["requester_unclear"] = False

    if text:
        draft["messages"].append(text)

        # Прямые ответы на уточняющие вопросы
        if not draft.get("task_text") and len(text) > 3:
            draft["task_text"] = text
        if not draft.get("due_date"):
            resolved = _resolve_due_date(text, None)
            if resolved:
                draft["due_date"] = resolved
    projects = load_projects()
    members = await asyncio.to_thread(_members_cached, context)

    column_names: list[str] = []
    if draft.get("board_id"):
        cols = await asyncio.to_thread(_columns_cached, context, int(draft["board_id"]))
        column_names = [str(c.get("name") or "") for c in cols]

    history = draft.get("messages") or []
    latest = text or (history[-1] if history else "")
    if latest:
        try:
            extracted = await asyncio.to_thread(
                extract_task_draft_fields,
                latest,
                history,
                projects,
                members,
                column_names,
            )
            _merge_extracted(draft, extracted)
            if not draft.get("project_confirmed"):
                pk = extracted.get("project_key")
                if pk:
                    p = project_by_collection(projects, str(pk).strip())
                    if p:
                        _set_draft_project(draft, p)
        except Exception:
            logger.exception("extract_task_draft_fields failed")

    _apply_due_date_from_context(draft)
    _apply_project_from_context(draft, projects)
    _apply_assignee_from_context(draft, members)

    if draft.get("board_id") and not draft.get("board_column_id"):
        await _resolve_column(draft, context)

    _ensure_due_date(draft)

    missing = _missing_field(draft)
    if missing:
        await _ask_missing(update, context, draft, missing)
        return

    await _submit_task(update, context, draft)


async def handle_task_request(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
) -> None:
    if not weeek_api_configured():
        await _reply(update, context, _MISSING_TOKEN_MSG)
        return

    context.user_data[USER_TASK_FLOW] = True
    context.user_data[USER_TASK_DRAFT] = _new_draft()
    if not (text or "").strip():
        await _reply(update, context, "Опишите задачу. Отмена: /cancel")
    else:
        chat = update.effective_chat
        if chat:
            await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    await process_task_message(update, context, text)


async def on_task_project_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not query.data.startswith("tp:"):
        return
    await query.answer()
    coll = query.data[3:]
    projects = load_projects()
    p = project_by_collection(projects, coll)
    if not p:
        await query.edit_message_text("Проект не найден.")
        return
    draft = _get_draft(context)
    _set_draft_project(draft, p)
    await query.edit_message_text(f"Проект: {p.label} · доска {p.board_id}")
    await process_task_message(update, context, "")


async def on_task_column_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not query.data.startswith("tc:"):
        return
    await query.answer()
    try:
        cid = int(query.data[3:])
    except ValueError:
        await query.edit_message_text("Некорректная колонка.")
        return
    draft = _get_draft(context)
    draft["board_column_id"] = cid
    cols = draft.get("_column_list") or []
    for c in cols:
        if c.get("id") == cid:
            draft["column_name"] = str(c.get("name") or "")
            break
    await query.edit_message_text(f"Колонка: {draft.get('column_name') or cid}")
    await process_task_message(update, context, "")


async def on_task_assignee_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data or not query.data.startswith("ta:"):
        return
    await query.answer()
    uid = query.data[3:]
    draft = _get_draft(context)
    draft["assignee_user_id"] = uid
    members = await asyncio.to_thread(_members_cached, context)
    for m in members:
        if member_id(m) == uid:
            draft["assignee_name"] = member_display_name(m)
            break
    await query.edit_message_text(f"Исполнитель: {draft.get('assignee_name') or uid}")
    await process_task_message(update, context, "")
