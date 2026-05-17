from __future__ import annotations

import asyncio
import html as html_module
import logging
import re
import secrets
from typing import Any

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ChatAction, ParseMode
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from weeek_kb.config import (
    INTENT_CONFIDENCE_THRESHOLD,
    ROOT,
    TELEGRAM_BOT_TOKEN,
    TOP_K_AFTER_MERGE,
    TOP_OVERFLOW,
    VECTOR_SEARCH_PER_QUERY,
)
from weeek_kb.add.task_add import (
    clear_task_session,
    handle_task_request,
    is_task_flow,
    on_task_assignee_callback,
    on_task_column_callback,
    on_task_project_callback,
    process_task_message,
)
from weeek_kb.intent import classify_message_intent
from weeek_kb.projects import Project, guess_project_from_keywords, load_projects, project_by_collection
from weeek_kb.search.llm import (
    build_answer_html,
    detect_project,
    merge_vector_hits,
    pick_top_tasks,
    reformulate_queries,
    summarize_overflow_tasks,
)
from weeek_kb.search.vector_store import query_collection
from weeek_kb.transcribe import transcribe_audio_bytes

LOG_PATH = ROOT / "weeek_kb.log"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

NO_CONFIDENT_ANSWER = (
    "По задачам в базе нельзя дать уверенный ответ на этот вопрос: в подобранных карточках "
    "нет достаточной информации. Я намеренно не додумываю ответ."
)

USER_PROJECT = "project_collection"
USER_PENDING = "pending_query"
USER_PENDING_INTENT = "pending_intent_text"

# Telegram лимит сообщения; запас под entities
TG_HTML_MAX = 4000


def _split_html_by_p_blocks(fragment: str, max_len: int = TG_HTML_MAX) -> list[str]:
    """Режет по целым <p>...</p>, чтобы не ломать HTML при длинном ответе."""
    if len(fragment) <= max_len:
        return [fragment]
    blocks = re.findall(r"<p>.*?</p>", fragment, flags=re.DOTALL)
    if not blocks:
        return [fragment[i : i + max_len] for i in range(0, len(fragment), max_len)]
    out: list[str] = []
    buf: list[str] = []
    n = 0
    for b in blocks:
        if buf and n + len(b) > max_len:
            out.append("".join(buf))
            buf = [b]
            n = len(b)
        else:
            buf.append(b)
            n += len(b)
    if buf:
        out.append("".join(buf))
    return out


def _link_to_plain(m: re.Match[str]) -> str:
    href_raw = m.group(1) or ""
    inner_raw = (m.group(2) or "").strip()
    href = html_module.unescape(href_raw.replace("&amp;", "&"))
    inner = html_module.unescape(inner_raw)
    if not inner:
        return href
    if inner == href:
        return inner
    # Совпадение после нормализации (слэш, сущности)
    if inner.rstrip("/") == href.rstrip("/"):
        return inner
    if inner.startswith("http") and href.startswith("http") and inner.rstrip("/") == href.rstrip("/"):
        return href
    return f"{inner} ({href})"


def _html_to_plain_fallback(fragment: str) -> str:
    """Если Telegram отверг HTML — показать текст без тегов, без сырого <br/>."""
    t = re.sub(r"<br\s*/?>", "\n", fragment, flags=re.IGNORECASE)
    t = re.sub(r"</p>\s*", "\n", t)
    t = re.sub(r"<p[^>]*>", "", t)
    t = re.sub(r'<a\s+href="([^"]*)"[^>]*>([^<]*)</a>', _link_to_plain, t)
    t = re.sub(r"</?b>", "", t)
    t = re.sub(r"<[^>]+>", "", t)
    t = html_module.unescape(t)
    return re.sub(r"\n{3,}", "\n\n", t).strip()[:TG_HTML_MAX]


def _projects_keyboard(projects: list[Project]) -> InlineKeyboardMarkup:
    rows = []
    for p in projects:
        cb = f"p:{p.collection_name}"[:64]
        rows.append([InlineKeyboardButton(text=p.label[:64], callback_data=cb)])
    return InlineKeyboardMarkup(rows)


def _intent_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Вопрос", callback_data="i:q"),
                InlineKeyboardButton("Поставить задачу", callback_data="i:t"),
            ],
        ]
    )


async def _send_overflow_html(context: ContextTypes.DEFAULT_TYPE, chat_id: int, html: str) -> None:
    """Отправка блока «остальные задачи»: HTML, при отказе Telegram — plain без дублей URL."""
    raw = (html or "").strip()
    if not raw:
        return
    chunks = _split_html_by_p_blocks(raw)
    try:
        for chunk in chunks:
            await context.bot.send_message(chat_id=chat_id, text=chunk, parse_mode=ParseMode.HTML)
            await asyncio.sleep(0.05)
    except BadRequest as e:
        logger.warning("overflow HTML rejected, sending plain: %s", e)
        plain = _html_to_plain_fallback(raw)
        for i in range(0, len(plain), TG_HTML_MAX):
            await context.bot.send_message(chat_id=chat_id, text=plain[i : i + TG_HTML_MAX])
            await asyncio.sleep(0.05)


async def cmd_cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if is_task_flow(context):
        clear_task_session(context)
        msg = "Постановка задачи отменена."
    else:
        msg = "Нет активной постановки задачи."
    if update.message:
        await update.message.reply_text(msg)


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "👋 Привет! Я отвечаю на вопросы по задачам из Weeek и принимаю постановку новых задач.\n\n"
        "Вопрос — ищу по доскам и собираю ответ из описаний и комментариев. "
        "Постановка задачи — опиши, что нужно сделать; при необходимости уточню детали и создам карточку в Weeek (/cancel — отмена).\n\n"
        "Пиши обычным языком, по делу: чем конкретнее запрос, тем полезнее ответ.\n\n"
        "🌐 Укажи в сообщении название сайта. Формулируй запрос подробно, можно голосовым.\n\n"
        "🖊 В ответе будут до трёх релевантных задач со ссылками; остальные совпадения можно открыть кнопкой "
        "«Показать другие задачи».\n\n"
        "🔥 Приоритет задачам в статусе «Завершена», новым задачам. Заниженный приоритет задачам в колонке «Идеи».\n\n"
        "📆 Актуальная база до 17.04.2026"
    )
    if update.message:
        logger.info("cmd_start chat_id=%s", update.effective_chat.id if update.effective_chat else None)
        await update.message.reply_text(text)


async def on_intent_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    data = query.data
    if not data.startswith("i:"):
        return
    choice = data[2:]
    pending = context.user_data.pop(USER_PENDING_INTENT, None)
    if not pending:
        await query.edit_message_text("Сообщение устарело. Напишите снова.")
        return
    if choice == "q":
        await query.edit_message_text("Понял: вопрос.")
        await process_question_query(update, context, pending)
    elif choice == "t":
        await query.edit_message_text("Понял: постановка задачи.")
        await handle_task_request(update, context, pending)
    else:
        await query.edit_message_text("Некорректная кнопка.")


async def on_project_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    await query.answer()
    data = query.data
    if not data.startswith("p:"):
        return
    coll = data[2:]
    projects = load_projects()
    p = project_by_collection(projects, coll)
    if not p:
        await query.edit_message_text("Проект не найден. Попробуйте /start.")
        return
    context.user_data[USER_PROJECT] = coll
    pending = context.user_data.pop(USER_PENDING, None)
    if pending:
        await query.edit_message_text(f"Проект: {p.label}. Ищу ответ…")
        await run_pipeline(update, context, pending, p)
    else:
        await query.edit_message_text(f"Выбран проект: {p.label}. Напиши вопрос текстом.")


def _overflow_storage_key(user_id: int, token: str) -> str:
    return f"ovf_{user_id}_{token}"


async def on_show_all_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query or not query.data:
        return
    data = query.data
    try:
        await query.answer()
    except BadRequest:
        pass

    if not data.startswith("show:"):
        return
    token = data[5:].strip()
    if len(token) != 16:
        if query.message:
            await query.message.reply_text("Некорректная кнопка. Задай вопрос ещё раз.")
        return

    uid = query.from_user.id if query.from_user else 0
    key = _overflow_storage_key(uid, token)
    html = context.bot_data.get(key)
    if not html:
        if query.message:
            await query.message.reply_text("Данные устарели — задай вопрос ещё раз.")
        return

    chat = update.effective_chat
    if not chat:
        return

    try:
        await _send_overflow_html(context, chat.id, html)
        context.bot_data.pop(key, None)
    except Exception:
        logger.exception("send overflow failed")
        if query.message:
            await query.message.reply_text(
                "Не удалось отправить список задач. Задай вопрос ещё раз или открой weeek_kb.log.",
            )


async def run_pipeline(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    user_query: str,
    project: Project,
) -> None:
    chat = update.effective_chat
    if chat:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

    q1, q2, q3 = reformulate_queries(user_query)
    logger.info("Queries: %r | %r | %r", q1, q2, q3)

    async def one_q(q: str) -> dict[str, Any]:
        return await asyncio.to_thread(
            query_collection,
            project.collection_name,
            q,
            VECTOR_SEARCH_PER_QUERY,
        )

    r1, r2, r3 = await asyncio.gather(one_q(q1), one_q(q2), one_q(q3))
    merged = merge_vector_hits([r1, r2, r3], TOP_K_AFTER_MERGE)

    if not merged:
        msg = "По этому проекту ничего похожего не нашлось. Переформулируй вопрос или проверь индекс (ingest)."
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(msg)
        elif update.message:
            await update.message.reply_text(msg)
        return

    id_to_meta = {t[0]: t[2] for t in merged}
    id_to_doc = {t[0]: t[3] for t in merged}

    if chat:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    top3_ids, insufficient = await asyncio.to_thread(pick_top_tasks, user_query, merged)
    top3_ids = top3_ids[:3]
    if insufficient or not top3_ids:
        await _reply_plain(
            update,
            context,
            NO_CONFIDENT_ANSWER,
        )
        return

    top_set = set(top3_ids)

    rest_tuples = [t for t in merged if t[0] not in top_set][:TOP_OVERFLOW]

    if chat:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    answer_html = await asyncio.to_thread(
        build_answer_html,
        user_query,
        top3_ids,
        id_to_doc,
        id_to_meta,
    )

    overflow_token: str | None = None
    if rest_tuples:
        if chat:
            await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
        overflow_items = [(tid, id_to_meta[tid], id_to_doc[tid]) for tid, _, _, _ in rest_tuples]
        overflow_html = await asyncio.to_thread(
            summarize_overflow_tasks,
            user_query,
            overflow_items,
        )
        uid = update.effective_user.id if update.effective_user else 0
        overflow_token = secrets.token_hex(8)
        context.bot_data[_overflow_storage_key(uid, overflow_token)] = overflow_html

    keyboard = None
    if rest_tuples and overflow_token:
        cb = f"show:{overflow_token}"
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("Показать другие задачи", callback_data=cb)]]
        )

    await _reply_html(update, context, answer_html, keyboard)


async def _reply_plain(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
) -> None:
    if update.callback_query and update.callback_query.message:
        await update.callback_query.message.reply_text(text)
    elif update.message:
        await update.message.reply_text(text)


async def _reply_html(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    html: str,
    keyboard: InlineKeyboardMarkup | None,
) -> None:
    raw = (html or "").strip()
    if not raw:
        return
    chunks = _split_html_by_p_blocks(raw)
    try:
        for i, chunk in enumerate(chunks):
            is_last = i == len(chunks) - 1
            markup = keyboard if is_last else None
            if update.callback_query and update.callback_query.message:
                await update.callback_query.message.reply_text(
                    chunk,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
            elif update.message:
                await update.message.reply_text(
                    chunk,
                    parse_mode=ParseMode.HTML,
                    reply_markup=markup,
                )
            if not is_last:
                await asyncio.sleep(0.05)
    except BadRequest as e:
        logger.warning("HTML parse failed, sending as plain (keyboard preserved): %s", e)
        plain = _html_to_plain_fallback(html)
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(
                plain,
                reply_markup=keyboard,
            )
        elif update.message:
            await update.message.reply_text(
                plain,
                reply_markup=keyboard,
            )


async def process_question_query(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text: str,
) -> None:
    """Вопрос по базе задач: определение проекта и run_pipeline."""
    text = text.strip()
    if not text:
        return

    projects = load_projects()
    coll = context.user_data.get(USER_PROJECT)

    kw = guess_project_from_keywords(text, projects)
    guessed_llm = await asyncio.to_thread(detect_project, text, projects)

    guessed: Project | None = None
    if kw and guessed_llm and kw.collection_name != guessed_llm.collection_name:
        guessed = kw
        logger.info(
            "project: keyword=%s overrides llm=%s",
            kw.collection_name,
            guessed_llm.collection_name,
        )
    elif kw:
        guessed = kw
    elif guessed_llm:
        guessed = guessed_llm

    if guessed:
        context.user_data[USER_PROJECT] = guessed.collection_name
        if coll and coll != guessed.collection_name:
            logger.info("project switch: %s -> %s", coll, guessed.collection_name)
        await run_pipeline(update, context, text, guessed)
        return

    if not coll:
        context.user_data[USER_PENDING] = text
        msg = "Не удалось понять, о каком проекте речь. Выбери доску:"
        markup = _projects_keyboard(projects)
        if update.callback_query and update.callback_query.message:
            await update.callback_query.message.reply_text(msg, reply_markup=markup)
        elif update.message:
            await update.message.reply_text(msg, reply_markup=markup)
        return

    proj = project_by_collection(projects, coll)
    if not proj:
        context.user_data.pop(USER_PROJECT, None)
        await _reply_plain(update, context, "Сессия сброшена. Напиши вопрос снова.")
        return

    await run_pipeline(update, context, text, proj)


async def process_user_query(update: Update, context: ContextTypes.DEFAULT_TYPE, text: str) -> None:
    """Общая логика для текста и для результата распознавания голоса."""
    if not update.message:
        return
    text = text.strip()
    if not text:
        return

    logger.info(
        "query chat_id=%s len=%s preview=%r",
        update.effective_chat.id if update.effective_chat else None,
        len(text),
        text[:120],
    )

    chat = update.effective_chat
    if chat:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

    try:
        intent, conf = await asyncio.to_thread(classify_message_intent, text)
    except Exception:
        logger.exception("intent classification failed")
        intent, conf = None, 0.0

    logger.info("intent=%r confidence=%.2f threshold=%.2f", intent, conf, INTENT_CONFIDENCE_THRESHOLD)

    threshold = INTENT_CONFIDENCE_THRESHOLD
    if intent == "task" and conf >= threshold:
        await handle_task_request(update, context, text)
        return
    if intent == "question" and conf >= threshold:
        await process_question_query(update, context, text)
        return

    context.user_data[USER_PENDING_INTENT] = text
    await update.message.reply_text(
        "Не уверен, что вы имели в виду. Уточните:",
        reply_markup=_intent_keyboard(),
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text:
        return
    text = update.message.text.strip()
    if not text:
        return
    if is_task_flow(context):
        await process_task_message(update, context, text)
        return
    await process_user_query(update, context, text)


async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.voice:
        return
    chat = update.effective_chat
    if chat:
        await context.bot.send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
    try:
        tg_file = await context.bot.get_file(update.message.voice.file_id)
        data = bytes(await tg_file.download_as_bytearray())
    except Exception:
        logger.exception("voice download failed")
        await update.message.reply_text("Не удалось скачать голосовое сообщение. Попробуйте ещё раз.")
        return
    try:
        text = await asyncio.to_thread(transcribe_audio_bytes, data, "voice.ogg")
    except Exception:
        logger.exception("OpenAI transcription failed")
        await update.message.reply_text(
            "Не удалось распознать речь. Проверьте ключ OpenAI и доступ к API транскрибации."
        )
        return
    if not text.strip():
        await update.message.reply_text("Речь не распознана. Попробуйте громче или напишите текстом.")
        return
    preview = text.strip()
    if len(preview) > 500:
        preview = preview[:497] + "…"
    await update.message.reply_text(f"🎤 {preview}")
    if is_task_flow(context):
        await process_task_message(update, context, text.strip())
    else:
        await process_user_query(update, context, text.strip())


async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled exception in handler: %s", context.error)
    if isinstance(update, Update) and update.effective_chat:
        try:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Произошла внутренняя ошибка. Подробности в файле weeek_kb.log рядом с ботом.",
            )
        except Exception:
            logger.exception("Failed to notify user about error")


def main() -> None:
    if not TELEGRAM_BOT_TOKEN:
        raise SystemExit("TELEGRAM_BOT_TOKEN (or TELEGRAM_TOKEN) is not set")

    logger.info("Log file: %s", LOG_PATH.resolve())
    app = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    app.add_error_handler(on_error)
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("cancel", cmd_cancel))
    app.add_handler(CallbackQueryHandler(on_intent_callback, pattern=r"^i:"))
    app.add_handler(CallbackQueryHandler(on_task_project_callback, pattern=r"^tp:"))
    app.add_handler(CallbackQueryHandler(on_task_column_callback, pattern=r"^tc:"))
    app.add_handler(CallbackQueryHandler(on_task_assignee_callback, pattern=r"^ta:"))
    app.add_handler(CallbackQueryHandler(on_project_callback, pattern=r"^p:"))
    app.add_handler(CallbackQueryHandler(on_show_all_callback, pattern=r"^show:"))
    app.add_handler(MessageHandler(filters.VOICE, handle_voice))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("Bot polling…")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
