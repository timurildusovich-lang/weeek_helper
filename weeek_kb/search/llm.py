from __future__ import annotations

import html
import json
import re
from typing import Any

from weeek_kb._openai import get_openai_client
from weeek_kb.config import OPENAI_CHAT_MODEL, TOP_PRESENT, task_url
from weeek_kb.projects import Project
from weeek_kb.search.query_normalize import normalize_search_query


def detect_project(user_query: str, projects: list[Project]) -> Project | None:
    """Return a project if the model is confident; else None."""
    if not projects:
        return None
    if len(projects) == 1:
        return projects[0]

    lines = [f"- key={p.collection_name!r} label={p.label!r} file={p.file_stem}" for p in projects]
    system = (
        "Ты определяешь, о каком проекте (доске) из списка идёт речь в ЭТОМ сообщении пользователя. "
        "Каждое новое сообщение может относиться к другому проекту — ориентируйся только на текст вопроса, "
        "не предполагай предыдущий контекст. Если названы фамилия, домен, бренд или метка — сопоставь с label/file.\n"
        "Ответь строго одним JSON: {\"key\": \"...\" | null, \"confidence\": 0..1}. "
        "key — одно из значений key из списка, либо null если нельзя уверенно выбрать один проект."
    )
    user = f"Вопрос:\n{user_query}\n\nПроекты:\n" + "\n".join(lines)
    resp = get_openai_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    key = data.get("key")
    conf = float(data.get("confidence") or 0)
    if not key or conf < 0.55:
        return None
    for p in projects:
        if p.collection_name == key:
            return p
    return None


def reformulate_queries(original: str) -> tuple[str, str, str]:
    """Original + two paraphrases for hybrid search; строки нормализуются (1С→1C и т.д.)."""
    original = normalize_search_query(original.strip())
    system = (
        "Сгенерируй два альтернативных формулирования того же вопроса по-русски "
        "(учитывай разный транслит имён и брендов, если они есть). "
        "Ответ строго JSON: {\"q2\": \"...\", \"q3\": \"...\"} без пояснений."
    )
    resp = get_openai_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Оригинал:\n{original}"},
        ],
        temperature=0.6,
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{[\s\S]*\}", text)
    q2, q3 = original, original
    if m:
        try:
            data = json.loads(m.group(0))
            q2 = str(data.get("q2") or original).strip() or original
            q3 = str(data.get("q3") or original).strip() or original
        except json.JSONDecodeError:
            pass
    return (
        normalize_search_query(original),
        normalize_search_query(q2),
        normalize_search_query(q3),
    )


def merge_vector_hits(
    results_list: list[dict[str, Any]],
    top_k: int,
) -> list[tuple[str, float, dict[str, Any], str]]:
    """
    Merge Chroma query results: group by logical task_id (metadata.task_id),
    dedupe chunk rows by Chroma id keeping best distance, concatenate chunk texts in order.
    Returns list of (task_id, distance, metadata, merged_document) sorted by distance ascending.
    """
    chunk_best: dict[tuple[str, str], tuple[float, dict[str, Any], str]] = {}
    for res in results_list:
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        for i, cid in enumerate(ids):
            d = float(dists[i]) if i < len(dists) else 1e9
            meta = metas[i] if i < len(metas) else {}
            doc = docs[i] if i < len(docs) else ""
            base = str(meta.get("task_id") or (cid.split("_")[0] if cid else ""))
            key = (base, cid)
            prev = chunk_best.get(key)
            if prev is None or d < prev[0]:
                chunk_best[key] = (d, meta, doc)

    by_task: dict[str, list[tuple[float, dict[str, Any], str]]] = {}
    for (base, _cid), (d, meta, doc) in chunk_best.items():
        by_task.setdefault(base, []).append((d, meta, doc))

    def _chunk_order(row: tuple[float, dict[str, Any], str]) -> int:
        m = row[1]
        try:
            return int(m.get("chunk_index", 0))
        except (TypeError, ValueError):
            return 0

    merged: list[tuple[str, float, dict[str, Any], str]] = []
    for base_id, rows in by_task.items():
        best_d = min(r[0] for r in rows)
        rows_sorted = sorted(rows, key=_chunk_order)
        merged_doc = "\n\n---\n\n".join(r[2] for r in rows_sorted)
        meta0 = dict(rows_sorted[0][1])
        merged.append((base_id, best_d, meta0, merged_doc))
    merged.sort(key=lambda x: x[1])
    return merged[:top_k]


def pick_top_tasks(
    user_query: str,
    candidates: list[tuple[str, float, dict[str, Any], str]],
) -> tuple[list[str], bool]:
    """OpenAI picks TOP_PRESENT task ids; (ids, insufficient) if nothing answers the question."""
    if len(candidates) <= TOP_PRESENT:
        return [t[0] for t in candidates], False
    payload = []
    for tid, dist, meta, doc in candidates:
        payload.append(
            {
                "id": tid,
                "vector_distance": round(dist, 6),
                "title": meta.get("title"),
                "status": meta.get("status"),
                "column": meta.get("column"),
                "created": meta.get("created"),
                "completed": meta.get("completed"),
                "text": doc[:12000],
            }
        )
    system = (
        "Из списка релевантных задач выбери до "
        f"{TOP_PRESENT} id, которые реально помогают ответить на вопрос по содержанию текста.\n"
        "Если ни одна задача не содержит информации для ответа (или только отдалённо похожа), "
        'верни строго: {"insufficient": true, "ids": []}.\n'
        "Иначе: приоритет — новее по дате создания; колонка «Сделано»; статус «Завершена»; "
        "понижающий приоритет — колонка «Идеи».\n"
        f'Ответ JSON: {{"insufficient": false, "ids": ["id1","id2","id3"]}} — ids ровно {TOP_PRESENT} строк, '
        "если кандидатов хватает; иначе меньше или insufficient."
    )
    resp = get_openai_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Вопрос:\n{user_query}\n\nКандидаты (JSON):\n{json.dumps(payload, ensure_ascii=False)}"},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content or ""
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return [c[0] for c in candidates[:TOP_PRESENT]], False
    try:
        data = json.loads(m.group(0))
        if data.get("insufficient"):
            return [], True
        ids = data.get("ids") or []
        out = [str(x) for x in ids][:TOP_PRESENT]
        valid = {c[0] for c in candidates}
        out = [x for x in out if x in valid]
        if len(out) < TOP_PRESENT:
            for c in candidates:
                if c[0] not in out:
                    out.append(c[0])
                if len(out) >= TOP_PRESENT:
                    break
        return out[:TOP_PRESENT], False
    except json.JSONDecodeError:
        return [c[0] for c in candidates[:TOP_PRESENT]], False


def _esc(s: str) -> str:
    return html.escape(s or "", quote=False)


def _esc_body_html(s: str) -> str:
    """Экранирование и перевод строк в <br/> (списки и абзацы внутри карточки задачи)."""
    if not (s or "").strip():
        return ""
    return "<br/>".join(_esc(line) for line in (s or "").split("\n"))


def _href_attr(url: str) -> str:
    return (url or "").replace("&", "&amp;")


def _strip_json_fence(raw: str) -> str:
    t = (raw or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```\w*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _strip_duplicate_paren_url(body: str, task_id: str) -> str:
    """Убирает из текста модели повтор URL в скобках для этой же задачи."""
    if not body:
        return body
    u = task_url(str(task_id))
    return re.sub(r"\s*\(" + re.escape(u) + r"\)", "", body).strip()


def _task_block_html(task_id: str, title: str, body: str) -> str:
    """
    Два <p> подряд: строка 🔹 + одна кликабельная ссылка (без <b> внутри <a> — иначе Telegram ломает разметку),
    затем «В задаче "название"» + текст.
    """
    url = task_url(task_id)
    href = _href_attr(url)
    u_esc = _esc(url)
    t_esc = _esc(title or task_id)
    b = (body or "").strip()
    p_link = f'<p>🔹 <a href="{href}">{u_esc}</a></p>'
    if b:
        p_body = f'<p>В задаче <b>"{t_esc}"</b> {b}</p>'
    else:
        p_body = f'<p>В задаче <b>"{t_esc}"</b> См. описание и комментарии в карточке.</p>'
    return p_link + p_body


TASK_BLOCK_GAP = "<p><br/></p>"


def _collapse_extra_breaks(fragment: str) -> str:
    s = (fragment or "").strip()
    s = re.sub(r"<p>\s*</p>", "", s)
    return s


def _fallback_narrative(top_ids: list[str], id_to_meta: dict[str, dict[str, Any]]) -> str:
    blocks = []
    for tid in top_ids:
        title = (id_to_meta.get(tid) or {}).get("title") or tid
        blocks.append(_task_block_html(str(tid), str(title), ""))
    return _collapse_extra_breaks(TASK_BLOCK_GAP.join(blocks))


def _main_docs_payload(
    top_ids: list[str],
    id_to_doc: dict[str, str],
    id_to_meta: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tid in top_ids:
        m = id_to_meta.get(tid) or {}
        out.append(
            {
                "id": tid,
                "title": m.get("title"),
                "status": m.get("status"),
                "column": m.get("column"),
                "text": id_to_doc.get(tid, ""),
            }
        )
    return out


def _system_prompt_answer(order: str, n_tasks: int) -> str:
    return (
        "Ты отвечаешь посетителю на его вопрос по данным из задач Weeek. Источник правды — поле text каждой задачи "
        "(описание и комментарии), при необходимости status и column.\n\n"
        "Главное: дай именно ОТВЕТ НА ВОПРОС посетителя, а не пересказ всей карточки и не копипаст всего текста задачи. "
        "Ссылка на задачу и её название уже показываются в интерфейсе отдельно — в поле text для каждой задачи напиши "
        "только то, что из описания и комментариев этой задачи относится к вопросу: сформулируй по смыслу. "
        "Если для ответа нужны конкретные пункты (страницы, URL, списки) — перечисли их выборочно из текста задачи, "
        "без лишнего шума и без учётных данных, если они не нужны для ответа на вопрос.\n\n"
        "Поле text может содержать несколько коротких абзацев и строк со списком (строки с «- »); переносы строк допустимы.\n"
        "Если вопрос про готовность/статус: опирайся на status/column. «Завершена» — можно сказать о завершении по Weeek; "
        "«Активна» / идеи / бэклог — не утверждай, что работа полностью сделана.\n\n"
        "Если по выбранным текстам нельзя честно ответить — верни JSON: {\"insufficient\": true}.\n"
        "Иначе верни JSON (без markdown, без HTML внутри строк):\n"
        '{"intro": "одно-два предложения общего ответа на вопрос или пустая строка", '
        '"paragraphs": ['
        '{"task_id": "<id>", "text": "фрагмент ответа, основанный на этой задаче"}'
        "]}\n"
        f"Ровно {n_tasks} объектов в paragraphs, порядок task_id: {order}. "
        "Каждый text непустой. Не выдумывай факты. Не дублируй URL карточки Weeek и не добавляй его в скобках."
    )


def _render_answer_from_json(
    data: dict[str, Any],
    top_ids: list[str],
    id_to_meta: dict[str, dict[str, Any]],
) -> str:
    if data.get("insufficient"):
        return (
            "<p>По текстам выбранных задач нельзя собрать ответ на этот вопрос без додумывания.</p>"
        )
    intro = (data.get("intro") or "").strip()
    paragraphs = data.get("paragraphs") or []
    by_id = {str(p.get("task_id")): p for p in paragraphs if isinstance(p, dict)}
    task_blocks: list[str] = []
    for tid in top_ids:
        para = by_id.get(str(tid)) or {}
        body_raw = str(para.get("text") or "").strip()
        body_raw = _strip_duplicate_paren_url(body_raw, tid)
        body_html = _esc_body_html(body_raw)
        title = (id_to_meta.get(tid) or {}).get("title") or tid
        task_blocks.append(_task_block_html(str(tid), str(title), body_html if body_raw else ""))
    chunks: list[str] = []
    if intro:
        chunks.append(f"<p>{_esc_body_html(intro)}</p>")
    if task_blocks:
        if intro:
            chunks.append(TASK_BLOCK_GAP)
        chunks.append(TASK_BLOCK_GAP.join(task_blocks))
    return _collapse_extra_breaks("".join(chunks))


def build_answer_html(
    user_query: str,
    top_ids: list[str],
    id_to_doc: dict[str, str],
    id_to_meta: dict[str, dict[str, Any]],
) -> str:
    """Один вызов LLM: ответ на вопрос посетителя по 3 задачам; разметка — в коде (Telegram HTML)."""
    main_docs = _main_docs_payload(top_ids, id_to_doc, id_to_meta)
    order = ", ".join(top_ids)
    system = _system_prompt_answer(order, len(top_ids))
    user = json.dumps({"question": user_query, "tasks": main_docs}, ensure_ascii=False)
    resp = get_openai_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
    )
    raw = _strip_json_fence(resp.choices[0].message.content or "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _fallback_narrative(top_ids, id_to_meta)
    return _render_answer_from_json(data, top_ids, id_to_meta)


def summarize_overflow_tasks(
    user_query: str,
    items: list[tuple[str, dict[str, Any], str]],
) -> str:
    """Краткие описания остальных задач; разметка в коде."""
    if not items:
        return ""
    payload = [
        {"id": tid, "title": meta.get("title"), "text": doc[:6000]}
        for tid, meta, doc in items
    ]
    system = (
        "По каждой задаче напиши одну короткую строку только по полю text: суть и статус. "
        "Верни ТОЛЬКО JSON: {\"lines\": [{\"task_id\": \"...\", \"blurb\": \"...\"}]} "
        "в том же порядке, что и список задач. Без HTML. Не выдумывай."
    )
    resp = get_openai_client().chat.completions.create(
        model=OPENAI_CHAT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({"question": user_query, "tasks": payload}, ensure_ascii=False)},
        ],
        temperature=0.3,
    )
    raw = _strip_json_fence(resp.choices[0].message.content or "")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}

    lines = data.get("lines") or []
    by_id = {str(x.get("task_id")): x for x in lines if isinstance(x, dict)}
    task_chunks: list[str] = []
    for tid, meta, _doc in items:
        row = by_id.get(str(tid), {})
        blurb = _esc(str(row.get("blurb") or "См. карточку задачи.").strip())
        title = (meta or {}).get("title") or tid
        url = task_url(tid)
        href = _href_attr(url)
        u_esc = _esc(url)
        t_esc = _esc(str(title))
        task_chunks.append(
            f'<p>🔹 <a href="{href}">{u_esc}</a></p>'
            f'<p><b>"{t_esc}"</b> — {blurb}</p>'
        )
    return _collapse_extra_breaks(TASK_BLOCK_GAP.join(task_chunks))
