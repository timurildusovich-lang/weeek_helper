from __future__ import annotations

import argparse
import json
from pathlib import Path

import tiktoken

from weeek_kb.config import DATA_DIR, OPENAI_EMBED_MODEL
from weeek_kb.parsing.html_utils import format_comments, strip_html
from weeek_kb.projects import Project, collection_name_from_stem, load_projects
from weeek_kb.search.vector_store import create_collection

# OpenAI embedding API: max 8192 tokens per input
_EMBED_MAX_TOKENS = 8000
_tiktoken_enc: tiktoken.Encoding | None = None


def _embedding_encoder() -> tiktoken.Encoding:
    global _tiktoken_enc
    if _tiktoken_enc is None:
        try:
            _tiktoken_enc = tiktoken.encoding_for_model(OPENAI_EMBED_MODEL)
        except KeyError:
            _tiktoken_enc = tiktoken.get_encoding("cl100k_base")
    return _tiktoken_enc


def truncate_for_embedding(text: str, max_tokens: int = _EMBED_MAX_TOKENS) -> tuple[str, bool]:
    """Clip text so OpenAI embeddings do not exceed the API token limit."""
    enc = _embedding_encoder()
    tokens = enc.encode(text)
    if len(tokens) <= max_tokens:
        return text, False
    return enc.decode(tokens[:max_tokens]), True


def build_document_text(task: dict, meta: dict) -> str:
    title = task.get("название") or ""
    desc = strip_html(task.get("описание") or "")
    comments = format_comments(task.get("comments"))
    parts = [
        f"Название: {title}",
        f"Описание:\n{desc}" if desc else "Описание:",
    ]
    if comments:
        parts.append(f"Комментарии:\n{comments}")
    label = meta.get("метка")
    if label:
        parts.append(f"Проект/метка: {label}")
    return "\n\n".join(parts)


def _chunk_header_lines(title: str, label: str | None, part: int, total: int) -> str:
    lines = [f"Название: {title}"]
    if label:
        lines.append(f"Проект/метка: {label}")
    if total > 1:
        lines.append(f"Фрагмент {part}/{total}")
    return "\n".join(lines) + "\n\n"


def chunk_task_for_embedding(
    task: dict,
    meta: dict,
    max_tokens: int = _EMBED_MAX_TOKENS,
) -> list[str]:
    """
    One embedding document per task if it fits; otherwise several chunks, each ≤ max_tokens,
    with the task title (and label) repeated in every chunk and description/comments split by tokens.
    """
    enc = _embedding_encoder()
    full = build_document_text(task, meta)
    if len(enc.encode(full)) <= max_tokens:
        return [full]

    title = task.get("название") or ""
    desc = strip_html(task.get("описание") or "")
    comments = format_comments(task.get("comments"))
    label = (meta.get("метка") or "").strip() or None

    body_parts: list[str] = []
    if desc:
        body_parts.append(f"Описание:\n{desc}")
    if comments:
        body_parts.append(f"Комментарии:\n{comments}")
    body = "\n\n".join(body_parts) if body_parts else ""

    if not body:
        t, _ = truncate_for_embedding(_chunk_header_lines(title, label, 1, 1).rstrip(), max_tokens)
        return [t]

    # Worst-case header size so body slices never overflow after we choose n.
    worst_header = _chunk_header_lines(title, label, 999, 999)
    overhead = len(enc.encode(worst_header))
    per_body = max(64, max_tokens - overhead - 8)
    body_tokens = enc.encode(body)
    n = max(1, (len(body_tokens) + per_body - 1) // per_body)

    chunks: list[str] = []
    for i in range(n):
        start = i * len(body_tokens) // n
        end = (i + 1) * len(body_tokens) // n
        slice_tokens = body_tokens[start:end]
        header = _chunk_header_lines(title, label, i + 1, n)
        piece = header + enc.decode(slice_tokens)
        if len(enc.encode(piece)) > max_tokens:
            room = max_tokens - len(enc.encode(header))
            if room < 1:
                piece = header
            else:
                piece = header + enc.decode(slice_tokens[:room])
        piece, _ = truncate_for_embedding(piece, max_tokens)
        chunks.append(piece)

    return chunks


def task_metadata(task: dict, meta: dict) -> dict:
    """Chroma: metadata values must be str / int / float / bool — use str for ids in filters."""
    created = task.get("датаСоздания") or ""
    completed = task.get("датаЗавершения") or ""
    if completed is None:
        completed = ""
    return {
        "task_id": str(int(task["id"])),
        "title": (task.get("название") or "")[:2000],
        "status": (task.get("статус") or "")[:200],
        "column": (task.get("колонка") or "")[:200],
        "created": str(created)[:64],
        "completed": str(completed)[:64],
        "project_id": str(int(meta.get("projectId") or 0)),
        "board_id": str(int(meta.get("boardId") or 0)),
        "label": str(meta.get("метка") or "")[:500],
    }


def ingest_file(project: Project, reset: bool) -> int:
    with open(project.file_path, encoding="utf-8") as f:
        data = json.load(f)
    meta = data.get("meta") or {}
    tasks = data.get("задачи") or []
    name = project.collection_name
    col = create_collection(name, reset=reset)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict] = []
    chunked_tasks = 0

    for task in tasks:
        tid = int(task["id"])
        task_chunks = chunk_task_for_embedding(task, meta)
        if len(task_chunks) > 1:
            chunked_tasks += 1
        for ci, doc in enumerate(task_chunks):
            ids.append(f"{tid}_{ci}")
            documents.append(doc)
            m = task_metadata(task, meta)
            m["chunk_index"] = str(ci)
            m["chunk_total"] = str(len(task_chunks))
            metadatas.append(m)

    # OpenAI embeddings: max ~300k tokens per request; chunks may be up to _EMBED_MAX_TOKENS each.
    batch = 16
    for i in range(0, len(ids), batch):
        col.add(
            ids=ids[i : i + batch],
            documents=documents[i : i + batch],
            metadatas=metadatas[i : i + batch],
        )
    if chunked_tasks:
        print(f"  note: {chunked_tasks} task(s) split into multiple chunks (<={_EMBED_MAX_TOKENS} tokens each)")
    return len(ids)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Weeek JSON boards into ChromaDB")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory with board-*.json",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Only this collection stem (e.g. board-6-avrora-kanc-rf)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate collections",
    )
    args = parser.parse_args()

    projects = load_projects(args.data_dir)
    if args.only:
        stem = args.only.replace(".json", "")
        projects = [p for p in projects if p.file_stem == stem or p.collection_name == collection_name_from_stem(stem)]
        if not projects:
            raise SystemExit(f"No project matches --only {args.only!r}")

    total = 0
    for p in projects:
        n = ingest_file(p, reset=args.reset)
        print(f"{p.file_path.name} -> collection={p.collection_name} embedding_rows={n}")
        total += n
    print(f"Done. Total embedding rows (chunks): {total}")


if __name__ == "__main__":
    main()
