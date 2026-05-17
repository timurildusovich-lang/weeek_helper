from __future__ import annotations

import re

from bs4 import BeautifulSoup


def strip_html(html: str | None) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def format_comments(comments: list[dict] | None) -> str:
    if not comments:
        return ""
    parts: list[str] = []
    for c in comments:
        author = c.get("автор") or ""
        date = c.get("дата") or ""
        body = strip_html(c.get("комментарий") or "")
        if not body:
            continue
        head = f"{author} ({date})".strip()
        if head.startswith("("):
            head = date
        parts.append(f"- {head}: {body}" if head else f"- {body}")
    return "\n".join(parts)
