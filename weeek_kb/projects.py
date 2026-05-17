from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from weeek_kb.config import DATA_DIR


@dataclass(frozen=True)
class Project:
    """One board JSON file → one Chroma collection."""

    file_path: Path
    collection_name: str
    label: str
    project_id: int
    board_id: int

    @property
    def file_stem(self) -> str:
        return self.file_path.stem


def collection_name_from_stem(stem: str) -> str:
    """Chroma 3: ^[a-zA-Z0-9._-]{3,512}$"""
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in stem)
    safe = safe.strip("-._") or "board"
    if len(safe) < 3:
        safe = f"b-{safe}-x"[:32]
    return safe[:512]


def load_projects(data_dir: Path | None = None) -> list[Project]:
    base = data_dir or DATA_DIR
    projects: list[Project] = []
    for path in sorted(base.glob("*.json")):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("meta") or {}
        label = str(meta.get("метка") or path.stem)
        pid = int(meta.get("projectId") or 0)
        bid = int(meta.get("boardId") or 0)
        stem = path.stem
        projects.append(
            Project(
                file_path=path,
                collection_name=collection_name_from_stem(stem),
                label=label,
                project_id=pid,
                board_id=bid,
            )
        )
    return projects


def project_by_collection(projects: list[Project], name: str) -> Project | None:
    for p in projects:
        if p.collection_name == name:
            return p
    return None


def guess_project_from_keywords(user_text: str, projects: list[Project]) -> Project | None:
    """
    Явные упоминания проекта в тексте (метка, фамилия, домен) — приоритетнее «залипшей» сессии.
    """
    t = user_text.lower().replace("ё", "е")
    for p in projects:
        lab = (p.label or "").strip().lower().replace("ё", "е")
        if len(lab) >= 4 and lab in t:
            return p

    for p in projects:
        stem = p.file_stem.lower()
        hints: list[str] = []
        if "makukhin" in stem:
            hints.extend(("макухин", "makukhin"))
        if "avrora-kanc" in stem or ("avrora" in stem and "kanc" in stem):
            hints.extend(("аврора-канц", "avrora-kanc", "аврора канц"))
        if "avrorastore" in stem:
            hints.extend(("avrorastore",))
        if "textileavenue" in stem:
            hints.extend(("textileavenue", "textile", "текстиль"))
        if "akord" in stem and "kazan" in stem:
            hints.extend(("акорд", "akord"))
        if "giftec" in stem or "reflection" in stem:
            hints.extend(("giftec", "giftec-reflection", "reflection"))
        if "filinteriors" in stem:
            hints.extend(("filinteriors", "fil interior"))
        for h in hints:
            if len(h) >= 3 and h.lower() in t:
                return p
    return None


def _normalize_match_text(s: str) -> str:
    """Единый ключ для сравнения доменов (без пробелов, пунктуации, регистра)."""
    s = s.lower().replace("ё", "е")
    s = re.sub(r'[«»"\'`\s]', "", s)
    s = re.sub(r"[.\-/]", "", s)
    return s


def _label_variants(p: Project) -> list[str]:
    variants: list[str] = []
    lab = (p.label or "").strip()
    if lab:
        variants.append(_normalize_match_text(lab))
        bare = re.sub(r"^https?://", "", lab.lower().replace("ё", "е"))
        variants.append(_normalize_match_text(bare))
        variants.append(_normalize_match_text(re.sub(r"\.(рф|ru|com|net|org)$", "", bare, flags=re.I)))

    m = re.match(r"board-\d+-(.+)$", p.file_stem.lower())
    if m:
        slug = m.group(1)
        variants.append(_normalize_match_text(slug.replace("-", "")))
        parts = slug.split("-")
        if parts and parts[-1] in ("ru", "rf", "com", "net") and len(parts) >= 2:
            tld = "рф" if parts[-1] == "rf" else parts[-1]
            domain = ".".join(parts[:-1]) + "." + tld
            variants.append(_normalize_match_text(domain))

    out: list[str] = []
    seen: set[str] = set()
    for v in variants:
        if len(v) >= 4 and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _site_mentions_in_text(text: str) -> list[str]:
    t = text.lower().replace("ё", "е")
    mentions: list[str] = []
    for m in re.finditer(r'[«"\']([^»"\']{3,80})[»"\']', text):
        mentions.append(_normalize_match_text(m.group(1)))
    for m in re.finditer(
        r"[\w\u0400-\u04ff][\w\u0400-\u04ff.-]{2,60}\.(?:рф|ru|com|net|org)",
        t,
        flags=re.I,
    ):
        mentions.append(_normalize_match_text(m.group(0)))
    for m in re.finditer(r"[\w\u0400-\u04ff][\w\u0400-\u04ff-]{4,50}", t):
        token = m.group(0)
        if "-" in token or "." in token:
            mentions.append(_normalize_match_text(token))
    return mentions


def _score_project_mention(mention: str, variant: str) -> float:
    if not mention or not variant:
        return 0.0
    if mention == variant or mention in variant or variant in mention:
        return 1.0
    return difflib.SequenceMatcher(None, mention, variant).ratio()


def match_project_from_text(
    text: str,
    projects: list[Project],
    *,
    fuzzy_threshold: float = 0.78,
) -> Project | None:
    """
    Доска по упоминанию сайта в тексте (метка, домен, опечатки).
    Возвращает None при неоднозначном совпадении нескольких проектов.
    """
    if not (text or "").strip() or not projects:
        return None
    if len(projects) == 1:
        return projects[0]

    keyword_hit = guess_project_from_keywords(text, projects)
    if keyword_hit:
        return keyword_hit

    t_norm = _normalize_match_text(text)
    mentions = _site_mentions_in_text(text)
    if not mentions:
        mentions = [t_norm]

    scored: list[tuple[Project, float]] = []
    for p in projects:
        best = 0.0
        for variant in _label_variants(p):
            if variant in t_norm:
                best = max(best, 1.0)
                continue
            for men in mentions:
                best = max(best, _score_project_mention(men, variant))
            if len(variant) <= len(t_norm):
                for i in range(0, len(t_norm) - len(variant) + 1):
                    chunk = t_norm[i : i + len(variant)]
                    best = max(best, difflib.SequenceMatcher(None, chunk, variant).ratio())
        if best >= fuzzy_threshold:
            scored.append((p, best))

    if not scored:
        return None
    scored.sort(key=lambda x: x[1], reverse=True)
    top_p, top_score = scored[0]
    if len(scored) == 1:
        return top_p
    second_score = scored[1][1]
    if top_score >= 0.92 or top_score - second_score >= 0.06:
        return top_p
    return None
