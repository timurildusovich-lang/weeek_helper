"""Нормализация поисковых запросов: транслит и варианты написания для лучшего совпадения с индексом."""

from __future__ import annotations

import re

# Порядок: длинные совпадения раньше коротких при необходимости
_TRANSLIT_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("1С", "1C"),
    ("1с", "1c"),
    ("Мах", "Max"),
    ("мах", "max"),
)


def normalize_search_query(text: str) -> str:
    """Подмена типичных русских написаний на латиницу в духе 1С↔1C (для эмбеддингового поиска)."""
    if not text:
        return text
    s = text
    for src, dst in _TRANSLIT_REPLACEMENTS:
        s = s.replace(src, dst)
    # Отдельно слово MAX → Max (мессенджер), не трогаем MAX внутри других слов
    s = re.sub(r"\bMAX\b", "Max", s)
    return s
