from __future__ import annotations

from typing import Any

import httpx

from weeek_kb.config import WEEEK_API_BASE_URL, WEEEK_API_TOKEN


class WeeekApiError(Exception):
    """Ошибка вызова Weeek Public API."""


def _headers() -> dict[str, str]:
    if not WEEEK_API_TOKEN:
        raise WeeekApiError(
            "Не задан WEEEK_API_TOKEN в .env (токен из настроек workspace → API)."
        )
    return {
        "Authorization": f"Bearer {WEEEK_API_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _request(
    method: str,
    path: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    url = f"{WEEEK_API_BASE_URL}/{path.lstrip('/')}"
    with httpx.Client(timeout=45.0) as client:
        resp = client.request(method, url, headers=_headers(), params=params, json=json_body)
    try:
        data = resp.json()
    except Exception:
        data = {}
    if resp.status_code >= 400:
        msg = data.get("message") or data.get("error") or resp.text or resp.reason_phrase
        raise WeeekApiError(f"Weeek API {resp.status_code}: {msg}")
    if isinstance(data, dict) and data.get("success") is False:
        raise WeeekApiError(str(data.get("message") or "Weeek API returned success=false"))
    return data if isinstance(data, dict) else {}


def get_members() -> list[dict[str, Any]]:
    data = _request("GET", "/ws/members")
    members = data.get("members")
    return members if isinstance(members, list) else []


def get_board_columns(board_id: int) -> list[dict[str, Any]]:
    data = _request("GET", "/tm/board-columns", params={"boardId": board_id})
    cols = data.get("boardColumns")
    return cols if isinstance(cols, list) else []


def create_task(payload: dict[str, Any]) -> dict[str, Any]:
    data = _request("POST", "/tm/tasks", json_body=payload)
    task = data.get("task")
    if isinstance(task, dict):
        return task
    return data


def member_display_name(m: dict[str, Any]) -> str:
    for key in ("name", "fullName", "displayName", "email"):
        v = m.get(key)
        if v:
            return str(v).strip()
    return str(m.get("id") or "")


def member_id(m: dict[str, Any]) -> str:
    uid = m.get("id") or m.get("userId")
    return str(uid) if uid is not None else ""


def column_by_name(columns: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    target = (name or "").strip().casefold()
    if not target:
        return None
    for c in columns:
        cn = str(c.get("name") or "").strip().casefold()
        if cn == target:
            return c
    return None
