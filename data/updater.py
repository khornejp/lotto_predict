"""Append newly held draws to the spreadsheet: automatic fetch or manual entry."""
from __future__ import annotations

import json
import shutil
import urllib.request
import warnings
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from data.loader import MAIN_COLS, load_draws, save_draws, validate_draws

# Verified 2026-10-28; the older common.do?method=getLottoNumber endpoint now returns HTML.
DEFAULT_API = "https://www.dhlottery.co.kr/lt645/selectPstLt645Info.do?srchLtEpsd={n}"


class UpdateError(RuntimeError):
    """The draw source could not be reached or returned something unexpected."""


def fetch_draw(draw_no: int, url: str = DEFAULT_API, timeout: float = 10.0) -> dict[str, int] | None:
    """Fetch one draw's winning numbers.

    Args:
        draw_no: 1-indexed draw number.
        url: Endpoint template with ``{n}``.
        timeout: Seconds.

    Returns:
        ``{"draw", "n1".."n6", "bonus"}``, or None if the draw has not been held yet.

    Raises:
        UpdateError: Network failure, non-JSON body, missing keys, or a mismatched draw number.
    """
    try:
        with urllib.request.urlopen(url.format(n=draw_no), timeout=timeout) as resp:
            items = json.loads(resp.read())["data"]["list"]
        if not items:
            return None
        it = items[0]
        if it["ltEpsd"] != draw_no:
            raise UpdateError(f"asked for draw {draw_no}, got {it['ltEpsd']}")
        return {"draw": draw_no, **{f"n{i}": int(it[f"tm{i}WnNo"]) for i in range(1, 7)}, "bonus": int(it["bnsWnNo"])}
    except UpdateError:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as e:
        raise UpdateError(f"draw {draw_no}: {e!r}") from e


def append_draws(path: str | Path, rows: list[dict[str, int]]) -> None:
    """Validate, back up to ``.xlsx.bak``, then write the file with ``rows`` appended.

    Args:
        path: Spreadsheet path.
        rows: New draws ``{"draw", "n1".."n6", "bonus"}``.

    Raises:
        ValueError: If the result would be invalid; the file is left untouched.
    """
    path = Path(path)
    df = _with_rows(load_draws(path).drop(columns="date"), rows)
    validate_draws(df)
    shutil.copy2(path, path.with_suffix(".xlsx.bak"))
    save_draws(df, path)


def _with_rows(df: pd.DataFrame, rows: list[dict[str, int]]) -> pd.DataFrame:
    """Return ``df`` with ``rows`` appended (main balls sorted)."""
    new = pd.DataFrame(rows, columns=["draw", *MAIN_COLS, "bonus"]).astype(int)
    new[MAIN_COLS] = np.sort(new[MAIN_COLS].to_numpy(), axis=1)
    return pd.concat([df, new], ignore_index=True)


def update_draws(
    path: str | Path, fetch: Callable[[int], dict[str, int] | None] = fetch_draw, max_new: int = 20
) -> int:
    """Fetch draws after the last one on file and append them.

    Never raises for source or file problems: a failed fetch, an invalid fetched row,
    or a locked file only warns, and valid draws fetched before the problem are kept.

    Args:
        path: Spreadsheet path.
        fetch: Draw source (``fetch_draw`` by default).
        max_new: Upper bound on draws fetched per call.

    Returns:
        Number of draws added.
    """
    base = load_draws(path).drop(columns="date")
    n = int(base["draw"].iloc[-1]) + 1
    rows: list[dict[str, int]] = []
    for draw_no in range(n, n + max_new):
        try:
            r = fetch(draw_no)
            if r is not None:
                validate_draws(_with_rows(base, [*rows, r]))
        except (UpdateError, ValueError) as e:
            warnings.warn(f"auto update failed at draw {draw_no}: {e}")
            break
        if r is None:
            break
        rows.append(r)
    if not rows:
        return 0
    try:
        append_draws(path, rows)
    except OSError as e:
        warnings.warn(f"auto update could not write {path}: {e}")
        return 0
    return len(rows)
