"""Load, validate and save the historical Lotto 6/45 draw spreadsheet.

File layout (``lotto.xlsx``): header row ``count,1,2,3,4,5,6,7``; one row per
draw in descending draw order; columns 1..6 are the main balls, 7 the bonus.
"""
from __future__ import annotations

import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

MAIN_COLS = ["n1", "n2", "n3", "n4", "n5", "n6"]
FIRST_DRAW_DATE = date(2002, 12, 7)  # draw 1; draws are weekly with no gaps


def draw_date(draw_no: int) -> date:
    """Return the Saturday on which draw ``draw_no`` is held.

    Args:
        draw_no: 1-indexed draw number.

    Returns:
        The draw date.
    """
    return FIRST_DRAW_DATE + timedelta(weeks=draw_no - 1)


def validate_draws(df: pd.DataFrame) -> None:
    """Check every draw row; raise on the first invalid one.

    Args:
        df: Draws with columns draw, n1..n6, bonus, sorted by draw ascending.

    Raises:
        ValueError: ``"draw N: <reason>"`` for the first violation.
    """
    for row in df.itertuples(index=False):
        mains = [getattr(row, c) for c in MAIN_COLS]
        if not all(1 <= v <= 45 for v in (*mains, row.bonus)):
            raise ValueError(f"draw {row.draw}: ball outside 1..45")
        if len(set(mains)) != 6:
            raise ValueError(f"draw {row.draw}: duplicate main ball")
        if row.bonus in mains:
            raise ValueError(f"draw {row.draw}: bonus equals a main ball")
    expected = np.arange(1, len(df) + 1)
    bad = np.flatnonzero(df["draw"].to_numpy() != expected)
    if bad.size:
        raise ValueError(f"draw {expected[bad[0]]}: missing or out of sequence")


def load_draws(path: str | Path) -> pd.DataFrame:
    """Read and validate the draw spreadsheet.

    Args:
        path: Path to the ``.xlsx`` (or its ``.xlsx.bak`` backup).

    Returns:
        DataFrame ``draw, n1..n6 (sorted), bonus, date`` sorted by draw, index 0..T-1.

    Raises:
        ValueError: If any row is invalid (see ``validate_draws``).
    """
    raw = pd.read_excel(path, header=0, engine="openpyxl")
    raw.columns = ["draw", *MAIN_COLS, "bonus"]
    vals = raw.to_numpy(dtype=float)  # Shape: [T, 8]
    bad = np.flatnonzero(~np.isfinite(vals).all(1) | (vals != np.round(vals)).any(1))
    if bad.size:  # reject before the int cast, which would silently truncate 1.9 -> 1
        d = vals[bad[0], 0]
        where = f"draw {int(d)}" if np.isfinite(d) and d == round(d) else f"row {bad[0] + 2}"
        raise ValueError(f"{where}: blank or non-integer value")
    df = raw.astype(int).sort_values("draw").reset_index(drop=True)
    df[MAIN_COLS] = np.sort(df[MAIN_COLS].to_numpy(), axis=1)
    validate_draws(df)
    df["date"] = pd.to_datetime([draw_date(d) for d in df["draw"]])
    return df


def save_draws(df: pd.DataFrame, path: str | Path) -> None:
    """Write draws in the original file layout, atomically.

    Args:
        df: Draws with columns draw, n1..n6, bonus.
        path: Destination ``.xlsx``.
    """
    path = Path(path)
    out = df[["draw", *MAIN_COLS, "bonus"]].sort_values("draw", ascending=False)
    out.columns = ["count", 1, 2, 3, 4, 5, 6, 7]
    fd, name = tempfile.mkstemp(suffix=".tmp.xlsx", dir=path.parent)  # unique: concurrent writers never share it
    os.close(fd)
    tmp = Path(name)
    try:
        out.to_excel(tmp, index=False, engine="openpyxl")
        os.replace(tmp, path)  # fails on Windows while the file is open in Excel
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def draws_to_multihot(df: pd.DataFrame) -> np.ndarray:
    """Encode main balls as a binary matrix (bonus excluded).

    Args:
        df: Draws from ``load_draws``.

    Returns:
        float32 array. Shape: [T, 45]
    """
    m = np.zeros((len(df), 45), dtype=np.float32)  # Shape: [T, 45]
    rows = np.repeat(np.arange(len(df)), 6)
    m[rows, df[MAIN_COLS].to_numpy().ravel() - 1] = 1.0
    return m
