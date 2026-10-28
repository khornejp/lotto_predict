"""Cyclical calendar encodings of the target draw date (known before the draw)."""
from __future__ import annotations

import calendar
import math
from datetime import date

import numpy as np


def calendar_features(d: date) -> np.ndarray:
    """Sine/cosine encodings of position in month and in year.

    Args:
        d: Draw date.

    Returns:
        [sin, cos of (day-1)/days_in_month, sin, cos of (doy-1)/365.25] * 2*pi. Shape: [4]
    """
    month_phase = 2 * math.pi * (d.day - 1) / calendar.monthrange(d.year, d.month)[1]
    year_phase = 2 * math.pi * (d.timetuple().tm_yday - 1) / 365.25
    return np.array(
        [math.sin(month_phase), math.cos(month_phase), math.sin(year_phase), math.cos(year_phase)],
        dtype=np.float32,
    )
