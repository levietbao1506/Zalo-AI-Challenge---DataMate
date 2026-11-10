"""Vietnamese Calendar and Relative Date Resolution Utilities.

Provides robust functions for parsing Vietnamese date strings, calculating
relative date ranges (e.g., 'hôm qua', 'tuần trước', 'tháng này', 'quý 2 năm ngoái')
anchored on a reference 'as_of_date', and handling Lunar Calendar conversions.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
import re
from typing import Optional, Tuple, Union

try:
    import lunarcalendar
    from lunarcalendar import Converter, Solar, Lunar
    _HAS_LUNAR = True
except ImportError:
    _HAS_LUNAR = False

# Known Vietnamese Lunar New Year (Tết Nguyên Đán - Mùng 1 Tết) dates for common evaluation years
VIETNAMESE_TET_SOLAR_MAP: dict[int, date] = {
    2020: date(2020, 1, 25),
    2021: date(2021, 2, 12),
    2022: date(2022, 2, 1),
    2023: date(2023, 1, 22),
    2024: date(2024, 2, 10),
    2025: date(2025, 1, 29),
    2026: date(2026, 2, 17),
    2027: date(2027, 2, 6),
    2028: date(2028, 1, 26),
    2029: date(2029, 2, 13),
    2030: date(2030, 2, 3),
}


def resolve_as_of_date(as_of: Optional[Union[str, date, datetime]] = None) -> date:
    """Normalize input reference date into a datetime.date object.

    Args:
        as_of: Reference date as string, date, or datetime. If None, defaults to current date.

    Returns:
        Normalized datetime.date instance.
    """
    if as_of is None:
        return date.today()
    if isinstance(as_of, datetime):
        return as_of.date()
    if isinstance(as_of, date):
        return as_of
    if isinstance(as_of, str):
        parsed = parse_vietnamese_date(as_of.strip())
        if parsed is not None:
            return parsed
        # Fallback to standard ISO
        try:
            return datetime.fromisoformat(as_of.strip()).date()
        except ValueError:
            pass
    return date.today()


def parse_vietnamese_date(text: str) -> Optional[date]:
    """Parse common Vietnamese and ISO date formats.

    Supported patterns:
        - "2024-05-15"
        - "15/05/2024", "15-05-2024"
        - "ngày 15 tháng 5 năm 2024"
        - "15/5/24"

    Args:
        text: Date string to parse.

    Returns:
        Parsed datetime.date or None if unrecognized.
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip().lower()

    # Pattern: ngày DD tháng MM năm YYYY
    m_vn = re.search(r"(?:ngày\s+)?(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})", cleaned)
    if m_vn:
        try:
            d, m, y = int(m_vn.group(1)), int(m_vn.group(2)), int(m_vn.group(3))
            return date(y, m, d)
        except ValueError:
            return None

    # Pattern: DD/MM/YYYY or DD-MM-YYYY
    m_slash = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4})\b", cleaned)
    if m_slash:
        try:
            d, m, y = int(m_slash.group(1)), int(m_slash.group(2)), int(m_slash.group(3))
            return date(y, m, d)
        except ValueError:
            pass

    # Pattern: YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", cleaned)
    if m_iso:
        try:
            y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            return date(y, m, d)
        except ValueError:
            pass

    return None


def get_month_date_range(year: int, month: int) -> Tuple[date, date]:
    """Get the start and end date (inclusive) for a specific month.

    Args:
        year: Full year (e.g., 2024).
        month: Month (1 to 12).

    Returns:
        Tuple of (start_date, end_date).
    """
    _, last_day = calendar.monthrange(year, month)
    return date(year, month, 1), date(year, month, last_day)


def get_quarter(dt: date) -> int:
    """Return the quarter (1, 2, 3, or 4) for a given date."""
    return (dt.month - 1) // 3 + 1


def get_quarter_date_range(year: int, quarter: int) -> Tuple[date, date]:
    """Get the start and end date (inclusive) for a specific quarter.

    Args:
        year: Year (e.g., 2024).
        quarter: Quarter index (1 to 4).

    Returns:
        Tuple of (start_date, end_date).
    """
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"Quarter must be 1, 2, 3, or 4; got {quarter}")
    start_month = 3 * (quarter - 1) + 1
    end_month = start_month + 2
    _, last_day = calendar.monthrange(year, end_month)
    return date(year, start_month, 1), date(year, end_month, last_day)


def resolve_vietnamese_relative_date(
    expression: str,
    as_of: Optional[Union[str, date, datetime]] = None,
) -> Optional[Tuple[date, date]]:
    """Resolve a Vietnamese relative date phrase against a reference as_of_date.

    Phrases supported:
        - 'hôm nay', 'ngày hiện tại' -> (as_of, as_of)
        - 'hôm qua' -> (as_of - 1d, as_of - 1d)
        - 'hôm kia' -> (as_of - 2d, as_of - 2d)
        - 'ngày mai' -> (as_of + 1d, as_of + 1d)
        - 'ngày kia', 'ngày mốt' -> (as_of + 2d, as_of + 2d)
        - 'tuần này' -> Monday to Sunday of the week containing as_of
        - 'tuần trước' -> Monday to Sunday of the previous week
        - 'tuần sau', 'tuần tới' -> Monday to Sunday of next week
        - 'tháng này' -> 1st to last day of as_of's month
        - 'tháng trước' -> 1st to last day of previous month
        - 'tháng sau', 'tháng tới' -> 1st to last day of next month
        - 'quý này' -> start to end of current quarter
        - 'quý trước' -> start to end of previous quarter
        - 'quý sau' -> start to end of next quarter
        - 'quý X năm YYYY' -> start to end of quarter X in year YYYY
        - 'tháng M năm YYYY' -> start to end of month M in year YYYY
        - 'năm nay' -> Jan 1 to Dec 31 of as_of's year
        - 'năm ngoái', 'năm trước' -> Jan 1 to Dec 31 of previous year
        - 'năm kia' -> Jan 1 to Dec 31 of (year - 2)
        - 'năm sau', 'năm tới' -> Jan 1 to Dec 31 of next year

    Args:
        expression: Natural language phrase in Vietnamese.
        as_of: Reference anchor date.

    Returns:
        Tuple of (start_date, end_date) inclusive, or None if unrecognized.
    """
    ref_date = resolve_as_of_date(as_of)
    expr = expression.strip().lower()

    # 1. Day-level relative terms
    if expr in ("hôm nay", "ngày hiện tại", "hiện tại"):
        return ref_date, ref_date
    if expr in ("hôm qua", "ngày hôm qua"):
        d = ref_date - timedelta(days=1)
        return d, d
    if expr in ("hôm kia", "ngày hôm kia"):
        d = ref_date - timedelta(days=2)
        return d, d
    if expr in ("ngày mai", "hôm sau"):
        d = ref_date + timedelta(days=1)
        return d, d
    if expr in ("ngày kia", "ngày mốt"):
        d = ref_date + timedelta(days=2)
        return d, d

    # 2. Week-level relative terms (Monday = 0, Sunday = 6)
    start_of_current_week = ref_date - timedelta(days=ref_date.weekday())
    end_of_current_week = start_of_current_week + timedelta(days=6)

    if expr in ("tuần này", "trong tuần này"):
        return start_of_current_week, end_of_current_week
    if expr in ("tuần trước", "tuần vừa rồi", "tuần vừa qua"):
        start_prev = start_of_current_week - timedelta(days=7)
        return start_prev, start_prev + timedelta(days=6)
    if expr in ("tuần sau", "tuần tới"):
        start_next = start_of_current_week + timedelta(days=7)
        return start_next, start_next + timedelta(days=6)

    # 3. Explicit month & year (e.g. "tháng 5 năm 2024" or "tháng 05/2024")
    m_month_year = re.search(r"tháng\s+(\d{1,2})(?:\s+năm|\s*/\s*|\s*-)(\d{4})", expr)
    if m_month_year:
        m, y = int(m_month_year.group(1)), int(m_month_year.group(2))
        if 1 <= m <= 12:
            return get_month_date_range(y, m)

    # 4. Month-level relative terms
    if expr in ("tháng này", "trong tháng này"):
        return get_month_date_range(ref_date.year, ref_date.month)
    if expr in ("tháng trước", "tháng vừa rồi", "tháng vừa qua"):
        if ref_date.month == 1:
            return get_month_date_range(ref_date.year - 1, 12)
        return get_month_date_range(ref_date.year, ref_date.month - 1)
    if expr in ("tháng sau", "tháng tới"):
        if ref_date.month == 12:
            return get_month_date_range(ref_date.year + 1, 1)
        return get_month_date_range(ref_date.year, ref_date.month + 1)

    # 5. Explicit quarter & year (e.g. "quý 2 năm 2024" or "quý 2/2024")
    m_quarter_year = re.search(r"quý\s+([1-4])(?:\s+năm|\s*/\s*|\s*-)(\d{4})", expr)
    if m_quarter_year:
        q, y = int(m_quarter_year.group(1)), int(m_quarter_year.group(2))
        return get_quarter_date_range(y, q)

    # 6. Quarter-level relative terms
    current_q = get_quarter(ref_date)
    if expr in ("quý này", "trong quý này"):
        return get_quarter_date_range(ref_date.year, current_q)
    if expr in ("quý trước", "quý vừa rồi", "quý vừa qua"):
        if current_q == 1:
            return get_quarter_date_range(ref_date.year - 1, 4)
        return get_quarter_date_range(ref_date.year, current_q - 1)
    if expr in ("quý sau", "quý tới"):
        if current_q == 4:
            return get_quarter_date_range(ref_date.year + 1, 1)
        return get_quarter_date_range(ref_date.year, current_q + 1)

    # 7. Explicit year (e.g. "năm 2023")
    m_year = re.search(r"\bnăm\s+(\d{4})\b", expr)
    if m_year:
        y = int(m_year.group(1))
        return date(y, 1, 1), date(y, 12, 31)

    # 8. Year-level relative terms
    if expr in ("năm nay", "trong năm nay"):
        return date(ref_date.year, 1, 1), date(ref_date.year, 12, 31)
    if expr in ("năm ngoái", "năm trước", "năm vừa rồi"):
        prev_y = ref_date.year - 1
        return date(prev_y, 1, 1), date(prev_y, 12, 31)
    if expr in ("năm kia",):
        two_years_ago = ref_date.year - 2
        return date(two_years_ago, 1, 1), date(two_years_ago, 12, 31)
    if expr in ("năm sau", "năm tới"):
        next_y = ref_date.year + 1
        return date(next_y, 1, 1), date(next_y, 12, 31)

    return None


def solar_to_lunar(solar_date: date) -> Tuple[int, int, int, bool]:
    """Convert a Gregorian solar date to Lunar calendar date.

    Args:
        solar_date: Solar datetime.date.

    Returns:
        Tuple of (lunar_day, lunar_month, lunar_year, is_leap_month).
    """
    if _HAS_LUNAR:
        s = Solar(solar_date.year, solar_date.month, solar_date.day)
        l = Converter.Solar2Lunar(s)
        return l.day, l.month, l.year, l.isleap
    # Fallback placeholder if lunarcalendar package is not present
    return solar_date.day, solar_date.month, solar_date.year, False


def get_tet_date(year: int) -> Optional[date]:
    """Get the Solar date for Vietnamese Lunar New Year (Tết Mùng 1) for a given year.

    Args:
        year: Target year.

    Returns:
        Solar date of Tết Nguyên Đán, or None if unavailable.
    """
    if year in VIETNAMESE_TET_SOLAR_MAP:
        return VIETNAMESE_TET_SOLAR_MAP[year]

    if _HAS_LUNAR:
        try:
            l = Lunar(year, 1, 1, isleap=False)
            s = Converter.Lunar2Solar(l)
            return date(s.year, s.month, s.day)
        except Exception:
            return None
    return None
