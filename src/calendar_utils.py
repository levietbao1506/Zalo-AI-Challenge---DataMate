"""Vietnamese Calendar and Relative Date Resolution Utilities.

Provides robust functions for parsing Vietnamese date strings, calculating
relative date ranges (e.g., 'hôm qua', 'tuần rồi', 'tháng này', 'cùng kỳ năm ngoái', 'Tết')
anchored on a reference 'as_of_date', dynamically parsing <data_dir>/calendar.csv
at runtime, and handling Lunar Calendar conversions.
"""

from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

try:
    import lunarcalendar
    from lunarcalendar import Converter, Solar, Lunar
    _HAS_LUNAR = True
except ImportError:
    _HAS_LUNAR = False

logger = logging.getLogger(__name__)

# Known Vietnamese Lunar New Year fallback dates (used only if calendar.csv and lunarcalendar are unavailable)
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

_GLOBAL_CALENDAR_PATH: Optional[str] = None
_GLOBAL_CALENDAR_CACHE: Optional[Any] = None
_GLOBAL_CALENDAR_MTIME: Optional[float] = None


def set_calendar_csv_path(path: Optional[Union[str, Path]]) -> None:
    """Set global path to calendar.csv for dynamic runtime resolution."""
    global _GLOBAL_CALENDAR_PATH, _GLOBAL_CALENDAR_CACHE, _GLOBAL_CALENDAR_MTIME
    if path:
        str_p = str(Path(path).resolve())
        if str_p != _GLOBAL_CALENDAR_PATH:
            _GLOBAL_CALENDAR_PATH = str_p
            _GLOBAL_CALENDAR_CACHE = None
            _GLOBAL_CALENDAR_MTIME = None
    else:
        _GLOBAL_CALENDAR_PATH = None
        _GLOBAL_CALENDAR_CACHE = None
        _GLOBAL_CALENDAR_MTIME = None


def get_calendar_csv_path() -> Optional[Path]:
    """Find calendar.csv path from global setting, environment, or default locations."""
    if _GLOBAL_CALENDAR_PATH and Path(_GLOBAL_CALENDAR_PATH).is_file():
        return Path(_GLOBAL_CALENDAR_PATH)

    env_p = os.environ.get("CALENDAR_CSV_PATH")
    if env_p and Path(env_p).is_file():
        return Path(env_p)

    data_dir = os.environ.get("DATA_DIR", "./data")
    candidates = [
        Path(data_dir) / "calendar.csv",
        Path("/data/calendar.csv"),
        Path("./data/calendar.csv"),
        Path("calendar.csv"),
    ]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def load_calendar(calendar_path: Optional[Union[str, Path]] = None) -> Optional[Any]:
    """Dynamically load and parse calendar.csv at runtime.

    Standardizes solar dates, lunar dates, and holiday indicators.
    """
    global _GLOBAL_CALENDAR_CACHE, _GLOBAL_CALENDAR_MTIME
    if not _HAS_PANDAS:
        return None

    target_path = Path(calendar_path).resolve() if calendar_path else get_calendar_csv_path()
    if not target_path or not target_path.is_file():
        return None

    try:
        mtime = target_path.stat().st_mtime
        if _GLOBAL_CALENDAR_CACHE is not None and _GLOBAL_CALENDAR_MTIME == mtime:
            return _GLOBAL_CALENDAR_CACHE

        for enc in ("utf-8-sig", "utf-8", "latin-1"):
            try:
                df = pd.read_csv(target_path, encoding=enc)
                break
            except Exception:
                continue
        else:
            return None

        # Identify solar date column
        solar_col = None
        for col in df.columns:
            c_low = str(col).strip().lower()
            if c_low in ("date", "solar_date", "ngay", "ngay_duong", "solar", "full_date", "day"):
                solar_col = col
                break
        if not solar_col:
            for col in df.columns:
                if "date" in str(col).lower() or "ngay" in str(col).lower():
                    solar_col = col
                    break

        if not solar_col:
            return None

        df["_solar_date"] = pd.to_datetime(df[solar_col], errors="coerce").dt.date
        df = df.dropna(subset=["_solar_date"]).copy()

        # Identify lunar date column
        lunar_col = None
        for col in df.columns:
            c_low = str(col).strip().lower()
            if c_low in ("lunar_date", "ngay_am", "ngay_am_lich", "lunar"):
                lunar_col = col
                break

        if lunar_col:
            # Parse lunar date strings if present
            def _parse_lunar_str(v: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
                if pd.isna(v):
                    return None, None, None
                s = str(v).strip()
                # Patterns: YYYY-MM-DD or DD/MM/YYYY
                m_iso = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", s)
                if m_iso:
                    return int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
                m_slash = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", s)
                if m_slash:
                    return int(m_slash.group(3)), int(m_slash.group(2)), int(m_slash.group(1))
                return None, None, None

            parsed_lunar = df[lunar_col].apply(_parse_lunar_str)
            df["_lunar_year"] = parsed_lunar.apply(lambda x: x[0])
            df["_lunar_month"] = parsed_lunar.apply(lambda x: x[1])
            df["_lunar_day"] = parsed_lunar.apply(lambda x: x[2])
        else:
            # Check for separate lunar_day, lunar_month, lunar_year
            for col in df.columns:
                c_low = str(col).lower()
                if "lunar_day" in c_low or "ngay_am" in c_low:
                    df["_lunar_day"] = pd.to_numeric(df[col], errors="coerce")
                elif "lunar_month" in c_low or "thang_am" in c_low:
                    df["_lunar_month"] = pd.to_numeric(df[col], errors="coerce")
                elif "lunar_year" in c_low or "nam_am" in c_low:
                    df["_lunar_year"] = pd.to_numeric(df[col], errors="coerce")

        # Identify holiday or event column
        event_col = None
        for col in df.columns:
            c_low = str(col).strip().lower()
            if c_low in ("holiday", "holiday_name", "event", "su_kien", "ten_le", "type", "ghi_chu", "note", "is_holiday"):
                event_col = col
                break

        if event_col:
            df["_event"] = df[event_col].astype(str).str.strip().str.lower()
        else:
            df["_event"] = ""

        _GLOBAL_CALENDAR_CACHE = df
        _GLOBAL_CALENDAR_MTIME = mtime
        return df
    except Exception as e:
        logger.warning(f"Error dynamically loading calendar from {target_path}: {e}")
        return None


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
            return None

    # Pattern: YYYY-MM-DD or YYYY/MM/DD
    m_iso = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", cleaned)
    if m_iso:
        try:
            y, m, d = int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            return date(y, m, d)
        except ValueError:
            return None

    return None


def get_month_date_range(year: int, month: int) -> Tuple[date, date]:
    """Get start and end date (inclusive) for a specific month."""
    _, last_day = calendar.monthrange(year, month)
    return date(year, month, 1), date(year, month, last_day)


def get_quarter(dt: date) -> int:
    """Return quarter (1, 2, 3, or 4) for a given date."""
    return (dt.month - 1) // 3 + 1


def get_quarter_date_range(year: int, quarter: int) -> Tuple[date, date]:
    """Get start and end date (inclusive) for a specific quarter."""
    if quarter not in (1, 2, 3, 4):
        raise ValueError(f"Quarter must be 1, 2, 3, or 4; got {quarter}")
    start_month = 3 * (quarter - 1) + 1
    end_month = start_month + 2
    _, last_day = calendar.monthrange(year, end_month)
    return date(year, start_month, 1), date(year, end_month, last_day)


def get_week_date_range(
    ref_date: date,
    week_start_monday: bool = True,
) -> Tuple[date, date]:
    """Get start and end date of the week containing ref_date.

    Args:
        ref_date: Anchor date.
        week_start_monday: If True, Monday=0..Sunday=6. If False, Sunday=0..Saturday=6.
    """
    if week_start_monday:
        start_of_week = ref_date - timedelta(days=ref_date.weekday())
        end_of_week = start_of_week + timedelta(days=6)
    else:
        # Sunday as start of week
        idx = (ref_date.weekday() + 1) % 7
        start_of_week = ref_date - timedelta(days=idx)
        end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week


def get_same_period_last_year(
    start_date: Union[str, date, datetime],
    end_date: Union[str, date, datetime],
) -> Tuple[date, date]:
    """Shift date interval back by exactly 1 year (same period last year / cùng kỳ năm ngoái).

    Safely handles February 29 on leap years.
    """
    s = resolve_as_of_date(start_date)
    e = resolve_as_of_date(end_date)

    def _shift_year_back(d: date) -> date:
        target_year = d.year - 1
        max_days = calendar.monthrange(target_year, d.month)[1]
        return date(target_year, d.month, min(d.day, max_days))

    return _shift_year_back(s), _shift_year_back(e)


def solar_to_lunar(
    solar_date: Union[str, date, datetime],
    calendar_path: Optional[Union[str, Path]] = None,
) -> Tuple[int, int, int, bool]:
    """Convert a Gregorian solar date to Lunar calendar date.

    Prioritizes dynamic lookup in calendar.csv before falling back to lunarcalendar library.

    Returns:
        Tuple of (lunar_day, lunar_month, lunar_year, is_leap_month).
    """
    s_date = resolve_as_of_date(solar_date)

    # 1. Dynamic lookup in calendar.csv
    df = load_calendar(calendar_path)
    if df is not None and "_solar_date" in df.columns:
        match = df[df["_solar_date"] == s_date]
        if not match.empty:
            row = match.iloc[0]
            if "_lunar_day" in row and pd.notna(row["_lunar_day"]):
                ld = int(row["_lunar_day"])
                lm = int(row["_lunar_month"]) if pd.notna(row.get("_lunar_month")) else 1
                ly = int(row["_lunar_year"]) if pd.notna(row.get("_lunar_year")) else s_date.year
                return ld, lm, ly, False

    # 2. Lunarcalendar fallback
    if _HAS_LUNAR:
        try:
            s = Solar(s_date.year, s_date.month, s_date.day)
            l = Converter.Solar2Lunar(s)
            return l.day, l.month, l.year, l.isleap
        except Exception:
            pass

    return s_date.day, s_date.month, s_date.year, False


def lunar_to_solar(
    lunar_year: int,
    lunar_month: int,
    lunar_day: int,
    is_leap: bool = False,
    calendar_path: Optional[Union[str, Path]] = None,
) -> Optional[date]:
    """Convert a Lunar calendar date to Gregorian solar date.

    Prioritizes dynamic lookup in calendar.csv before falling back to lunarcalendar library.
    """
    # 1. Dynamic lookup in calendar.csv
    df = load_calendar(calendar_path)
    if df is not None and "_solar_date" in df.columns and "_lunar_day" in df.columns:
        match = df[
            (df["_lunar_year"] == lunar_year)
            & (df["_lunar_month"] == lunar_month)
            & (df["_lunar_day"] == lunar_day)
        ]
        if not match.empty:
            return match.iloc[0]["_solar_date"]

    # 2. Lunarcalendar library fallback
    if _HAS_LUNAR:
        try:
            l = Lunar(lunar_year, lunar_month, lunar_day, isleap=is_leap)
            s = Converter.Lunar2Solar(l)
            return date(s.year, s.month, s.day)
        except Exception:
            pass

    return None


def get_tet_date(
    year: int,
    calendar_path: Optional[Union[str, Path]] = None,
) -> Optional[date]:
    """Get Solar date of Vietnamese Lunar New Year (Tết Mùng 1) for a given year.

    Dynamically resolves from calendar.csv, lunarcalendar library, or fallback map.
    """
    # 1. Dynamic check in calendar.csv (Lunar 1/1)
    solar_mung_1 = lunar_to_solar(lunar_year=year, lunar_month=1, lunar_day=1, calendar_path=calendar_path)
    if solar_mung_1 is not None:
        return solar_mung_1

    # 2. Dynamic check in calendar.csv for event mentioning Mùng 1 Tết
    df = load_calendar(calendar_path)
    if df is not None and "_event" in df.columns:
        tet_rows = df[
            df["_solar_date"].apply(lambda d: d.year == year and d.month in (1, 2))
            & df["_event"].str.contains(r"(mùng 1|mung 1|tết nguyên đán|tet nguyen dan)", regex=True)
        ]
        if not tet_rows.empty:
            return tet_rows.iloc[0]["_solar_date"]

    # 3. Lunarcalendar library
    if _HAS_LUNAR:
        try:
            l = Lunar(year, 1, 1, isleap=False)
            s = Converter.Lunar2Solar(l)
            return date(s.year, s.month, s.day)
        except Exception:
            pass

    # 4. Fallback map
    return VIETNAMESE_TET_SOLAR_MAP.get(year)


def get_tet_window(
    year: Optional[int] = None,
    as_of: Optional[Union[str, date, datetime]] = None,
    calendar_path: Optional[Union[str, Path]] = None,
) -> Tuple[date, date]:
    """Compute the Tết holiday window (start_date, end_date inclusive).

    Dynamically extracts Tết dates from calendar.csv if available; otherwise computes
    the standard holiday period (from 29 tháng Chạp to Mùng 5 Tết).
    """
    ref_date = resolve_as_of_date(as_of)
    target_year = year if year is not None else ref_date.year

    # 1. Dynamic lookup in calendar.csv for event records containing 'tết' or 'tet'
    df = load_calendar(calendar_path)
    if df is not None and "_event" in df.columns:
        # Match dates in Jan/Feb of target year where event mentions Tết
        mask = (
            df["_solar_date"].apply(lambda d: d.year == target_year and d.month in (1, 2))
            & df["_event"].str.contains(r"\bt[eế]t\b|nguy[eê]n\s*đ[aá]n", regex=True)
        )
        matching = df[mask]
        if not matching.empty:
            return matching["_solar_date"].min(), matching["_solar_date"].max()

    # 2. Derive window from Tết Mùng 1 (29 tháng Chạp -> Mùng 5 Tết)
    tet_mung_1 = get_tet_date(target_year, calendar_path=calendar_path)
    if tet_mung_1:
        start_tet = tet_mung_1 - timedelta(days=2)
        end_tet = tet_mung_1 + timedelta(days=5)
        return start_tet, end_tet

    # 3. Fallback
    return date(target_year, 1, 25), date(target_year, 2, 5)


def get_holiday_range(
    holiday_name: str,
    year: Optional[int] = None,
    as_of: Optional[Union[str, date, datetime]] = None,
    calendar_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[date, date]]:
    """Dynamically find date range of a named holiday from calendar.csv.

    Args:
        holiday_name: Name of holiday (e.g. '30/4', '1/5', 'quốc khánh', 'giỗ tổ', 'tết').
        year: Full target year.
        as_of: Reference date anchor.
    """
    ref_date = resolve_as_of_date(as_of)
    target_year = year if year is not None else ref_date.year
    df = load_calendar(calendar_path)
    if df is None or "_event" not in df.columns:
        return None

    norm_h = holiday_name.strip().lower()
    matching = df[
        df["_solar_date"].apply(lambda d: d.year == target_year)
        & df["_event"].str.contains(re.escape(norm_h), regex=True)
    ]
    if not matching.empty:
        return matching["_solar_date"].min(), matching["_solar_date"].max()
    return None


def resolve_vietnamese_relative_date(
    expression: str,
    as_of: Optional[Union[str, date, datetime]] = None,
    calendar_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[date, date]]:
    """Resolve a Vietnamese relative date phrase against a reference as_of_date.

    Phrases supported:
        - 'hôm nay', 'ngày hiện tại' -> (as_of, as_of)
        - 'hôm qua' -> (as_of - 1d, as_of - 1d)
        - 'hôm kia' -> (as_of - 2d, as_of - 2d)
        - 'ngày mai' -> (as_of + 1d, as_of + 1d)
        - 'ngày kia', 'ngày mốt' -> (as_of + 2d, as_of + 2d)
        - 'tuần này' -> Monday to Sunday of the week containing as_of
        - 'tuần trước', 'tuần rồi', 'tuần vừa rồi', 'tuần qua' -> previous week
        - 'tuần sau', 'tuần tới' -> next week
        - 'tháng này', 'trong tháng này', 'tháng hiện tại' -> 1st to last day of as_of's month
        - 'tháng trước', 'tháng rồi', 'tháng vừa rồi', 'tháng vừa qua' -> previous month
        - 'tháng sau', 'tháng tới' -> next month
        - 'quý này', 'trong quý này' -> start to end of current quarter
        - 'quý trước', 'quý rồi', 'quý vừa rồi' -> start to end of previous quarter
        - 'quý sau', 'quý tới' -> next quarter
        - 'cùng kỳ năm ngoái', 'cùng kỳ năm trước' -> same month or period in previous year
        - 'Tết', 'dịp Tết', 'Tết Nguyên Đán', 'nghỉ Tết', 'mùa Tết', 'vụ Tết' -> Tết window
        - 'năm nay' -> Jan 1 to Dec 31 of as_of's year
        - 'năm ngoái', 'năm trước' -> Jan 1 to Dec 31 of previous year
        - 'năm kia' -> Jan 1 to Dec 31 of (year - 2)
        - 'năm sau', 'năm tới' -> next year
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
    start_of_current_week, end_of_current_week = get_week_date_range(ref_date, week_start_monday=True)

    if expr in ("tuần này", "trong tuần này", "tuần hiện tại"):
        return start_of_current_week, end_of_current_week
    if expr in ("tuần trước", "tuần rồi", "tuần vừa rồi", "tuần vừa qua", "tuần qua"):
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
    if expr in ("tháng này", "trong tháng này", "tháng hiện tại"):
        return get_month_date_range(ref_date.year, ref_date.month)
    if expr in ("tháng trước", "tháng rồi", "tháng vừa rồi", "tháng vừa qua", "tháng qua"):
        if ref_date.month == 1:
            return get_month_date_range(ref_date.year - 1, 12)
        return get_month_date_range(ref_date.year, ref_date.month - 1)
    if expr in ("tháng sau", "tháng tới"):
        if ref_date.month == 12:
            return get_month_date_range(ref_date.year + 1, 1)
        return get_month_date_range(ref_date.year, ref_date.month + 1)

    # 5. Tết window / holiday terms
    if any(k in expr for k in ("tết nguyên đán", "dịp tết", "nghỉ tết", "kỳ nghỉ tết", "mùa tết", "vụ tết", "tết")):
        # Check if an explicit year is mentioned (e.g. "Tết 2024")
        m_tet_year = re.search(r"\b(\d{4})\b", expr)
        t_year = int(m_tet_year.group(1)) if m_tet_year else ref_date.year
        return get_tet_window(year=t_year, as_of=ref_date, calendar_path=calendar_path)

    # 6. Same period last year (cùng kỳ năm ngoái)
    if expr in ("cùng kỳ năm ngoái", "cùng kỳ năm trước", "so với cùng kỳ năm ngoái"):
        # Anchored on as_of month
        return get_month_date_range(ref_date.year - 1, ref_date.month)

    # 7. Explicit quarter & year (e.g. "quý 2 năm 2024" or "quý 2/2024")
    m_quarter_year = re.search(r"quý\s+([1-4])(?:\s+năm|\s*/\s*|\s*-)(\d{4})", expr)
    if m_quarter_year:
        q, y = int(m_quarter_year.group(1)), int(m_quarter_year.group(2))
        return get_quarter_date_range(y, q)

    # 8. Quarter-level relative terms
    current_q = get_quarter(ref_date)
    if expr in ("quý này", "trong quý này", "quý hiện tại"):
        return get_quarter_date_range(ref_date.year, current_q)
    if expr in ("quý trước", "quý rồi", "quý vừa rồi", "quý vừa qua"):
        if current_q == 1:
            return get_quarter_date_range(ref_date.year - 1, 4)
        return get_quarter_date_range(ref_date.year, current_q - 1)
    if expr in ("quý sau", "quý tới"):
        if current_q == 4:
            return get_quarter_date_range(ref_date.year + 1, 1)
        return get_quarter_date_range(ref_date.year, current_q + 1)
    if expr in ("quý cùng kỳ năm ngoái", "quý cùng kỳ năm trước"):
        return get_quarter_date_range(ref_date.year - 1, current_q)

    # 9. Explicit year (e.g. "năm 2023")
    m_year = re.search(r"\bnăm\s+(\d{4})\b", expr)
    if m_year:
        y = int(m_year.group(1))
        return date(y, 1, 1), date(y, 12, 31)

    # 10. Year-level relative terms
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
