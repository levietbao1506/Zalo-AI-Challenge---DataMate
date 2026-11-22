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
import unicodedata
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

_GLOBAL_CALENDAR_PATH: Optional[str] = None
_GLOBAL_CALENDAR_CACHE: Optional[Any] = None
_GLOBAL_CALENDAR_MTIME: Optional[float] = None
_GLOBAL_CALENDAR_LOADED_PATH: Optional[str] = None


def set_calendar_csv_path(path: Optional[Union[str, Path]]) -> None:
    """Set global path to calendar.csv for dynamic runtime resolution."""
    global _GLOBAL_CALENDAR_PATH, _GLOBAL_CALENDAR_CACHE, _GLOBAL_CALENDAR_MTIME, _GLOBAL_CALENDAR_LOADED_PATH
    if path:
        str_p = str(Path(path).resolve())
        if str_p != _GLOBAL_CALENDAR_PATH:
            _GLOBAL_CALENDAR_PATH = str_p
            _GLOBAL_CALENDAR_CACHE = None
            _GLOBAL_CALENDAR_MTIME = None
            _GLOBAL_CALENDAR_LOADED_PATH = None
    else:
        _GLOBAL_CALENDAR_PATH = None
        _GLOBAL_CALENDAR_CACHE = None
        _GLOBAL_CALENDAR_MTIME = None
        _GLOBAL_CALENDAR_LOADED_PATH = None


def get_calendar_csv_path() -> Optional[Path]:
    """Find calendar.csv path from global setting, environment, or default locations."""
    if _GLOBAL_CALENDAR_PATH and Path(_GLOBAL_CALENDAR_PATH).is_file():
        return Path(_GLOBAL_CALENDAR_PATH)

    env_p = os.environ.get("CALENDAR_CSV_PATH")
    if env_p and Path(env_p).is_file():
        return Path(env_p)

    data_dir = os.environ.get("DATA_DIR", "/data")
    candidates = [Path(data_dir) / "calendar.csv"]
    for c in candidates:
        if c.is_file():
            return c.resolve()
    return None


def load_normalized_calendar_df(calendar_csv_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Read calendar CSV, normalize date columns, and return a clean DataFrame."""
    if not _HAS_PANDAS:
        return pd.DataFrame()

    target_path = Path(calendar_csv_path).resolve() if calendar_csv_path else get_calendar_csv_path()
    empty_cols = [
        "solar_date", "solar_year", "solar_month", "solar_day",
        "iso_weekday", "weekday_vi", "lunar_year", "lunar_month", "lunar_day", "is_leap_month"
    ]
    if not target_path or not target_path.is_file():
        return pd.DataFrame(columns=empty_cols)

    df = None
    for enc in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            with target_path.open("r", encoding=enc) as handle:
                first_line = handle.readline()
            delimiter = max((",", ";", "\t"), key=first_line.count)
            df = pd.read_csv(target_path, encoding=enc, sep=delimiter)
            break
        except Exception:
            continue

    if df is None or df.empty:
        return pd.DataFrame(columns=empty_cols)

    solar_col = None
    for col in df.columns:
        c_low = str(col).strip().lower()
        if c_low in ("solar_date", "date", "ngay", "ngay_duong", "duong_lich", "solar", "full_date", "day", "solar_date_iso"):
            solar_col = col
            break
    if not solar_col:
        for col in df.columns:
            c_low = str(col).strip().lower()
            if ("date" in c_low or "ngay" in c_low) and "am" not in c_low and "lunar" not in c_low:
                solar_col = col
                break
    if not solar_col:
        solar_col = df.columns[0]

    day_first = df[solar_col].astype(str).str.match(r"^\d{1,2}/\d{1,2}/\d{4}").mean() > 0.5
    parsed_dates = pd.to_datetime(df[solar_col], errors="coerce", dayfirst=day_first).dt.date
    valid_mask = parsed_dates.notna()
    df = df[valid_mask].copy()
    parsed_dates = parsed_dates[valid_mask]

    vi_weekday_map = {
        1: "Thứ 2",
        2: "Thứ 3",
        3: "Thứ 4",
        4: "Thứ 5",
        5: "Thứ 6",
        6: "Thứ 7",
        7: "Chủ nhật",
    }

    df["solar_date"] = parsed_dates
    df["solar_year"] = parsed_dates.map(lambda d: d.year).astype(int)
    df["solar_month"] = parsed_dates.map(lambda d: d.month).astype(int)
    df["solar_day"] = parsed_dates.map(lambda d: d.day).astype(int)
    df["iso_weekday"] = parsed_dates.map(lambda d: d.isoweekday()).astype(int)
    df["weekday_vi"] = df["iso_weekday"].map(vi_weekday_map)

    lunar_date_col = None
    for col in df.columns:
        c_low = str(col).strip().lower()
        if c_low in ("lunar_date", "ngay_am", "ngay_am_lich", "lunar", "am_lich"):
            lunar_date_col = col
            break

    l_year_col = next((c for c in df.columns if str(c).strip().lower() in ("lunar_year", "nam_am", "am_nam")), None)
    l_month_col = next((c for c in df.columns if str(c).strip().lower() in ("lunar_month", "thang_am", "am_thang")), None)
    l_day_col = next((c for c in df.columns if str(c).strip().lower() in ("lunar_day", "ngay_am_so", "am_ngay")), None)

    if l_year_col and l_month_col and l_day_col:
        df["lunar_year"] = pd.to_numeric(df[l_year_col], errors="coerce")
        df["lunar_month"] = pd.to_numeric(df[l_month_col], errors="coerce")
        df["lunar_day"] = pd.to_numeric(df[l_day_col], errors="coerce")
    elif lunar_date_col:
        def _parse_lunar_parts(val: Any) -> Tuple[Optional[int], Optional[int], Optional[int]]:
            if pd.isna(val):
                return None, None, None
            s = str(val).strip()
            m_iso = re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", s)
            if m_iso:
                return int(m_iso.group(1)), int(m_iso.group(2)), int(m_iso.group(3))
            m_slash = re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", s)
            if m_slash:
                return int(m_slash.group(3)), int(m_slash.group(2)), int(m_slash.group(1))
            return None, None, None

        parsed_lunar = df[lunar_date_col].map(_parse_lunar_parts)
        df["lunar_year"] = pd.to_numeric(parsed_lunar.map(lambda p: p[0]), errors="coerce").astype("Int64")
        df["lunar_month"] = pd.to_numeric(parsed_lunar.map(lambda p: p[1]), errors="coerce").astype("Int64")
        df["lunar_day"] = pd.to_numeric(parsed_lunar.map(lambda p: p[2]), errors="coerce").astype("Int64")
    else:
        df["lunar_year"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
        df["lunar_month"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")
        df["lunar_day"] = pd.Series([pd.NA] * len(df), index=df.index, dtype="Int64")

    leap_col = next(
        (c for c in df.columns if str(c).strip().lower() in ("is_leap_month", "is_leap", "nhuan", "thang_nhuan", "leap_month")),
        None,
    )
    if leap_col:
        df["is_leap_month"] = df[leap_col].astype(str).str.strip().str.lower().isin(("true", "1", "yes", "co", "có", "t"))
    else:
        df["is_leap_month"] = False

    df["_solar_date"] = df["solar_date"]
    df["_lunar_day"] = df["lunar_day"]
    df["_lunar_month"] = df["lunar_month"]
    df["_lunar_year"] = df["lunar_year"]

    event_col = next(
        (c for c in df.columns if str(c).strip().lower() in ("holiday", "holiday_name", "event", "su_kien", "ten_le", "type", "ghi_chu", "note", "is_holiday")),
        None,
    )
    if event_col:
        df["_event"] = df[event_col].astype(str).str.strip().str.lower()
    else:
        df["_event"] = ""

    return df


def load_calendar(calendar_path: Optional[Union[str, Path]] = None) -> Optional[Any]:
    """Dynamically load and parse calendar.csv at runtime."""
    global _GLOBAL_CALENDAR_CACHE, _GLOBAL_CALENDAR_MTIME, _GLOBAL_CALENDAR_LOADED_PATH
    if not _HAS_PANDAS:
        return None

    target_path = Path(calendar_path).resolve() if calendar_path else get_calendar_csv_path()
    if not target_path or not target_path.is_file():
        return None

    try:
        mtime = target_path.stat().st_mtime
        if (_GLOBAL_CALENDAR_CACHE is not None and _GLOBAL_CALENDAR_MTIME == mtime
                and _GLOBAL_CALENDAR_LOADED_PATH == str(target_path)):
            return _GLOBAL_CALENDAR_CACHE

        df = load_normalized_calendar_df(target_path)
        if df is None or df.empty:
            return None

        _GLOBAL_CALENDAR_CACHE = df
        _GLOBAL_CALENDAR_MTIME = mtime
        _GLOBAL_CALENDAR_LOADED_PATH = str(target_path)
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

    return None


def get_tet_window(
    year: Optional[int] = None,
    as_of: Optional[Union[str, date, datetime]] = None,
    calendar_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[date, date]]:
    """Return the split calendar's Tết window, or None when it is undefined."""
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
    return None


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


def load_rulebook(data_dir: Optional[Union[str, Path]] = None) -> str:
    """Read only the active split's rulebook; never search another split."""
    root = Path(data_dir or os.environ.get("DATA_DIR", "/data"))
    path = root / "rulebook.md"
    return path.read_text(encoding="utf-8-sig") if path.is_file() else ""


def _plain(text: str) -> str:
    text = unicodedata.normalize("NFD", text.lower().replace("đ", "d"))
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def parse_time_rules_from_rulebook(rulebook_text: str = "") -> Dict[str, Any]:
    text = rulebook_text or ""
    plain = _plain(text)

    sunday_start = bool(re.search(
        r"(?:tuan|week).{0,100}(?:chu nhat|sunday).{0,35}(?:thu bay|saturday)|"
        r"(?:chu nhat|sunday).{0,35}(?:thu bay|saturday).{0,100}(?:tuan|week)|"
        r"(?:tuan|week).{0,40}(?:bat dau|start).{0,20}(?:chu nhat|sunday)",
        plain, re.DOTALL,
    ))
    week_start = "sunday" if sunday_start else "monday"

    full_month = bool(re.search(
        r"(?:thang nay|trong thang nay).{0,60}(?:toan bo|ca thang|full month|ngay cuoi thang|het thang)",
        plain, re.DOTALL,
    ))
    this_month_definition = "full_month" if full_month else "mtd"

    tet_start_lunar = None
    tet_end_lunar = None

    m_tet_start = re.search(
        r"(?:tu|bat dau tu)\s+(\d{1,2})\s*(?:thang chap|am|am lich|tet)?\s*(?:den|het|-)",
        plain,
    )
    if not m_tet_start:
        m_tet_start = re.search(r"(\d{1,2})\s*(?:thang chap|tet)\s*(?:den|het|-)", plain)

    m_tet_end = re.search(
        r"(?:den|het|-)\s*(?:ngay\s*)?(?:mung\s*)?(\d{1,2})\s*(?:thang gieng|tet|am lich)?",
        plain,
    )
    if "mung" in plain:
        m_mung = re.search(r"mung\s*(\d{1,2})", plain)
        if m_mung:
            tet_end_lunar = (1, int(m_mung.group(1)))

    if m_tet_start:
        start_day = int(m_tet_start.group(1))
        if 20 <= start_day <= 30:
            tet_start_lunar = (12, start_day)
        elif 1 <= start_day <= 10:
            tet_start_lunar = (1, start_day)

    if m_tet_end and not tet_end_lunar:
        end_day = int(m_tet_end.group(1))
        if 1 <= end_day <= 15:
            tet_end_lunar = (1, end_day)

    return {
        "week_start": week_start,
        "this_month_definition": this_month_definition,
        "as_of_date_inclusive": True,
        "tet_lunar_start": tet_start_lunar,
        "tet_lunar_end": tet_end_lunar,
        "raw_rulebook_length": len(text),
    }


def resolve_relative_period(
    period_key: str,
    as_of_date: Union[str, date, datetime],
    calendar_df: Optional[pd.DataFrame] = None,
    time_rules: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    ref_date = resolve_as_of_date(as_of_date)
    df = calendar_df if (calendar_df is not None and not calendar_df.empty) else load_calendar()
    if time_rules is None:
        time_rules = parse_time_rules_from_rulebook(load_rulebook())

    key = _plain(str(period_key or "")).strip().replace(" ", "_")
    sunday_start = (time_rules.get("week_start") == "sunday")
    this_month_def = time_rules.get("this_month_definition", "mtd")

    if key in ("this_month", "thang_nay", "thang_hien_tai", "trong_thang_nay"):
        start_m = date(ref_date.year, ref_date.month, 1)
        if this_month_def == "full_month":
            _, last_day = calendar.monthrange(ref_date.year, ref_date.month)
            end_m = date(ref_date.year, ref_date.month, last_day)
        else:
            end_m = ref_date
        return start_m.isoformat(), end_m.isoformat()

    if key in ("last_month", "thang_truoc", "thang_roi", "thang_vua_roi", "thang_qua"):
        if ref_date.month == 1:
            y, m = ref_date.year - 1, 12
        else:
            y, m = ref_date.year, ref_date.month - 1
        start_lm = date(y, m, 1)
        _, last_day = calendar.monthrange(y, m)
        end_lm = date(y, m, last_day)
        return start_lm.isoformat(), end_lm.isoformat()

    if key in ("this_week", "tuan_nay", "trong_tuan_nay", "tuan_hien_tai"):
        offset = (ref_date.weekday() + 1) % 7 if sunday_start else ref_date.weekday()
        start_w = ref_date - timedelta(days=offset)
        end_w = start_w + timedelta(days=6)
        return start_w.isoformat(), end_w.isoformat()

    if key in ("last_week", "tuan_truoc", "tuan_roi", "tuan_vua_roi", "tuan_qua"):
        offset = (ref_date.weekday() + 1) % 7 if sunday_start else ref_date.weekday()
        this_w_start = ref_date - timedelta(days=offset)
        start_lw = this_w_start - timedelta(days=7)
        end_lw = start_lw + timedelta(days=6)
        return start_lw.isoformat(), end_lw.isoformat()

    if key in ("same_period_last_year", "cung_ky_nam_ngoai", "cung_ky_nam_truoc", "so_voi_cung_ky_nam_ngoai"):
        target_year = ref_date.year - 1
        month = ref_date.month
        start_sp = date(target_year, month, 1)
        if this_month_def == "full_month":
            _, last_day = calendar.monthrange(target_year, month)
            end_sp = date(target_year, month, last_day)
        else:
            _, max_day = calendar.monthrange(target_year, month)
            end_sp = date(target_year, month, min(ref_date.day, max_day))
        return start_sp.isoformat(), end_sp.isoformat()

    if key in ("tet_current_year", "tet_last_year", "tet", "dip_tet", "tet_nguyen_dan", "mua_tet", "vu_tet", "nghi_tet", "tet_nam_ngoai"):
        target_year = ref_date.year - 1 if "last" in key or "ngoai" in key else ref_date.year
        l_start = time_rules.get("tet_lunar_start")
        l_end = time_rules.get("tet_lunar_end")
        if df is not None and not df.empty and l_start and l_end and "lunar_year" in df.columns:
            date_col = "solar_date" if "solar_date" in df.columns else "_solar_date"
            if date_col in df.columns:
                m_s = (df["lunar_year"] == target_year - 1) & (df["lunar_month"] == l_start[0]) & (df["lunar_day"] == l_start[1])
                m_e = (df["lunar_year"] == target_year) & (df["lunar_month"] == l_end[0]) & (df["lunar_day"] == l_end[1])
                r_s = df[m_s]
                r_e = df[m_e]
                if not r_s.empty and not r_e.empty:
                    val_s = r_s[date_col].iloc[0]
                    val_e = r_e[date_col].iloc[0]
                    return str(val_s)[:10], str(val_e)[:10]

        if df is not None and not df.empty and ("_event" in df.columns or "holiday" in df.columns):
            event_series = df["_event"] if "_event" in df.columns else df.get("holiday", pd.Series(dtype=object))
            event_clean = event_series.fillna("").astype(str).str.lower()
            date_col = "solar_date" if "solar_date" in df.columns else "_solar_date"
            if date_col in df.columns:
                m_tet = (
                    df[date_col].map(lambda d: hasattr(d, "year") and d.year == target_year and hasattr(d, "month") and d.month in (1, 2))
                    & event_clean.str.contains(r"\bt[eế]t\b|nguy[eê]n\s*đ[aá]n", regex=True)
                )
                matched = df[m_tet]
                if not matched.empty:
                    return str(matched[date_col].min())[:10], str(matched[date_col].max())[:10]

        win = get_tet_window(year=target_year, as_of=ref_date)
        if win:
            return win[0].isoformat(), win[1].isoformat()

        mung_1 = get_tet_date(target_year)
        if mung_1:
            return (mung_1 - timedelta(days=2)).isoformat(), (mung_1 + timedelta(days=4)).isoformat()

        return f"{target_year}-02-01", f"{target_year}-02-15"

    if key in ("ytd", "nam_nay", "trong_nam_nay"):
        return date(ref_date.year, 1, 1).isoformat(), ref_date.isoformat()

    if key in ("last_year", "nam_ngoai", "nam_truoc", "nam_vua_roi"):
        return date(ref_date.year - 1, 1, 1).isoformat(), date(ref_date.year - 1, 12, 31).isoformat()

    pair = resolve_vietnamese_relative_date(period_key, as_of=ref_date)
    if pair:
        return pair[0].isoformat(), pair[1].isoformat()

    return ref_date.isoformat(), ref_date.isoformat()


def resolve_time_context(
    as_of_date: Union[str, date, datetime],
    calendar_df: Optional[Any] = None,
    rulebook_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Return ISO date anchors from the active split for prompts or generated code.

    Holiday windows are absent when the split calendar does not define them.
    No revenue rule is interpreted here; the original rulebook remains available
    to callers for business calculations.
    """
    if isinstance(as_of_date, str):
        anchor = date.fromisoformat(as_of_date)
    elif isinstance(as_of_date, datetime):
        anchor = as_of_date.date()
    elif isinstance(as_of_date, date):
        anchor = as_of_date
    else:
        raise ValueError("as_of_date must be an ISO date or date object")
    rb = rulebook_text if rulebook_text is not None else load_rulebook()
    week_rule = _plain(rb)
    sunday_start = bool(re.search(
        r"(?:tuan|week).{0,100}(?:chu nhat|sunday).{0,35}(?:thu bay|saturday)|"
        r"(?:chu nhat|sunday).{0,35}(?:thu bay|saturday).{0,100}(?:tuan|week)|"
        r"(?:tuan|week).{0,40}(?:bat dau|start).{0,20}(?:chu nhat|sunday)",
        week_rule, re.DOTALL,
    ))

    def span(start: date, end: date) -> Dict[str, str]:
        return {"start": start.isoformat(), "end": end.isoformat()}

    week_offset = (anchor.weekday() + (1 if sunday_start else 0)) % 7
    week_start = anchor - timedelta(days=week_offset)
    month_start = anchor.replace(day=1)
    prev_month_end = month_start - timedelta(days=1)
    quarter_start = date(anchor.year, ((anchor.month - 1) // 3) * 3 + 1, 1)
    prev_quarter_end = quarter_start - timedelta(days=1)
    prev_quarter_start = date(prev_quarter_end.year, ((prev_quarter_end.month - 1) // 3) * 3 + 1, 1)
    prior_mtd_end = date(anchor.year - 1, anchor.month, min(anchor.day, calendar.monthrange(anchor.year - 1, anchor.month)[1]))
    prior_month_start = date(anchor.year - 1, anchor.month, 1)
    prior_month_end = date(anchor.year - 1, anchor.month, calendar.monthrange(anchor.year - 1, anchor.month)[1])
    context: Dict[str, Any] = {
        "as_of_date": anchor.isoformat(),
        "weekday": anchor.strftime("%A"),
        "yesterday": (anchor - timedelta(days=1)).isoformat(),
        "week_start": "sunday" if sunday_start else "monday",
        "this_week": span(week_start, week_start + timedelta(days=6)),
        "last_week": span(week_start - timedelta(days=7), week_start - timedelta(days=1)),
        "this_month": span(month_start, anchor),
        "last_month": span(prev_month_end.replace(day=1), prev_month_end),
        "same_month_last_year_mtd": span(prior_month_start, prior_mtd_end),
        "same_month_last_year_full": span(prior_month_start, prior_month_end),
        "this_quarter": span(quarter_start, anchor),
        "last_quarter": span(prev_quarter_start, prev_quarter_end),
        "this_year": span(date(anchor.year, 1, 1), anchor),
        "last_year": span(date(anchor.year - 1, 1, 1), date(anchor.year - 1, 12, 31)),
        "lunar_windows": {},
    }

    df = calendar_df if calendar_df is not None else load_calendar()
    if df is not None and _HAS_PANDAS and not df.empty:
        date_col = "_solar_date" if "_solar_date" in df else next(
            (c for c in ("date", "solar_date", "duong_lich", "ngay_duong") if c in df), None
        )
        event_col = "_event" if "_event" in df else next(
            (c for c in ("holiday", "holiday_name", "event", "su_kien", "ten_le", "ghi_chu") if c in df), None
        )
        if date_col and event_col:
            day_first = df[date_col].astype(str).str.match(r"^\d{1,2}/\d{1,2}/\d{4}").mean() > 0.5
            dates = pd.to_datetime(df[date_col], errors="coerce", dayfirst=day_first).dt.date
            events = df[event_col].fillna("").astype(str).map(_plain)
            for year in (anchor.year, anchor.year - 1):
                yearly: Dict[str, Dict[str, str]] = {}
                for name, pattern in (("tet_nguyen_dan", r"\btet\b|tet nguyen dan"),
                                      ("trung_thu", r"trung thu|mid.?autumn")):
                    selected = dates[(dates.map(lambda d: d is not None and not pd.isna(d) and d.year == year))
                                     & events.str.contains(pattern, regex=True)]
                    if not selected.empty:
                        yearly[name] = span(selected.min(), selected.max())
                if yearly:
                    context["lunar_windows"][str(year)] = yearly
    return context


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


def get_precomputed_intervals_summary(
    as_of: Optional[Union[str, date, datetime]] = None,
    calendar_path: Optional[Union[str, Path]] = None,
) -> str:
    """Pre-compute common Vietnamese relative date intervals anchored on as_of_date.

    Generates explicit ISO date ranges for 'hôm nay', 'hôm qua', 'tuần này',
    'tuần trước' (Mon-Sun), 'tháng này' (MTD & full month), 'tháng trước',
    'cùng kỳ năm ngoái', 'năm nay' (YTD & full year), 'năm trước', 'quý này',
    'quý trước', and 'Tết Nguyên Đán' window so the LLM never misinterprets
    relative date boundaries.

    Returns:
        Formatted Markdown text block suitable for prompt injection.
    """
    ref_date = resolve_as_of_date(as_of)
    yesterday = ref_date - timedelta(days=1)

    # Week boundaries (Mon-Sun)
    this_week_start, this_week_end = get_week_date_range(ref_date, week_start_monday=True)
    last_week_start = this_week_start - timedelta(days=7)
    last_week_end = this_week_start - timedelta(days=1)

    # Month boundaries
    cur_month_start = date(ref_date.year, ref_date.month, 1)
    _, cur_month_last_day = calendar.monthrange(ref_date.year, ref_date.month)
    cur_month_full_end = date(ref_date.year, ref_date.month, cur_month_last_day)
    mtd_end = ref_date

    # Last month
    if ref_date.month == 1:
        last_month_year, last_month_num = ref_date.year - 1, 12
    else:
        last_month_year, last_month_num = ref_date.year, ref_date.month - 1
    last_month_start, last_month_end = get_month_date_range(last_month_year, last_month_num)

    # Same period last year (cùng kỳ năm ngoái)
    same_period_mtd_start, same_period_mtd_end = get_same_period_last_year(cur_month_start, mtd_end)
    same_period_month_start, same_period_month_end = get_same_period_last_year(cur_month_start, cur_month_full_end)

    # Quarter boundaries
    q_num = get_quarter(ref_date)
    this_q_start, this_q_full_end = get_quarter_date_range(ref_date.year, q_num)
    if q_num == 1:
        last_q_year, last_q_num = ref_date.year - 1, 4
    else:
        last_q_year, last_q_num = ref_date.year, q_num - 1
    last_q_start, last_q_end = get_quarter_date_range(last_q_year, last_q_num)

    # Year boundaries
    ytd_start = date(ref_date.year, 1, 1)
    ytd_end = ref_date
    last_year_start = date(ref_date.year - 1, 1, 1)
    last_year_end = date(ref_date.year - 1, 12, 31)

    # Tet windows
    try:
        tet_start, tet_end = get_tet_window(ref_date.year, as_of=ref_date, calendar_path=calendar_path)
        tet_str = f"từ `{tet_start.isoformat()}` đến `{tet_end.isoformat()}`"
    except Exception:
        tet_str = "Chưa xác định"

    lines = [
        "### KHOẢNG THỜI GIAN ĐÃ TÍNH TOÁN SẴN (PRE-COMPUTED TIME INTERVALS DỰA VÀO as_of_date):",
        f"- Ngày tham chiếu / Hôm nay (Today / as_of_date): `{ref_date.isoformat()}`",
        f"- Hôm qua (Yesterday): `{yesterday.isoformat()}`",
        f"- Tuần này (This week, Thứ 2 - Chủ Nhật): từ `{this_week_start.isoformat()}` đến `{this_week_end.isoformat()}`",
        f"- Tuần trước / Tuần rồi (Last week, Thứ 2 - Chủ Nhật): từ `{last_week_start.isoformat()}` đến `{last_week_end.isoformat()}`",
        f"- Tháng này (Month-to-Date, từ đầu tháng đến as_of_date): từ `{cur_month_start.isoformat()}` đến `{mtd_end.isoformat()}`",
        f"- Toàn bộ tháng này (Full current month): từ `{cur_month_start.isoformat()}` đến `{cur_month_full_end.isoformat()}`",
        f"- Tháng trước / Tháng rồi (Last month): từ `{last_month_start.isoformat()}` đến `{last_month_end.isoformat()}`",
        f"- Cùng kỳ tháng này năm ngoái (Same period last year MTD): từ `{same_period_mtd_start.isoformat()}` đến `{same_period_mtd_end.isoformat()}`",
        f"- Quý này (This quarter Q{q_num}): từ `{this_q_start.isoformat()}` đến `{this_q_full_end.isoformat()}` (hoặc đến as_of_date `{ref_date.isoformat()}`)",
        f"- Quý trước (Last quarter Q{last_q_num}/{last_q_year}): từ `{last_q_start.isoformat()}` đến `{last_q_end.isoformat()}`",
        f"- Năm nay (Year-to-Date): từ `{ytd_start.isoformat()}` đến `{ytd_end.isoformat()}`",
        f"- Năm ngoái / Năm trước (Last year): từ `{last_year_start.isoformat()}` đến `{last_year_end.isoformat()}`",
        f"- Dịp Tết Nguyên Đán năm {ref_date.year} (Tet holiday window): {tet_str}",
    ]
    return "\n".join(lines)
