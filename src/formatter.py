"""Output Normalizer for Zalo AI Challenge 2026 - DataMate Track.

Normalizes model outputs per competition rules:
1. NFC Unicode normalization for Vietnamese text.
2. Floating point rounding to 2 decimal places.
3. Proper integer, boolean, and date string representation.
4. Unwrapping of Pandas Series, DataFrame, and NumPy scalars.
5. Strict fallback to abstain token (null / None) for unanswerable or failed questions
   to prevent negative penalties under Penalized Accuracy evaluation.
"""

from __future__ import annotations

from datetime import date, datetime
import math
import re
from typing import Any, Optional, Union
import unicodedata

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    _HAS_PANDAS = False

# Default abstain representation for submission.jsonl (null in JSON)
DEFAULT_ABSTAIN_VALUE: Any = None


def normalize_unicode_nfc(text: str) -> str:
    """Normalize Vietnamese text to canonical precomposed Unicode (NFC).

    Args:
        text: Input string.

    Returns:
        NFC normalized and whitespace-trimmed string.
    """
    if not isinstance(text, str):
        return str(text)
    return unicodedata.normalize("NFC", text.strip())


def is_abstain_signal(val: Any) -> bool:
    """Check if the extracted result indicates an intentional or error-induced abstain."""
    if val is None:
        return True
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("abstain", "none", "null", "nan", "không có", "không xác định", "n/a", "na", ""):
            return True
    if isinstance(val, float) and math.isnan(val):
        return True
    if _HAS_PANDAS and pd.isna(val):
        return True
    return False


def unwrap_container(val: Any) -> Any:
    """Unwrap 1-element Pandas DataFrame, Series, or NumPy scalar to Python primitives.

    Args:
        val: Container or scalar value.

    Returns:
        Unwrapped Python primitive or original value.
    """
    if _HAS_PANDAS:
        if isinstance(val, pd.DataFrame):
            if val.empty:
                return None
            if val.shape == (1, 1):
                val = val.iloc[0, 0]
            elif val.shape[0] == 1:
                return val.iloc[0].to_dict()
            else:
                return val.to_dict(orient="records")

        if isinstance(val, pd.Series):
            if val.empty:
                return None
            if len(val) == 1:
                val = val.iloc[0]
            else:
                return val.tolist()

    if _HAS_NUMPY:
        if isinstance(val, np.generic):
            return val.item()

    return val


def format_scalar(
    val: Any,
    round_decimals: int = 2,
    abstain_fallback: Any = DEFAULT_ABSTAIN_VALUE,
) -> Any:
    """Format and normalize a scalar value per competition rules.

    Rules applied:
    - Null / NaN / empty -> abstain_fallback (avoids penalized accuracy penalty)
    - float -> rounded to 2 decimal places (if integer-equivalent like 15.0, cast to int 15)
    - int -> int
    - date / datetime -> ISO format string YYYY-MM-DD
    - str -> NFC normalized string with surrounding quotes removed
    - bool -> bool (True / False)

    Args:
        val: Input scalar value.
        round_decimals: Number of decimal places to round floats to (default: 2).
        abstain_fallback: Value to return on unanswerable or invalid result.

    Returns:
        Normalized value.
    """
    val = unwrap_container(val)

    if is_abstain_signal(val):
        return abstain_fallback

    # Date / Datetime
    if isinstance(val, datetime):
        return val.strftime("%Y-%m-%d %H:%M:%S") if (val.hour or val.minute or val.second) else val.strftime("%Y-%m-%d")
    if isinstance(val, date):
        return val.strftime("%Y-%m-%d")

    # Boolean (must precede int check since bool subclasses int)
    if isinstance(val, bool):
        return val

    # Integers
    if isinstance(val, int):
        return val

    # Floats
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return abstain_fallback
        rounded = round(val, round_decimals)
        # If float represents an exact integer (e.g. count or sum like 42.0), convert to int
        if rounded.is_integer():
            return int(rounded)
        return rounded

    # Strings
    if isinstance(val, str):
        cleaned = val.strip()
        # Remove accidental surrounding quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        # Try parsing numeric strings that might be formatted as strings
        try:
            if re_match := re.match(r"^-?\d+$", cleaned):
                return int(cleaned)
            if re_float := re.match(r"^-?\d+\.\d+$", cleaned):
                num = round(float(cleaned), round_decimals)
                return int(num) if num.is_integer() else num
        except Exception:
            pass

        return normalize_unicode_nfc(cleaned)

    # Lists / Tuples
    if isinstance(val, (list, tuple, set)):
        normalized_list = [format_scalar(item, round_decimals, abstain_fallback) for item in val]
        # Filter out trailing/leading abstain items if list is empty
        if not normalized_list:
            return abstain_fallback
        return normalized_list

    # Dictionaries
    if isinstance(val, dict):
        return {
            normalize_unicode_nfc(k): format_scalar(v, round_decimals, abstain_fallback)
            for k, v in val.items()
        }

    return normalize_unicode_nfc(str(val))

