"""Output Normalizer & Submission Formatter for Zalo AI Challenge 2026 - DataMate Track (Bước 5).

Normalizes model outputs and Pandas query results per competition rules:
1. Chuẩn hóa String: NFC Unicode, đưa về chữ thường, thay ký tự đặc biệt bằng khoảng trắng, xóa khoảng trắng thừa.
2. Chuẩn hóa Number / Percent: Ép kiểu float, làm tròn đúng 2 chữ số thập phân (round(val, 2)).
3. Quyết định Abstain: Nếu chạy quá 3 lần vẫn lỗi hoặc trả về DataFrame rỗng vô lý -> gán answer = "abstain".
4. Format dictionary thành chuỗi JSON chứa id, type, answer và ghi append vào file submission.jsonl.
"""

from __future__ import annotations

from __future__ import annotations

from datetime import date, datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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

# Default abstain representation per BTC rules
DEFAULT_ABSTAIN_TOKEN: str = "abstain"
DEFAULT_ABSTAIN_VALUE: str = DEFAULT_ABSTAIN_TOKEN


def normalize_unicode_nfc(text: str) -> str:
    """Normalize Vietnamese text to canonical precomposed Unicode (NFC).

    Args:
        text: Input string.

    Returns:
        NFC normalized and trimmed string.
    """
    if not isinstance(text, str):
        return str(text)
    return unicodedata.normalize("NFC", text.strip())


def normalize_for_comparison(text: Any) -> str:
    """Exact competition normalization helper for validation and ground-truth comparison.

    Rules:
    1. unicodedata.normalize('NFC', s).lower()
    2. Replace every character that is not a letter, digit, or whitespace with ' '
    3. Strip and collapse whitespace runs to single spaces
    4. Keep Vietnamese diacritics intact

    Args:
        text: Input string or value.

    Returns:
        Normalized comparison string.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. NFC normalization + lowercase
    text = unicodedata.normalize("NFC", text.strip()).lower()

    # 2. Replace non-alphanumeric/whitespace chars and underscores with spaces
    text = re.sub(r"[^\w\s]|_", " ", text)

    # 3. Collapse multiple whitespace runs to single space and strip
    return re.sub(r"\s+", " ", text).strip()


# Backward-compatible alias
clean_string = normalize_for_comparison


def format_string(text: Any, clean_text: bool = False) -> str:
    """Format string output for competition submission.

    When clean_text is False (default for submission), preserves the exact canonical
    capitalization and diacritics of the product/entity from the business table,
    stripping surrounding redundant quotes.
    When clean_text is True (for comparison), applies full clean_string normalization.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    cleaned = text.strip()
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()

    if clean_text:
        return normalize_for_comparison(cleaned)
    return normalize_unicode_nfc(cleaned)


def format_number(val: Any, round_decimals: int = 2) -> Union[int, float]:
    """Strictly format numeric value (int or float rounded to 2 decimal places).

    Cleans VND amounts, currency symbols (VND, vnđ, vn₫, đ, ₫, $, usd), and
    thousand-separator commas/dots so the result is a clean Python int or float.

    Args:
        val: Numeric value or string representation.
        round_decimals: Number of decimal places to round floats to (default: 2).

    Returns:
        Rounded float or simplified integer.
    """
    if val is None:
        raise ValueError("Cannot parse None as number.")
    if isinstance(val, bool):
        return int(val)
    if _HAS_NUMPY and isinstance(val, (np.integer, np.floating)):
        val = val.item()

    if isinstance(val, (int, float)):
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Cannot parse {val} as number.")
            rounded = round(val, round_decimals)
            return int(rounded) if rounded.is_integer() else rounded
        return int(val)

    if isinstance(val, str):
        cleaned = val.strip()

        # Handle percentage strings in format_number (backward compatibility)
        if m_pct := re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$", cleaned):
            num = round(float(m_pct.group(1)), round_decimals)
            return int(num) if num.is_integer() else num

        # Strip currency symbols: vnd, vnđ, vn₫, đ, ₫, $, usd, đồng
        cleaned = re.sub(r"(?i)(?:vnd|vnđ|vn₫|usd|đồng|dong|[$₫đ])", "", cleaned).strip()
        cleaned = re.sub(r"\s+", "", cleaned)

        if not cleaned:
            raise ValueError(f"Cannot parse '{val}' as number.")

        # Handle thousand-separator commas and dots
        if "," in cleaned and "." in cleaned:
            last_comma = cleaned.rfind(",")
            last_dot = cleaned.rfind(".")
            if last_comma > last_dot:
                # Format: 15.000.000,50 (dot thousand, comma decimal)
                cleaned = cleaned.replace(".", "").replace(",", ".")
            else:
                # Format: 15,000,000.50 (comma thousand, dot decimal)
                cleaned = cleaned.replace(",", "")
        elif "," in cleaned:
            parts = cleaned.split(",")
            if len(parts) > 2 or any(len(p) == 3 for p in parts[1:]):
                cleaned = "".join(parts)
            else:
                cleaned = cleaned.replace(",", ".")
        elif "." in cleaned:
            parts = cleaned.split(".")
            if len(parts) > 2:
                cleaned = "".join(parts)
            elif len(parts) == 2 and len(parts[1]) == 3 and any(sym in str(val).lower() for sym in ("vnd", "vnđ", "vn₫", "đ", "₫", "$")):
                cleaned = "".join(parts)

        num = round(float(cleaned), round_decimals)
        return int(num) if num.is_integer() else num

    raise ValueError(f"Cannot parse '{val}' as number.")


def format_percent(val: Any, round_decimals: int = 2) -> Union[int, float]:
    """Strictly format numeric percentage value per evaluation rules.

    Outputs numeric percentage value (e.g. -12.45, NOT '-12.45%' and NOT -0.1245),
    rounded to 2 decimal places with exact sign.

    Args:
        val: Numeric percentage, string percentage ('-12.45%'), or ratio (-0.1245).
        round_decimals: Decimal places to round to (default: 2).

    Returns:
        Rounded float or integer percentage value.
    """
    if val is None:
        raise ValueError("Cannot parse None as percent.")
    if _HAS_NUMPY and isinstance(val, (np.integer, np.floating)):
        val = val.item()

    if isinstance(val, str):
        cleaned = val.strip()
        has_pct_sym = "%" in cleaned
        cleaned = cleaned.replace("%", "").strip()
        cleaned = re.sub(r"\s+", "", cleaned)
        if "," in cleaned and "." not in cleaned:
            cleaned = cleaned.replace(",", ".")
        num = float(cleaned)
        # If input was a raw ratio string like "-0.1245" without % sign
        if not has_pct_sym and 0 < abs(num) < 1.0:
            num = num * 100.0
        rounded = round(num, round_decimals)
        return int(rounded) if rounded.is_integer() else rounded

    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            raise ValueError(f"Cannot parse {val} as percent.")
        num = float(val)
        # Convert float ratio e.g. -0.1245 -> -12.45
        if 0 < abs(num) < 1.0:
            num = num * 100.0
        rounded = round(num, round_decimals)
        return int(rounded) if rounded.is_integer() else rounded

    raise ValueError(f"Cannot parse '{val}' as percent.")


def format_date(val: Any, as_month: bool = False) -> str:
    """Strictly format date as ISO string 'YYYY-MM-DD' (or 'YYYY-MM' when month requested).

    Args:
        val: Date, datetime, Timestamp, or date string.
        as_month: If True, formats as 'YYYY-MM'.

    Returns:
        Clean ISO formatted date string.
    """
    if val is None:
        raise ValueError("Cannot format None as date.")
    if isinstance(val, datetime):
        if as_month:
            return val.strftime("%Y-%m")
        if val.hour or val.minute or val.second:
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val.strftime("%Y-%m-%d")
    if isinstance(val, date):
        return val.strftime("%Y-%m") if as_month else val.strftime("%Y-%m-%d")
    if _HAS_PANDAS and isinstance(val, pd.Timestamp):
        if as_month:
            return val.strftime("%Y-%m")
        if val.hour or val.minute or val.second:
            return val.strftime("%Y-%m-%d %H:%M:%S")
        return val.strftime("%Y-%m-%d")

    s = str(val).strip()
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        s = s[1:-1].strip()

    # YYYY-MM-DD HH:MM:SS
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", s):
        return s[:7] if as_month else s
    # YYYY-MM-DD or YYYY/MM/DD
    if m := re.match(r"^(\d{4})[/-](\d{1,2})[/-](\d{1,2})$", s):
        d = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return d.strftime("%Y-%m") if as_month else d.strftime("%Y-%m-%d")
    # YYYY-MM or YYYY/MM
    if m := re.match(r"^(\d{4})[/-](\d{1,2})$", s):
        return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}"
    # DD/MM/YYYY or DD-MM-YYYY
    if m := re.match(r"^(\d{1,2})[/-](\d{1,2})[/-](\d{4})$", s):
        d = date(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        return d.strftime("%Y-%m") if as_month else d.strftime("%Y-%m-%d")
    # MM/YYYY or MM-YYYY
    if m := re.match(r"^(\d{1,2})[/-](\d{4})$", s):
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}"
    # Vietnamese "tháng M/YYYY" or "tháng MM năm YYYY"
    if m := re.search(r"tháng\s*(\d{1,2})(?:[/-]|\s*năm\s*)(\d{4})", s, re.IGNORECASE):
        return f"{int(m.group(2)):04d}-{int(m.group(1)):02d}"

    return s


def format_boolean(val: Any) -> bool:
    """Strictly format value as JSON boolean True or False.

    Args:
        val: Boolean or truthy/falsy scalar/string.

    Returns:
        Python bool (serializes to JSON true or false).
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "t", "yes", "đúng", "dung", "có", "co"):
            return True
        if s in ("false", "0", "f", "no", "sai", "không", "khong"):
            return False
    return bool(val)


def sort_ranking_items(
    items: List[Any],
    criterion_fn: Optional[Callable[[Any], Any]] = None,
    code_fn: Optional[Callable[[Any], Any]] = None,
    descending: bool = True,
) -> List[Any]:
    """Sort items per rulebook default: descending criterion over valid orders; tie-break by item code ascending.

    Args:
        items: List of items to rank.
        criterion_fn: Function to extract primary criterion value (e.g. revenue, quantity).
        code_fn: Function to extract tie-breaking item code string.
        descending: Whether primary criterion is descending (default: True).

    Returns:
        Sorted list of items.
    """
    if not items:
        return []

    c_fn = criterion_fn or (lambda x: x[1] if isinstance(x, (tuple, list)) and len(x) > 1 else x)
    k_fn = code_fn or (lambda x: x[0] if isinstance(x, (tuple, list)) and len(x) > 0 else str(x))

    def sort_key(item: Any) -> Tuple[float, str]:
        c_val = c_fn(item)
        try:
            num = float(c_val)
        except (ValueError, TypeError):
            num = 0.0
        code_str = str(k_fn(item))
        primary = -num if descending else num
        return (primary, code_str)

    return sorted(items, key=sort_key)


def format_list(val: Any, round_decimals: int = 2, clean_text: bool = False) -> List[Any]:
    """Strictly format ordered JSON array matching question criterion or rulebook default.

    Args:
        val: Sequence, Series, ndarray, or JSON list string.
        round_decimals: Rounding decimals for float items (default: 2).
        clean_text: Whether to apply clean_string to text items.

    Returns:
        Clean Python list.
    """
    if _HAS_PANDAS and isinstance(val, (pd.Series, pd.Index)):
        val = val.tolist()
    elif _HAS_NUMPY and isinstance(val, np.ndarray):
        val = val.tolist()
    elif isinstance(val, (tuple, set)):
        val = list(val)
    elif isinstance(val, str):
        s = val.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    val = parsed
            except Exception:
                pass
        if not isinstance(val, list):
            if "," in s:
                val = [p.strip() for p in s.split(",") if p.strip()]
            else:
                val = [s]

    if not isinstance(val, list):
        val = [val]

    return [
        format_scalar(item, round_decimals=round_decimals, clean_text=clean_text)
        for item in val
    ]


def format_table(val: Any, round_decimals: int = 2, clean_text: bool = False) -> Dict[str, Any]:
    """Strictly format table as {"columns": [...], "rows": [[...], ...]}.

    Ensures normalized column names (NFC) and matching properly-typed cell values.

    Args:
        val: DataFrame, dict of lists, list of dicts, or table dict.
        round_decimals: Decimals to round numeric cells to.
        clean_text: Whether to clean text cells.

    Returns:
        Structured table dict {"columns": [...], "rows": [[...], ...]}.
    """
    if _HAS_PANDAS and isinstance(val, pd.DataFrame):
        columns = [normalize_unicode_nfc(str(c)) for c in val.columns]
        if val.empty:
            return {"columns": columns, "rows": []}
        rows = [
            [format_scalar(cell, round_decimals=round_decimals, clean_text=clean_text) for cell in row]
            for row in val.itertuples(index=False)
        ]
        return {"columns": columns, "rows": rows}

    if isinstance(val, dict):
        if "columns" in val and "rows" in val:
            columns = [normalize_unicode_nfc(str(c)) for c in val["columns"]]
            rows = [
                [format_scalar(cell, round_decimals=round_decimals, clean_text=clean_text) for cell in row]
                if isinstance(row, (list, tuple))
                else [format_scalar(row, round_decimals=round_decimals, clean_text=clean_text)]
                for row in val["rows"]
            ]
            return {"columns": columns, "rows": rows}
        columns = [normalize_unicode_nfc(str(k)) for k in val.keys()]
        if columns and all(isinstance(v, (list, tuple)) for v in val.values()):
            num_rows = len(val[columns[0]]) if columns else 0
            rows = []
            for i in range(num_rows):
                rows.append([
                    format_scalar(val[c][i], round_decimals=round_decimals, clean_text=clean_text)
                    for c in columns
                ])
            return {"columns": columns, "rows": rows}

    if isinstance(val, list):
        if not val:
            return {"columns": [], "rows": []}
        if all(isinstance(item, dict) for item in val):
            columns = [normalize_unicode_nfc(str(k)) for k in val[0].keys()]
            rows = [
                [format_scalar(item.get(c), round_decimals=round_decimals, clean_text=clean_text) for c in columns]
                for item in val
            ]
            return {"columns": columns, "rows": rows}
        if all(isinstance(item, (list, tuple)) for item in val):
            rows = [
                [format_scalar(cell, round_decimals=round_decimals, clean_text=clean_text) for cell in item]
                for item in val
            ]
            columns = [f"col_{i}" for i in range(len(rows[0]))] if rows else []
            return {"columns": columns, "rows": rows}

    return {"columns": [], "rows": []}


def is_abstain_signal(val: Any, success: bool = True) -> bool:
    """Kiểm tra điều kiện Abstain:

    - Nếu quá trình thực thi lỗi (success=False).
    - Hoặc kết quả là None, NaN, inf.
    - Hoặc là bảng DataFrame rỗng (df.empty), Series rỗng, collection rỗng.
    - Hoặc chuỗi trả về chứa tín hiệu không xác định ('abstain', 'null', 'none', 'không có', v.v.).

    Args:
        val: Extracted result value.
        success: Whether code execution succeeded.

    Returns:
        True if the result should trigger an abstain decision.
    """
    if not success:
        return True
    if val is None:
        return True

    # Empty Pandas containers
    if _HAS_PANDAS:
        if isinstance(val, (pd.DataFrame, pd.Series)):
            return bool(val.empty)
        try:
            if pd.isna(val):
                return True
        except Exception:
            pass

    # Empty collections
    if isinstance(val, (list, tuple, set, dict)) and len(val) == 0:
        return True

    # Float NaN / Inf
    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
        return True

    # String explicit abstain tokens
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("abstain", "unanswerable", "none", "null", "nan", "không có", "không xác định", "n/a", "na", ""):
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
    abstain_fallback: Any = DEFAULT_ABSTAIN_TOKEN,
    clean_text: bool = False,
    q_type: Optional[str] = None,
) -> Any:
    """Format and normalize a scalar value per competition rules.

    Args:
        val: Input scalar value.
        round_decimals: Number of decimal places to round floats to (default: 2).
        abstain_fallback: Value to return on unanswerable or invalid result (default: 'abstain').
        clean_text: If True, applies full clean_string() (lowercase, remove special chars).
        q_type: Optional question type ('number', 'percent', 'string', 'date', 'boolean', 'list', 'table').

    Returns:
        Normalized value matching evaluation requirements.
    """
    if q_type == "table":
        if is_abstain_signal(val):
            return abstain_fallback
        return format_table(val, round_decimals=round_decimals, clean_text=clean_text)

    val = unwrap_container(val)

    if is_abstain_signal(val):
        return abstain_fallback

    # Explicit type dispatch
    if q_type == "percent":
        try:
            return format_percent(val, round_decimals=round_decimals)
        except Exception:
            return abstain_fallback

    if q_type == "number":
        try:
            return format_number(val, round_decimals=round_decimals)
        except Exception:
            return abstain_fallback

    if q_type == "date":
        try:
            return format_date(val)
        except Exception:
            return abstain_fallback

    if q_type == "boolean":
        return format_boolean(val)

    if q_type == "list":
        return format_list(val, round_decimals=round_decimals, clean_text=clean_text)

    # Inferred types
    if isinstance(val, (datetime, date)) or (_HAS_PANDAS and isinstance(val, pd.Timestamp)):
        return format_date(val)

    if isinstance(val, bool):
        return val

    if isinstance(val, int):
        return val

    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return abstain_fallback
        rounded = round(val, round_decimals)
        return int(rounded) if rounded.is_integer() else rounded

    if isinstance(val, str):
        cleaned = val.strip()
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        # Handle percent strings (e.g. '15.5%')
        if m_pct := re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$", cleaned):
            num = round(float(m_pct.group(1)), round_decimals)
            return int(num) if num.is_integer() else num

        # Handle numeric strings
        try:
            if re.match(r"^[+-]?\d+$", cleaned):
                return int(cleaned)
            if re.match(r"^[+-]?\d+\.\d+$", cleaned):
                num = round(float(cleaned), round_decimals)
                return int(num) if num.is_integer() else num
        except Exception:
            pass

        # Handle ISO Date string preservation (e.g. '2024-05-15')
        if re.match(r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?$", cleaned):
            return format_date(cleaned)

        if clean_text:
            return clean_string(cleaned)
        return normalize_unicode_nfc(cleaned)

    if isinstance(val, (list, tuple, set)):
        normalized_list = [format_scalar(item, round_decimals, abstain_fallback, clean_text) for item in val]
        if not normalized_list:
            return abstain_fallback
        return normalized_list

    if isinstance(val, dict):
        if "columns" in val and "rows" in val:
            return format_table(val, round_decimals=round_decimals, clean_text=clean_text)
        return {
            normalize_unicode_nfc(k): format_scalar(v, round_decimals, abstain_fallback, clean_text)
            for k, v in val.items()
        }

    return normalize_unicode_nfc(str(val))


def format_output(
    val: Any,
    success: bool = True,
    round_decimals: int = 2,
    abstain_token: str = DEFAULT_ABSTAIN_TOKEN,
    clean_text: bool = False,
    q_type: Optional[str] = None,
) -> Any:
    """Top-level Bước 5 formatter:

    Quyết định Abstain:
    - Nếu chạy quá 3 lần vẫn lỗi (success=False),
    - Hoặc kết quả là DataFrame rỗng, None, NaN -> gán answer = "abstain".
    Ngược lại, chuẩn hóa số, chuỗi theo quy định.

    Args:
        val: Raw result from execution.
        success: Whether sandbox execution succeeded.
        round_decimals: Rounding decimals for floats (default: 2).
        abstain_token: Value to assign on abstain (default: 'abstain').
        clean_text: Whether to apply clean_string() to text answers.
        q_type: Optional question type string.

    Returns:
        Formatted answer per BTC rules.
    """
    if not success or is_abstain_signal(val, success=success):
        return abstain_token

    return format_scalar(
        val=val,
        round_decimals=round_decimals,
        abstain_fallback=abstain_token,
        clean_text=clean_text,
        q_type=q_type,
    )


def format_submission_record(
    q_id: str,
    answer: Any,
    q_type: Optional[str] = None,
    abstain_token: str = DEFAULT_ABSTAIN_TOKEN,
    reason: Optional[str] = None,
) -> Dict[str, Any]:
    """Format dictionary cuối cùng chứa id, type, answer theo luật BTC DataMate.

    Đảm bảo:
    - Bỏ qua trường 'alias' (strictly omit alias).
    - Khớp chuẩn: {"id": ..., "type": ..., "answer": ...} (hoặc có thêm "reason" cho unanswerable).
    - Không chứa placeholder rác ("None", "null", "nan", "N/A", "").

    Args:
        q_id: Question ID.
        answer: Normalized answer value.
        q_type: Question type (optional, auto-inferred if None).
        abstain_token: Token indicating an abstain response.
        reason: Optional explanation string for unanswerable questions.

    Returns:
        Dictionary {'id': ..., 'type': ..., 'answer': ...} (không chứa alias).
    """
    is_abstain = False
    if answer is None:
        is_abstain = True
    elif isinstance(answer, str) and answer.strip().lower() in (abstain_token.lower(), "abstain"):
        is_abstain = True
    elif q_type == "abstain":
        is_abstain = True

    is_unans = False
    if isinstance(answer, str) and answer == "unanswerable":
        is_unans = True
    elif q_type == "unanswerable":
        is_unans = True

    if is_abstain:
        return {
            "id": str(q_id),
            "type": "abstain",
        }
    elif is_unans:
        q_type = "unanswerable"
        answer = "unanswerable"
    elif q_type is None:
        if isinstance(answer, bool):
            q_type = "boolean"
        elif isinstance(answer, (int, float)):
            q_type = "number"
        elif isinstance(answer, (list, tuple)):
            q_type = "list"
        elif isinstance(answer, dict) and "columns" in answer and "rows" in answer:
            q_type = "table"
        elif _HAS_PANDAS and isinstance(answer, pd.DataFrame):
            q_type = "table"
        else:
            q_type = "string"

    # Enforce table formatting if type is table
    if q_type == "table" and not is_abstain and not is_unans:
        answer = format_table(answer)

    record: Dict[str, Any] = {
        "id": str(q_id),
        "type": str(q_type),
        "answer": answer,
    }

    if reason is not None and (q_type == "unanswerable" or answer == "unanswerable"):
        record["reason"] = str(reason)

    # Strictly omit alias field
    record.pop("alias", None)
    return record


def format_submission_json(
    q_id: str,
    answer: Any,
    q_type: Optional[str] = None,
    abstain_token: str = DEFAULT_ABSTAIN_TOKEN,
    reason: Optional[str] = None,
) -> str:
    """Format final dictionary into a single JSON line string."""
    record = format_submission_record(
        q_id=q_id,
        answer=answer,
        q_type=q_type,
        abstain_token=abstain_token,
        reason=reason,
    )
    return json.dumps(record, ensure_ascii=False)


def append_submission_line(
    output_file: Union[str, Path],
    record: Dict[str, Any],
) -> None:
    """Append a single JSON record to submission.jsonl file."""
    path = Path(output_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
