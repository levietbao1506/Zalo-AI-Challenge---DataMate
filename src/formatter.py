"""Validate and serialize DataMate submission values by their declared type."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
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
    capitalization, punctuation, identifiers, and diacritics from the business table.
    When clean_text is True (for comparison), applies full clean_string normalization.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    cleaned = text.strip()

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
    if isinstance(val, bool) or val is None:
        raise ValueError("A number must be a finite numeric value")
    if _HAS_NUMPY and isinstance(val, (np.integer, np.floating)):
        val = val.item()
    if isinstance(val, int):
        return val
    if isinstance(val, float):
        if not math.isfinite(val):
            raise ValueError("NaN and Infinity are invalid")
        raw = str(val)
    elif isinstance(val, Decimal):
        raw = str(val)
    elif isinstance(val, str):
        raw = val.strip()
        currency_pattern = r"(?i)(?:\s*(?:vnd|vnđ|vn₫|usd|đồng|dong|[₫đ])\s*$|^\s*\$\s*)"
        currency = bool(re.search(currency_pattern, raw))
        raw = re.sub(currency_pattern, "", raw)
        raw = re.sub(r"\s+", "", raw)
        if "%" in raw:
            raise ValueError("Percent symbol is invalid for number")
        if "," in raw and "." in raw:
            decimal_mark = "," if raw.rfind(",") > raw.rfind(".") else "."
            thousands_mark = "." if decimal_mark == "," else ","
            raw = raw.replace(thousands_mark, "").replace(decimal_mark, ".")
        elif re.fullmatch(r"[+-]?\d{1,3}(?:[.,]\d{3}){2,}(?:[.,]\d{1,2})?", raw):
            last = raw.rsplit(",", 1)[-1].rsplit(".", 1)[-1]
            has_decimal = len(last) <= 2 and ("," in raw or "." in raw)
            decimal_mark = raw[-len(last)-1] if has_decimal else None
            if has_decimal:
                whole, fraction = raw.rsplit(decimal_mark, 1)
                raw = re.sub(r"[.,]", "", whole) + "." + fraction
            else:
                raw = re.sub(r"[.,]", "", raw)
        elif "," in raw:
            raw = raw.replace(",", "" if re.fullmatch(r"[+-]?\d{1,3},\d{3}", raw) else ".")
        elif "." in raw and currency and re.fullmatch(r"[+-]?\d{1,3}\.\d{3}", raw):
            raw = raw.replace(".", "")
    else:
        raise ValueError(f"Cannot parse {type(val).__name__} as number")
    try:
        numeric = Decimal(raw)
        if not numeric.is_finite():
            raise ValueError("NaN and Infinity are invalid")
        rounded = numeric.quantize(Decimal(1).scaleb(-round_decimals), rounding=ROUND_HALF_UP)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid number: {val!r}") from exc
    if rounded == 0:
        return 0
    if rounded == rounded.to_integral_value():
        return int(rounded)
    return float(rounded)


def format_percent(val: Any, round_decimals: int = 2) -> Union[int, float]:
    """Strictly format numeric percentage value per evaluation rules.

    Outputs numeric percentage value (e.g. -12.45, NOT '-12.45%' and NOT -0.1245),
    rounded to 2 decimal places with exact sign.

    Args:
        val: Numeric percentage or string percentage such as '-12.45%'.
        round_decimals: Decimal places to round to (default: 2).

    Returns:
        Rounded float or integer percentage value.
    """
    if isinstance(val, str):
        cleaned = val.strip()
        if cleaned.endswith("%"):
            cleaned = cleaned[:-1].strip()
        return format_number(cleaned, round_decimals)
    return format_number(val, round_decimals)


def format_date(val: Any, as_month: bool = False) -> str:
    """Strictly format date as ISO string 'YYYY-MM-DD' (or 'YYYY-MM' when month requested).

    Args:
        val: Date, datetime, Timestamp, or date string.
        as_month: If True, formats as 'YYYY-MM'.

    Returns:
        Clean ISO formatted date string.
    """
    if val is None or (_HAS_PANDAS and val is pd.NaT):
        raise ValueError("Missing date")
    if isinstance(val, datetime):
        parsed = val.date()
    elif isinstance(val, date):
        parsed = val
    else:
        raw = str(val).strip()
        if re.fullmatch(r"\d{4}-\d{2}", raw):
            date.fromisoformat(raw + "-01")
            return raw
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?: 00:00:00)?", raw):
            parsed = date.fromisoformat(raw[:10])
        else:
            raise ValueError(f"Invalid ISO date: {val!r}")
    return parsed.strftime("%Y-%m") if as_month else parsed.isoformat()


def format_boolean(val: Any) -> bool:
    """Strictly format value as JSON boolean True or False.

    Args:
        val: Boolean or truthy/falsy scalar/string.

    Returns:
        Python bool (serializes to JSON true or false).
    """
    if isinstance(val, bool):
        return val
    if _HAS_NUMPY and isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, (int, float)) and val in (0, 1):
        return bool(val)
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "t", "yes", "đúng", "dung", "có", "co"):
            return True
        if s in ("false", "0", "f", "no", "sai", "không", "khong"):
            return False
    raise ValueError(f"Invalid boolean: {val!r}")


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
    elif isinstance(val, tuple):
        val = list(val)
    elif isinstance(val, str):
        val = json.loads(val)
    if not isinstance(val, list) or not val:
        raise ValueError("List must be a nonempty ordered array")
    return [_format_cell(item, round_decimals) for item in val]


def _format_cell(value: Any, round_decimals: int = 2) -> Union[str, int, float, bool]:
    """Normalize a table/list cell without turning text identifiers into numbers."""
    if value is None:
        raise ValueError("Null cell")
    if _HAS_NUMPY and isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, Decimal)):
        return format_number(value, round_decimals)
    if isinstance(value, str):
        text = format_string(value)
        if not text or text.casefold() in {"nan", "none", "null", "inf", "infinity"}:
            raise ValueError("Empty or non-finite cell")
        numeric_text = re.fullmatch(r"[+-]?(?:\d+(?:\.\d+)?|\d{1,3}(?:[.,]\d{3})+(?:[.,]\d{1,2})?)", text)
        currency_text = re.search(r"(?i)(?:vnd|vnđ|vn₫|đồng|dong|[₫đ])\s*$|^\s*\$", text)
        leading_zero_code = bool(re.fullmatch(r"[+-]?0\d+", text))
        if (numeric_text or currency_text) and not leading_zero_code:
            return format_number(text, round_decimals)
        return text
    if isinstance(value, (date, datetime)):
        return format_date(value)
    raise ValueError(f"Unsupported cell type: {type(value).__name__}")


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
        columns, rows = list(val.columns), list(val.itertuples(index=False, name=None))
    elif isinstance(val, dict) and set(val) == {"columns", "rows"}:
        columns, rows = val["columns"], val["rows"]
    elif isinstance(val, dict) and val and all(isinstance(v, (list, tuple)) for v in val.values()):
        columns = list(val)
        lengths = {len(v) for v in val.values()}
        if len(lengths) != 1:
            raise ValueError("Table columns have unequal lengths")
        rows = [list(row) for row in zip(*val.values())]
    elif isinstance(val, list) and val and all(isinstance(row, dict) for row in val):
        columns = list(val[0])
        if any(set(row) != set(columns) for row in val):
            raise ValueError("Table records have unequal columns")
        rows = [[row[column] for column in columns] for row in val]
    else:
        raise ValueError("Table must contain columns and rows")
    if not isinstance(columns, (list, tuple)) or not columns or not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("Table columns and rows are required")
    normalized_columns = [format_string(column) for column in columns]
    if any(not column for column in normalized_columns):
        raise ValueError("Empty table column name")
    normalized_rows = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != len(normalized_columns):
            raise ValueError("Table row width does not match columns")
        normalized_rows.append([_format_cell(cell, round_decimals) for cell in row])
    return {"columns": normalized_columns, "rows": normalized_rows}


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
        try:
            return format_table(val, round_decimals=round_decimals, clean_text=clean_text)
        except (ValueError, TypeError):
            return abstain_fallback

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
        try:
            return format_boolean(val)
        except ValueError:
            return abstain_fallback

    if q_type == "list":
        try:
            return format_list(val, round_decimals=round_decimals, clean_text=clean_text)
        except (ValueError, TypeError, json.JSONDecodeError):
            return abstain_fallback

    if q_type == "string":
        return format_string(val, clean_text=clean_text)

    # Inferred types
    if isinstance(val, (datetime, date)) or (_HAS_PANDAS and isinstance(val, pd.Timestamp)):
        return format_date(val)

    if isinstance(val, bool):
        return val

    if isinstance(val, int):
        return val

    if isinstance(val, float):
        try:
            return format_number(val, round_decimals)
        except ValueError:
            return abstain_fallback

    if isinstance(val, str):
        cleaned = val.strip()

        # Handle percent strings (e.g. '15.5%')
        if m_pct := re.match(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%\s*$", cleaned):
            return format_percent(m_pct.group(1), round_decimals)

        # Handle numeric strings
        try:
            if re.match(r"^[+-]?\d+$", cleaned):
                return int(cleaned)
            if re.match(r"^[+-]?\d+\.\d+$", cleaned):
                return format_number(cleaned, round_decimals)
        except Exception:
            pass

        # Handle ISO Date string preservation (e.g. '2024-05-15')
        if re.match(r"^\d{4}-\d{2}-\d{2}(?: 00:00:00)?$", cleaned):
            try:
                return format_date(cleaned)
            except ValueError:
                return abstain_fallback

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
    """Return a valid typed record, or ``{id, type: abstain}`` on invalid input."""
    record_id = str(q_id)
    abstain = {"id": record_id, "type": "abstain"}
    if isinstance(answer, dict) and answer.get("type") == "unanswerable":
        reason = answer.get("reason", reason)
        answer = None
        q_type = "unanswerable"
    if q_type == "abstain":
        return abstain
    if q_type == "unanswerable" or (isinstance(answer, str) and answer.strip().casefold() == "unanswerable"):
        valid_marker = answer is None or (isinstance(answer, str) and answer.strip().casefold() == "unanswerable")
        if not isinstance(reason, str) or not reason.strip() or not valid_marker:
            return abstain
        return {"id": record_id, "type": "unanswerable", "reason": format_string(reason)}
    if answer is None or (
        isinstance(answer, str) and answer.strip().casefold() in {abstain_token.casefold(), "abstain"}
    ):
        return abstain

    if q_type is None:
        if isinstance(answer, bool):
            q_type = "boolean"
        elif isinstance(answer, (int, float, Decimal)):
            q_type = "number"
        elif isinstance(answer, (list, tuple)):
            q_type = "list"
        elif isinstance(answer, dict) and "columns" in answer and "rows" in answer:
            q_type = "table"
        elif _HAS_PANDAS and isinstance(answer, pd.DataFrame):
            q_type = "table"
        else:
            q_type = "string"
    try:
        if q_type == "number":
            normalized = format_number(answer)
        elif q_type == "percent":
            normalized = format_percent(answer)
        elif q_type == "string":
            if not isinstance(answer, str):
                return abstain
            normalized = format_string(answer)
            if not normalized or normalized.casefold() in {"nan", "none", "null", "inf", "infinity"}:
                return abstain
        elif q_type == "date":
            normalized = format_date(answer)
        elif q_type == "boolean":
            normalized = format_boolean(answer)
        elif q_type == "list":
            normalized = format_list(answer)
        elif q_type == "table":
            normalized = format_table(answer)
        else:
            return abstain
        record = {"id": record_id, "type": q_type, "answer": normalized}
        json.dumps(record, ensure_ascii=False, allow_nan=False)
        return record
    except (ValueError, TypeError, OverflowError, InvalidOperation):
        return abstain


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
    return json.dumps(record, ensure_ascii=False, allow_nan=False)


def append_submission_line(
    output_file: Union[str, Path],
    record: Dict[str, Any],
) -> None:
    """Append a single JSON record to submission.jsonl file."""
    if not isinstance(record, dict) or "id" not in record:
        raise ValueError("Submission record needs an id")
    safe_record = format_submission_record(
        record["id"], record.get("answer"), q_type=record.get("type"),
        reason=record.get("reason"),
    )
    path = Path(output_file).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        try:
            import os
            os.fsync(f.fileno())
        except OSError:
            pass
