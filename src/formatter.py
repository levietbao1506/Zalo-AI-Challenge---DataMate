"""Output Normalizer & Submission Formatter for Zalo AI Challenge 2026 - DataMate Track (Bước 5).

Normalizes model outputs and Pandas query results per competition rules:
1. Chuẩn hóa String: NFC Unicode, đưa về chữ thường, thay ký tự đặc biệt bằng khoảng trắng, xóa khoảng trắng thừa.
2. Chuẩn hóa Number / Percent: Ép kiểu float, làm tròn đúng 2 chữ số thập phân (round(val, 2)).
3. Quyết định Abstain: Nếu chạy quá 3 lần vẫn lỗi hoặc trả về DataFrame rỗng vô lý -> gán answer = "abstain".
4. Format dictionary thành chuỗi JSON chứa id, type, answer và ghi append vào file submission.jsonl.
"""

from __future__ import annotations

from datetime import date, datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Union
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


def clean_string(text: str) -> str:
    """Chuẩn hóa String theo luật BTC:

    1. Dùng unicodedata.normalize('NFC', text).
    2. Đưa về chữ thường (.lower()).
    3. Thay ký tự đặc biệt bằng khoảng trắng (loại bỏ punctuation, giữ lại chữ và số kể cả tiếng Việt).
    4. Xóa khoảng trắng thừa.

    Args:
        text: Input string.

    Returns:
        Normalized and cleaned Vietnamese string.
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. NFC normalization
    text = unicodedata.normalize("NFC", text.strip())

    # 2. Lowercase
    text = text.lower()

    # 3. Replace special characters with spaces (keep unicode words \w and spaces, replace punctuation and _)
    text = re.sub(r"[^\w\s]|_", " ", text)

    # 4. Collapse multiple spaces into single space and strip
    return re.sub(r"\s+", " ", text).strip()


def format_number(val: Any, round_decimals: int = 2) -> Union[int, float]:
    """Chuẩn hóa Number/Percent: ép kiểu float và làm tròn đúng 2 chữ số thập phân.

    Hỗ trợ cả chuỗi phần trăm (e.g. '15.75%') và số thực.

    Args:
        val: Numeric value or string representation.
        round_decimals: Number of decimal places (default: 2).

    Returns:
        Rounded float or simplified integer.
    """
    if isinstance(val, str):
        cleaned = val.strip()
        # Handle percentage strings (e.g. '15.5%' or '25.0 %')
        if m_pct := re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*%\s*$", cleaned):
            num = round(float(m_pct.group(1)), round_decimals)
            return int(num) if num.is_integer() else num
        # Handle standard numeric strings
        try:
            num = round(float(cleaned), round_decimals)
            return int(num) if num.is_integer() else num
        except ValueError:
            raise ValueError(f"Cannot parse '{val}' as number.")

    num = round(float(val), round_decimals)
    return int(num) if num.is_integer() else num


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
            if val.empty:
                return True
        if pd.isna(val):
            return True

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
) -> Any:
    """Format and normalize a scalar value per competition rules.

    Args:
        val: Input scalar value.
        round_decimals: Number of decimal places to round floats to (default: 2).
        abstain_fallback: Value to return on unanswerable or invalid result (default: 'abstain').
        clean_text: If True, applies full clean_string() (lowercase, remove special chars).

    Returns:
        Normalized value (int, float, date string, or normalized text string).
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

    # Floats: làm tròn đúng 2 chữ số thập phân
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return abstain_fallback
        rounded = round(val, round_decimals)
        return int(rounded) if rounded.is_integer() else rounded

    # Strings
    if isinstance(val, str):
        cleaned = val.strip()
        # Remove accidental surrounding quotes
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()

        # Handle percent strings (e.g. '15.5%')
        if m_pct := re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*%\s*$", cleaned):
            num = round(float(m_pct.group(1)), round_decimals)
            return int(num) if num.is_integer() else num

        # Handle numeric strings
        try:
            if re.match(r"^-?\d+$", cleaned):
                return int(cleaned)
            if re.match(r"^-?\d+\.\d+$", cleaned):
                num = round(float(cleaned), round_decimals)
                return int(num) if num.is_integer() else num
        except Exception:
            pass

        # Handle ISO Date string preservation (e.g. '2024-05-15')
        if re.match(r"^\d{4}-\d{2}-\d{2}(?: \d{2}:\d{2}:\d{2})?$", cleaned):
            return cleaned

        # Full text cleaning (NFC, lowercase, replace special chars, collapse spaces)
        if clean_text:
            return clean_string(cleaned)
        return normalize_unicode_nfc(cleaned)

    # Lists / Tuples
    if isinstance(val, (list, tuple, set)):
        normalized_list = [format_scalar(item, round_decimals, abstain_fallback, clean_text) for item in val]
        if not normalized_list:
            return abstain_fallback
        return normalized_list

    # Dictionaries
    if isinstance(val, dict):
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

    Returns:
        Formatted answer per BTC rules.
    """
    if is_abstain_signal(val, success=success):
        return abstain_token

    return format_scalar(
        val=val,
        round_decimals=round_decimals,
        abstain_fallback=abstain_token,
        clean_text=clean_text,
    )


def format_submission_record(
    q_id: str,
    answer: Any,
    q_type: Optional[str] = None,
    abstain_token: str = DEFAULT_ABSTAIN_TOKEN,
) -> Dict[str, Any]:
    """Format dictionary cuối cùng chứa id, type, answer theo luật BTC.

    Args:
        q_id: Question ID.
        answer: Normalized answer value.
        q_type: Question type (optional, auto-inferred if None).
        abstain_token: Token indicating an abstain response.

    Returns:
        Dictionary {'id': ..., 'type': ..., 'answer': ...}.
    """
    if q_type is None:
        if answer == abstain_token or answer is None:
            q_type = "abstain"
        elif isinstance(answer, bool):
            q_type = "boolean"
        elif isinstance(answer, (int, float)):
            q_type = "number"
        elif isinstance(answer, (list, tuple)):
            q_type = "list"
        elif isinstance(answer, dict):
            q_type = "table"
        else:
            q_type = "string"

    return {
        "id": str(q_id),
        "type": str(q_type),
        "answer": answer,
    }


def format_submission_json(
    q_id: str,
    answer: Any,
    q_type: Optional[str] = None,
    abstain_token: str = DEFAULT_ABSTAIN_TOKEN,
) -> str:
    """Format final dictionary into a single JSON line string."""
    record = format_submission_record(q_id=q_id, answer=answer, q_type=q_type, abstain_token=abstain_token)
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
