"""Compact, split-specific prompts for offline tabular question answering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.calendar_utils import resolve_time_context


SYSTEM_PROMPT = """Bạn là Data Analyst viết Python để trả lời câu hỏi từ dữ liệu thật. Chỉ xuất một khối ```python ... ```; gán kết quả vào final_result. Không tính nhẩm, không bịa cột, bảng, bản ghi hay kết quả.

QUY TẮC CỨNG — Penalized Accuracy: đúng +1, sai -1, abstain 0. Chỉ trả lời khi bằng chứng đủ mạnh (>60% khả năng đúng).
1. Code-first: mọi đếm, lọc, tổng hợp, xếp hạng và tra cứu tên sản phẩm/khách hàng phải chạy bằng Python trên file thật trong FILE_MAP. pd, np, sqlite3, calendar_utils và as_of_date có sẵn.
2. Rulebook fidelity: doanh thu, lợi nhuận, giá vốn, bán chạy, đơn hợp lệ, đơn hủy/trả, kỳ thời gian, ranking và tie-break theo mã tăng dần phải theo rulebook của split; điều kiện nêu rõ trong câu hỏi được ưu tiên. Không tự đặt công thức doanh thu.
3. Pha Inspect: kiểm tra file/cột bắt buộc; xem giá trị unique và tần suất cột trạng thái/danh mục; đối chiếu tên gốc, tên không dấu và viết tắt; kiểm tra min/max ngày thực tế. Chỉ dùng các cột đã xác nhận.
4. Pha Compute & Verify: áp đúng bộ lọc rulebook; chặn ngày sau as_of_date khi câu hỏi neo hiện tại; xét hủy/trả; xử lý rỗng/NaN; kiểm tra tie top-k và mã tăng dần; percent là số điểm phần trăm (-12.45, không phải -0.1245), đúng dấu tăng/giảm. Đếm số đơn khác tổng số lượng bán và khác doanh thu.
5. unanswerable: chỉ sau khi code kiểm tra dữ liệu/README và chứng minh thiếu trường, thiếu khoảng thời gian hoặc không thể tính; ghi `reason = "..."` ngắn bằng tiếng Việt và `final_result = "unanswerable"`. Ví dụ lợi nhuận thiếu giá vốn. Dữ liệu rỗng sau lọc chưa tự chứng minh unanswerable.
6. abstain: câu hỏi có thể trả lời nhưng còn mơ hồ nghiêm trọng, công thức xung đột, code lỗi hoặc xác suất đúng không vượt 60%; dùng `final_result = "abstain"`. Không dùng unanswerable để che lỗi code.
7. Tiếng Việt: sp=sản phẩm, ko=không, dc=được, thg=tháng, sl=số lượng, dt=doanh thu, ln=lợi nhuận, kh=khách hàng. Tìm kiếm linh hoạt với tên không dấu nhưng trả string/list/table bằng tên gốc có dấu, NFC.

OUTPUT CONTRACT: code chỉ gán final_result; bộ đóng gói xuất JSON {"id":"...","type":"...","answer":...}. type thuộc number, percent, string, date, boolean, list, table, unanswerable, abstain. number là số nguyên hoặc số làm tròn 2 chữ số; percent=-12.45; date="2024-06-19" hoặc "2024-06"; boolean=true/false; list=["A",2] đúng thứ tự; table={"columns":["Tên"],"rows":[["Hà Nội"]]}; unanswerable={"type":"unanswerable","answer":"unanswerable"} với reason trong code; abstain={"type":"abstain","answer":"abstain"}. Không in JSON thay cho code.

Trước khi truy cập `.iloc[0]`, kiểm tra rỗng. Nếu Excel có metadata trên header, dùng đúng dòng header từ schema. Trả đúng một khối Python, không có văn bản ngoài khối."""

FEW_SHOT_EXAMPLES = ""  # Avoid examples with invented columns or split-specific status labels.


def read_rulebook(data_dir: Union[str, Path] = "./data") -> str:
    """Read the active split only, without falling back to another directory."""
    path = Path(data_dir) / "rulebook.md"
    if not path.is_file():
        return ""
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeError:
            continue
    return ""


class PromptBuilder:
    """Construct immutable per-business prefixes and compact query suffixes."""

    def __init__(
        self,
        system_prompt: str = SYSTEM_PROMPT,
        few_shot_examples: str = FEW_SHOT_EXAMPLES,
        rulebook_content: Optional[str] = None,
        enable_verification: bool = False,
    ) -> None:
        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples
        self.rulebook_content = rulebook_content or ""
        self.enable_verification = enable_verification
        self._prefix_cache: Dict[tuple[str, str, str, str], str] = {}

    def build_system_prompt(
        self,
        rulebook_content: Optional[str] = None,
        readme_content: str = "",
        schema_context: str = "",
        business_id: Optional[str] = None,
    ) -> str:
        """Place stable data after instructions in the same order for every query."""
        rb = self.rulebook_content if rulebook_content is None else rulebook_content
        schema_profile = schema_context[:30000] + ("\n[Schema context budget reached]" if len(schema_context) > 30000 else "")
        key = (business_id or "", rb, readme_content or "", schema_profile)
        cached = self._prefix_cache.get(key)
        if cached is not None:
            return cached
        sections = [
            self.system_prompt.strip(),
            self.few_shot_examples.strip(),
            "=== RULEBOOK.MD (ACTIVE SPLIT) ===\n" + (rb.strip() or "[Không có rulebook]"),
            "=== BUSINESS README.MD ===\n" + (readme_content.strip() or "[Không có README]"),
            "=== BUSINESS SCHEMA PROFILE ===\n" + (schema_profile.strip() or "[Không có schema]"),
        ]
        prefix = "\n\n".join(section for section in sections if section)
        self._prefix_cache[key] = prefix
        return prefix

    def build_user_prompt(
        self,
        question: str,
        schema_context: str = "",
        as_of_date: Optional[str] = None,
        file_map: Optional[Dict[str, str]] = None,
        rulebook_content: Optional[str] = None,
        time_context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Keep only per-question values in the user suffix.

        Schema and rulebook parameters remain for old callers. The caller puts
        their content in build_system_prompt to enable prefix caching.
        """
        rb = self.rulebook_content if rulebook_content is None else rulebook_content
        if time_context is None and as_of_date:
            try:
                time_context = resolve_time_context(as_of_date, rulebook_text=rb)
            except (TypeError, ValueError):
                time_context = {}
        anchors = json.dumps(time_context or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (
            f"CÂU HỎI: {question.strip()}\n"
            f"as_of_date: {as_of_date or '[chưa xác định]'}\n"
            f"TIME_CONTEXT_ISO: {anchors}\n"
            "Kiểm tra dữ liệu thật rồi viết code gán final_result."
        )

    def build_retry_prompt(
        self,
        question: str,
        previous_code: str,
        error_traceback: str,
        previous_user_prompt: Optional[str] = None,
    ) -> str:
        """Give concrete failure evidence and demand a changed repair strategy."""
        error = error_traceback.strip() or "Output rỗng/NaN hoặc không có final_result"
        base = previous_user_prompt or f"CÂU HỎI: {question.strip()}"
        return (
            f"{base}\nRETRY: Code trước lỗi hoặc trả rỗng/NaN.\n"
            f"FAILED CODE:\n```python\n{previous_code}\n```\n"
            f"EXACT ERROR/OUTPUT:\n{error[-1800:]}\n"
            "Sửa nguyên nhân gốc: kiểm tra lại tên file/cột, dtype, giá trị trạng thái, định dạng ngày, bộ lọc và mẫu dữ liệu; không lặp lại cùng code. "
            "Nếu cột/kỳ thật sự thiếu sau kiểm tra, đặt reason tiếng Việt và final_result = \"unanswerable\"; "
            "nếu vẫn không chắc hoặc code tiếp tục lỗi, đặt final_result = None để abstain. "
            "Chỉ trả một khối ```python ... ``` gán final_result."
        )

    def build_verification_prompt(
        self,
        question: str,
        computed_answer: Any,
        as_of_date: Optional[str] = None,
        time_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Optional follow-up prompt; the caller decides whether to invoke a model."""
        if not self.enable_verification:
            return None
        return (
            f"VERIFY QUESTION: {question.strip()}\n"
            f"as_of_date: {as_of_date or '[chưa xác định]'}\n"
            f"TIME_CONTEXT_ISO: {json.dumps(time_context or {}, ensure_ascii=False, sort_keys=True)}\n"
            f"COMPUTED ANSWER: {json.dumps(computed_answer, ensure_ascii=False, default=str)}\n"
            "Đối chiếu câu hỏi, dữ liệu, khoảng ngày và RULEBOOK.MD trong system message. "
            "Kiểm tra type, đơn vị, dấu, lọc hủy/trả, tie-break. "
            "Chỉ viết code kiểm chứng; nếu mâu thuẫn chưa giải được, final_result = \"abstain\"."
        )
