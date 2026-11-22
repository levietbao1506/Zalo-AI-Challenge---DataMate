"""Prefix-stable prompts for Adaptive Hybrid tabular question answering."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, Optional, Tuple, Union

from src.calendar_utils import resolve_time_context


ANSWER_TYPES = (
    "number",
    "percent",
    "string",
    "date",
    "boolean",
    "list",
    "table",
    "unanswerable",
    "abstain",
)

SYSTEM_PROMPT = """Bạn là DataMate Adaptive Hybrid Analyst. Không bịa bảng, cột, bản ghi, công thức hoặc kết quả. Mọi kết luận phải dựa trên dữ liệu thật và RULEBOOK.MD của split hiện tại.

QUY TẮC CỨNG — Penalized Accuracy: đúng +1, sai -1, abstain 0. Chỉ trả lời khi bằng chứng đủ mạnh; nếu chưa đủ căn cứ thì abstain an toàn.

OUTPUT CONTRACT: type chỉ thuộc number, percent, string, date, boolean, list, table, unanswerable, abstain.
- number: số nguyên hoặc số thực làm tròn tối đa 2 chữ số thập phân.
- percent: số điểm phần trăm đã nhân 100 và làm tròn 2 chữ số, ví dụ -12.45, không phải -0.1245.
- string: chuỗi NFC giữ nguyên dấu tiếng Việt từ dữ liệu gốc.
- date: YYYY-MM-DD hoặc YYYY-MM khi câu hỏi yêu cầu tháng.
- boolean: true hoặc false.
- list: mảng JSON có thứ tự.
- table: {"columns":["Tên"],"rows":[["Hà Nội"]]} với columns thẳng hàng rows.
- unanswerable: chỉ khi đã chứng minh thiếu dữ liệu/cột/khoảng thời gian bắt buộc; phải có reason ngắn.
- abstain: {"type":"abstain"} khi còn mơ hồ nghiêm trọng, xung đột hoặc không đủ độ tin cậy.

RANKING CONTRACT: top-k xếp metric DESC; đồng hạng xếp code ASC; nếu ít hơn k ứng viên hợp lệ thì lấy tất cả. Kiểm tra tie ở biên trước khi chốt.

RULEBOOK CONTRACT: đọc toàn văn RULEBOOK.MD động bên dưới. Không thay thế bằng quy tắc hardcode. Điều kiện ghi rõ trong câu hỏi được ưu tiên, sau đó mới áp dụng rulebook. Phải kiểm tra đơn hợp lệ, hủy/hoàn trả, VAT, voucher, as_of_date, làm tròn, dấu tăng/giảm và tie-break theo đúng rulebook.

Code-first: mọi phép lọc, đếm, tổng hợp và xếp hạng phải chạy trên dữ liệu thật.
Rulebook fidelity: doanh thu, lợi nhuận, giá vốn, bán chạy, trạng thái đơn và kỳ thời gian phải theo rulebook.
Pha Inspect: xác nhận bảng, cột, DISTINCT trạng thái, min/max ngày và tên gốc trước khi tính.
Pha Compute & Verify: kiểm tra dữ liệu rỗng, NaN, dấu, đơn vị, percent, top-k và tie-break.
Tiếng Việt: sp=sản phẩm, ko=không, dc=được, thg=tháng, sl=số lượng, dt=doanh thu, ln=lợi nhuận, kh=khách hàng. FILE_MAP và RAW_DATA_DIR chỉ được dùng ở nhánh Python.

Tương thích pipeline Python cũ: nếu được yêu cầu, gán final_result = "unanswerable" kèm reason hoặc final_result = "abstain". Không dùng unanswerable để che lỗi code."""

FEW_SHOT_EXAMPLES = ""


def read_rulebook(data_dir: Union[str, Path] = "./data") -> str:
    """Read the active split rulebook without falling back to another directory."""
    path = Path(data_dir) / "rulebook.md"
    if not path.is_file():
        return ""
    for encoding in ("utf-8-sig", "utf-8", "cp1258", "latin1"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeError:
            continue
    return ""


def build_business_prefix(rulebook_text: str, readme_text: str, sql_schema_card: str) -> str:
    """Build the byte-stable shared prefix used by every question of one business."""
    rulebook = rulebook_text if rulebook_text else "[Không có rulebook.md]"
    readme = readme_text if readme_text else "[Không có README.md]"
    schema = sql_schema_card if sql_schema_card else "[Không có SQL schema card]"
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== RULEBOOK.MD (ACTIVE SPLIT) ===\n{rulebook}\n\n"
        f"=== BUSINESS README.MD ===\n{readme}\n\n"
        f"=== BUSINESS SCHEMA PROFILE / SQL SCHEMA CARD ===\n{schema}"
    )


def build_sql_prompt(question: str, as_of_date: str, execution_mode: str = "SQL_FAST") -> str:
    """Build the dynamic Mode A suffix for one DuckDB SQL generation request."""
    return (
        f"MODE: {execution_mode}\n"
        "Viết đúng một truy vấn DuckDB SQL chỉ đọc trong một block ```sql```, sau đó đúng một block ```json``` metadata. "
        "Không viết giải thích ngoài hai block. Chỉ dùng SELECT hoặc WITH ... SELECT và các bảng trong SQL schema card.\n"
        "Dùng _calendar cho âm lịch, Tết và ngày lễ động. Dùng vn_unaccent hoặc vn_norm khi lọc tên sản phẩm, khách hàng hoặc thực thể tiếng Việt. "
        "Áp dụng bộ lọc đơn hợp lệ từ RULEBOOK.MD dựa trên các giá trị DISTINCT trong schema card. "
        "Ranking phải ORDER BY metric DESC, code ASC; nếu ít hơn k ứng viên thì trả hết. "
        "Percent phải nhân 100 và làm tròn 2 chữ số. Chỉ đặt is_unanswerable=true khi chắc chắn thiếu cột hoặc dữ liệu bắt buộc.\n"
        "Metadata bắt buộc đúng dạng:\n"
        "```json\n"
        '{"expected_type":"number","post_process":"none","is_unanswerable":false,"reason":null}\n'
        "```\n"
        "expected_type phải là một type trong OUTPUT CONTRACT.\n\n"
        f"QUESTION: {question.strip()}\n"
        f"as_of_date: {as_of_date or '[unknown]'}"
    )


def build_sql_repair_prompt(prev_sql: str, error_msg: str) -> str:
    """Build a compact single-retry SQL repair suffix."""
    error = error_msg.strip() or "DuckDB binder/syntax error"
    return (
        "MODE: SQL_REPAIR_ONCE\n"
        "Sửa đúng nguyên nhân binder/syntax bằng SQL schema card trong system prefix. Không đổi ý nghĩa câu hỏi. "
        "Chỉ trả một block ```sql``` và một block ```json``` metadata đúng contract Mode A.\n"
        f"PREVIOUS_SQL:\n```sql\n{prev_sql.strip()}\n```\n"
        f"DUCKDB_ERROR:\n{error[-1600:]}"
    )


def build_python_prompt(
    question: str,
    as_of_date: str,
    business_id: str,
    dual_path: bool = False,
    execution_mode: str = "PYTHON_FIRST",
) -> str:
    """Build the dynamic Mode B suffix for sandboxed Python computation."""
    independence = (
        "Bạn đang kiểm chứng Dual-Path độc lập. Không được xem, suy đoán hoặc yêu cầu câu SQL hay đáp án của nhánh SQL. Tự tính lại từ dữ liệu và rulebook."
        if dual_path
        else "Tự tính từ dữ liệu thật và kiểm chứng kết quả trước khi trả lời."
    )
    return (
        f"MODE: {execution_mode}\n"
        f"{independence}\n"
        "Chỉ trả đúng một block ```python``` chạy trong sandbox. Các biến đã có sẵn: DUCKDB_CONN là kết nối chỉ đọc, "
        f'DUCKDB_PATH="/tmp/datamate_duckdb/{business_id}.duckdb", RAW_DATA_DIR là thư mục dữ liệu gốc read-only, cùng pd, np và sqlite3. '
        "Ưu tiên DUCKDB_CONN; chỉ đọc lại CSV/XLSX/SQLite trong RAW_DATA_DIR khi cần kiểm tra ETL hoặc cấu trúc gốc.\n"
        "Code phải kiểm tra assert cơ bản cho cột bắt buộc, số dòng sau lọc, NaN, khoảng ngày và điều kiện rulebook. "
        "Không dùng mạng, không ghi file và không thay đổi dữ liệu.\n"
        "Gán kết quả cuối cùng đúng một lần vào FINAL_ANSWER={\"type\":...,\"answer\":...}. "
        "Nếu chứng minh thiếu dữ liệu, dùng FINAL_ANSWER={\"type\":\"unanswerable\",\"reason\":\"...\"}. "
        "Nếu không đủ chắc chắn, dùng FINAL_ANSWER={\"type\":\"abstain\"}.\n\n"
        f"QUESTION: {question.strip()}\n"
        f"as_of_date: {as_of_date or '[unknown]'}\n"
        f"business_id: {business_id}"
    )


def build_adjudicator_prompt(
    question: str,
    as_of_date: str,
    sql_query: str,
    sql_result: Any,
    python_code: str,
    python_stdout: str,
    python_result: Any,
) -> str:
    """Build the dynamic Mode C suffix used only for normalized answer conflicts."""
    return (
        "MODE: ADJUDICATOR_CONFLICT_ONLY\n"
        "Đối chiếu độc lập từng điều kiện trong RULEBOOK.MD: đơn hủy/hoàn trả, VAT, voucher, as_of_date, khoảng ngày, "
        "làm tròn, dấu âm/dương, percent và tie-breaker. Chỉ rõ nhánh sai ở bước nào. Có thể thêm tối đa một block ```python``` ngắn để xác minh. "
        "Kết thúc bằng đúng một block ```json``` có một trong ba kết luận: "
        '{"decision":"verified_answer","type":"...","answer":...,"reason":"..."}, '
        '{"decision":"unanswerable","type":"unanswerable","reason":"..."}, hoặc '
        '{"decision":"abstain","type":"abstain","reason":"..."}. '
        "Chỉ chọn verified_answer khi bằng chứng chắc chắn; nếu vẫn xung đột hoặc thiếu căn cứ thì abstain để tránh phạt -1.\n\n"
        f"QUESTION: {question.strip()}\n"
        f"as_of_date: {as_of_date or '[unknown]'}\n\n"
        f"SQL_BRANCH_QUERY:\n```sql\n{sql_query.strip()}\n```\n"
        f"SQL_BRANCH_RESULT:\n{_json_dump(sql_result)}\n\n"
        f"PYTHON_BRANCH_CODE:\n```python\n{python_code.strip()}\n```\n"
        f"PYTHON_BRANCH_STDOUT:\n{python_stdout.strip()}\n"
        f"PYTHON_BRANCH_RESULT:\n{_json_dump(python_result)}"
    )


def parse_sql_response(response: str) -> Tuple[str, Dict[str, Any]]:
    cleaned = _remove_think_blocks(response)
    sql_blocks = re.findall(r"```sql\s*\n?(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if not sql_blocks:
        sql_blocks = re.findall(r"```(?:duckdb)?\s*\n?(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if len(sql_blocks) != 1:
        raise ValueError(f"Expected exactly one sql block, found {len(sql_blocks)}")
    sql = sql_blocks[0].strip()
    if not sql:
        raise ValueError("The sql block is empty")

    json_blocks = re.findall(r"```json\s*\n?(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if len(json_blocks) > 1:
        raise ValueError(f"Expected exactly one json metadata block, found {len(json_blocks)}")
    if not json_blocks:
        metadata = {
            "expected_type": "number",
            "post_process": "scalar",
            "is_unanswerable": False,
            "reason": None,
        }
    else:
        try:
            metadata = json.loads(json_blocks[0].strip())
        except Exception as exc:
            raise ValueError("Invalid SQL metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("SQL metadata must be an object")
    metadata.setdefault("expected_type", "number")
    metadata.setdefault("post_process", "scalar")
    metadata.setdefault("is_unanswerable", False)
    metadata.setdefault("reason", None)
    if metadata["expected_type"] not in ANSWER_TYPES:
        raise ValueError("SQL metadata expected_type is invalid")
    if metadata["is_unanswerable"] and not str(metadata.get("reason") or "").strip():
        raise ValueError("Unanswerable SQL metadata requires reason")
    return sql, metadata


def parse_python_response(response: str) -> str:
    cleaned = _remove_think_blocks(response)
    blocks = re.findall(r"```python\s*\n?(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if not blocks:
        blocks = re.findall(r"```\s*\n?(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly one python block, found {len(blocks)}")
    if not blocks[0].strip():
        raise ValueError("The python block is empty")
    return blocks[0].strip()


def parse_json_response(response: str) -> Dict[str, Any]:
    cleaned = _remove_think_blocks(response)
    blocks = re.findall(r"```json\s*\n?(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if not blocks:
        return {}
    if len(blocks) != 1:
        raise ValueError(f"Expected exactly one json block, found {len(blocks)}")
    try:
        parsed = json.loads(blocks[0].strip())
    except Exception as exc:
        raise ValueError("Invalid adjudicator JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Adjudicator JSON must be an object")
    decision = parsed.get("decision")
    if decision not in {"verified_answer", "unanswerable", "abstain"}:
        raise ValueError("Adjudicator decision is invalid")
    if decision == "unanswerable" and not str(parsed.get("reason") or "").strip():
        raise ValueError("Unanswerable adjudicator output requires reason")
    return parsed


class PromptBuilder:
    """Construct cached business prefixes and per-question Adaptive Hybrid suffixes."""

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
        self._prefix_cache: Dict[Tuple[str, str, str, str, str], str] = {}

    def build_system_prompt(
        self,
        rulebook_content: Optional[str] = None,
        readme_content: str = "",
        schema_context: str = "",
        business_id: Optional[str] = None,
    ) -> str:
        """Return the cached byte-stable prefix for a business."""
        rulebook = self.rulebook_content if rulebook_content is None else rulebook_content
        key = (business_id or "", self.system_prompt, rulebook, readme_content or "", schema_context or "")
        cached = self._prefix_cache.get(key)
        if cached is not None:
            return cached
        if self.system_prompt == SYSTEM_PROMPT:
            prefix = build_business_prefix(rulebook, readme_content, schema_context)
        else:
            prefix = (
                f"{self.system_prompt}\n\n"
                f"=== RULEBOOK.MD (ACTIVE SPLIT) ===\n{rulebook}\n\n"
                f"=== BUSINESS README.MD ===\n{readme_content}\n\n"
                f"=== BUSINESS SCHEMA PROFILE / SQL SCHEMA CARD ===\n{schema_context}"
            )
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
        """Keep the legacy Python pipeline suffix dynamic and prefix-safe."""
        rulebook = self.rulebook_content if rulebook_content is None else rulebook_content
        if time_context is None and as_of_date:
            try:
                time_context = resolve_time_context(as_of_date, rulebook_text=rulebook)
            except (TypeError, ValueError):
                time_context = {}
        resolved_time_context = time_context or {}
        anchors = json.dumps(resolved_time_context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        interval_lines = []
        if resolved_time_context:
            last_week = resolved_time_context.get("last_week", {})
            this_month = resolved_time_context.get("this_month", {})
            interval_lines = [
                "PRE-COMPUTED TIME INTERVALS:",
                f"- Tuần trước / Tuần rồi: {last_week.get('start', '')}..{last_week.get('end', '')}",
                f"- Tháng này (Month-to-Date): {this_month.get('start', '')}..{this_month.get('end', '')}",
            ]
        intervals = "\n".join(interval_lines)
        return (
            f"CÂU HỎI: {question.strip()}\n"
            f"as_of_date: {as_of_date or '[chưa xác định]'}\n"
            f"TIME_CONTEXT_ISO: {anchors}\n"
            f"{intervals}\n"
            "Kiểm tra dữ liệu thật rồi viết code gán final_result."
        )

    def build_sql_prompt(self, question: str, as_of_date: str, execution_mode: str = "SQL_FAST") -> str:
        return build_sql_prompt(question, as_of_date, execution_mode)

    def build_sql_repair_prompt(self, prev_sql: str, error_msg: str) -> str:
        return build_sql_repair_prompt(prev_sql, error_msg)

    def build_python_prompt(
        self,
        question: str,
        as_of_date: str,
        business_id: str,
        dual_path: bool = False,
        execution_mode: str = "PYTHON_FIRST",
    ) -> str:
        return build_python_prompt(question, as_of_date, business_id, dual_path, execution_mode)

    def build_adjudicator_prompt(
        self,
        question: str,
        as_of_date: str,
        sql_query: str,
        sql_result: Any,
        python_code: str,
        python_stdout: str,
        python_result: Any,
    ) -> str:
        return build_adjudicator_prompt(
            question,
            as_of_date,
            sql_query,
            sql_result,
            python_code,
            python_stdout,
            python_result,
        )

    def build_retry_prompt(
        self,
        question: str,
        previous_code: str,
        error_traceback: str,
        previous_user_prompt: Optional[str] = None,
    ) -> str:
        """Build the legacy Python retry suffix without changing the shared prefix."""
        error = error_traceback.strip() or "Output rỗng/NaN hoặc không có final_result"
        base = previous_user_prompt or f"CÂU HỎI: {question.strip()}"
        return (
            f"{base}\nRETRY: Code trước lỗi hoặc trả rỗng/NaN.\n"
            f"FAILED CODE:\n```python\n{previous_code}\n```\n"
            f"EXACT ERROR/OUTPUT:\n{error[-1800:]}\n"
            "Sửa nguyên nhân gốc: kiểm tra lại tên file/cột, dtype, trạng thái, ngày, bộ lọc và mẫu dữ liệu; không lặp lại cùng code. "
            "Nếu dữ liệu bắt buộc thật sự thiếu, gán reason và final_result = \"unanswerable\"; nếu vẫn không chắc, gán final_result = \"abstain\". "
            "Chỉ trả một block ```python```."
        )

    def build_verification_prompt(
        self,
        question: str,
        computed_answer: Any,
        as_of_date: Optional[str] = None,
        time_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """Build the optional legacy verification suffix."""
        if not self.enable_verification:
            return None
        return (
            f"VERIFY QUESTION: {question.strip()}\n"
            f"as_of_date: {as_of_date or '[chưa xác định]'}\n"
            f"TIME_CONTEXT_ISO: {json.dumps(time_context or {}, ensure_ascii=False, sort_keys=True)}\n"
            f"COMPUTED ANSWER: {json.dumps(computed_answer, ensure_ascii=False, default=str)}\n"
            "Đối chiếu câu hỏi, dữ liệu và RULEBOOK.MD trong system message. Kiểm tra type, đơn vị, dấu, hủy/trả và tie-break. "
            "Chỉ viết code kiểm chứng; nếu chưa giải được mâu thuẫn, final_result = \"abstain\"."
        )


def _remove_think_blocks(response: str) -> str:
    text = str(response or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _extract_single_fenced_block(response: str, language: str) -> str:
    pattern = rf"```{re.escape(language)}\s*\n?(.*?)```"
    matches = re.findall(pattern, response, flags=re.IGNORECASE | re.DOTALL)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {language} block, found {len(matches)}")
    content = matches[0].strip()
    if not content:
        raise ValueError(f"The {language} block is empty")
    return content


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True, separators=(",", ":"))
