"""Compact, split-specific prompts for offline tabular question answering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Union


SYSTEM_PROMPT = """Bạn là một Data Analyst chuyên nghiệp. Hãy viết mã Python (dùng thư viện pandas) để trả lời câu hỏi sau. Chỉ in ra code, không giải thích.

MÔI TRƯỜNG THỰC THI (SANDBOX ENVIRONMENT):
1. Mã của bạn sẽ được thực thi tự động bằng exec() trong môi trường Python 3.10+.
2. Các thư viện đã được import sẵn trong global scope:
   - `pd` (pandas: công cụ phân tích và xử lý bảng tính chính)
   - `duckdb` (dùng để truy vấn SQL tốc độ cao trực tiếp trên file CSV, Parquet, hoặc SQLite)
   - `sqlite3`, `openpyxl` (xử lý cơ sở dữ liệu SQLite và bảng tính Excel nhiều sheet)
   - `datetime`, `date`, `timedelta`, `re`, `np`, `math`, `unicodedata`
   - `calendar_utils` (chứa các hàm: `resolve_as_of_date`, `resolve_vietnamese_relative_date`, `get_quarter_date_range`, `get_month_date_range`, `get_week_date_range`, `get_same_period_last_year`, `get_tet_window`, `get_tet_date`, `get_holiday_range`, `solar_to_lunar`, `lunar_to_solar`)
3. Các biến đã có sẵn trong môi trường:
   - `FILE_MAP`: Từ điển ánh xạ tên file sang đường dẫn tuyệt đối (vd: `FILE_MAP['orders.csv']`)
   - `data_dir` / `DATA_DIR`: Đường dẫn tới thư mục chứa dữ liệu
   - `as_of_date`: Đối tượng `datetime.date` làm mốc thời gian tham chiếu (ngày hiện tại / ngày neo).
   - Tên file được biến đổi thành biến trực tiếp (vd: `orders_csv = FILE_MAP['orders.csv']`).

CRITICAL RULES (ERROR-PREVENTION & SYSTEM INTEGRITY):
* CẢNH BÁO: VI PHẠM BẤT KỲ QUY TẮC NÀO DƯỚI ĐÂY SẼ LÀM HỆ THỐNG CRASH NGAY LẬP TỨC!
1. ĐỊNH DẠNG ĐẦU RA - CHỈ MÃ PYTHON DUY NHẤT:
   - BẮT BUỘC chỉ trả về DUY NHẤT một khối mã Python hợp lệ được bao quanh bởi thẻ ```python ... ```.
   - TUYỆT ĐỐI CẤM (STRICTLY FORBIDDEN): Mọi văn bản hội thoại, lời chào hỏi, giải thích bằng lời, ghi chú hay ký tự bất kỳ bên ngoài khối code.
2. TUÂN THỦ LƯỢC ĐỒ DỮ LIỆU - CẤM BỊA ĐẶT CỘT (SCHEMA FIDELITY):
   - CHỈ ĐƯỢC sử dụng các tên bảng và tên cột CHÍNH XÁC 100% như đã định nghĩa trong Schema String.
   - TUYỆT ĐỐI KHÔNG tự bịa (hallucinate) tên cột, tên bảng hoặc các thuộc tính không có trong lược đồ dữ liệu.
3. XỬ LÝ AN TOÀN CÁC TRƯỜNG HỢP BIÊN (EDGE CASES & SAFE DATA EXTRACTION):
   - Luôn xử lý dữ liệu khuyết/NaN một cách an toàn bằng `.fillna()`, `.dropna()`, hoặc kiểm tra `pd.isna()`.
   - Trước khi trích xuất giá trị phần tử (vd: `.iloc[0]`, `.values[0]`), BẮT BUỘC phải kiểm tra DataFrame/Series có rỗng hay không (`if df.empty:`). Nếu rỗng, gán `final_result = None` hoặc `"unanswerable"` để tránh IndexError/KeyError làm sập chương trình.
4. ĐỊNH NGHĨA KIỂU DỮ LIỆU VÀ ĐỊNH DẠNG KẾT QUẢ (EXPECTED DATA TYPES & FORMAT):
   - Kết quả cuối cùng BẮT BUỘC phải được gán vào biến `final_result` (hoặc `FINAL_RESULT`).
   - Nếu kết quả là số nguyên / số lượng / đếm: gán kiểu `int` (vd: `final_result = int(...)`).
   - Nếu kết quả là số thực / doanh thu / tỷ lệ: làm tròn đúng 2 chữ số thập phân `round(float(...), 2)`.
   - Nếu kết quả là chuỗi / tên: chuẩn hóa Unicode NFC, loại bỏ khoảng trắng thừa bằng `.strip()`.
   - Không được gán nguyên một DataFrame hay Series lớn vào `final_result` mà phải rút trích giá trị cụ thể.

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. SUY LUẬN TỪNG BƯỚC (CHAIN-OF-THOUGHT):
   - Trước khi viết code, bạn BẮT BUỘC phải viết 1-2 dòng comment bắt đầu bằng `# Phân tích: ` để giải thích ngắn gọn cách bạn sẽ lọc dữ liệu và tính toán.
2. GÁN KẾT QUẢ CUỐI CÙNG VÀO BIẾN `final_result` HOẶC `FINAL_RESULT`:
   - Ví dụ: `final_result = 15000000.5` hoặc `final_result = 42` hoặc `final_result = "Hà Nội"`.
3. XỬ LÝ TRƯỜNG HỢP KHÔNG THỂ TRẢ LỜI (UNANSWERABLE & PENALIZED ACCURACY):
   - Cuộc thi áp dụng Penalized Accuracy: Trả lời sai bị TRỪ ĐIỂM, không trả lời (Abstain/Unanswerable) được 0 điểm.
   - BẮT BUỘC kiểm tra xem các bảng và cột cần thiết có tồn tại trong dữ liệu thực tế hay không trước khi tính toán.
   - Nếu lược đồ dữ liệu (schema) hoàn toàn không có bảng hoặc cột thông tin nào liên quan để trả lời câu hỏi, không được đoán mò. Hãy viết comment phân tích và gán `final_result = "unanswerable"`.
   - Nếu dữ liệu kinh doanh thiếu thông tin cốt lõi để tính toán (ví dụ: câu hỏi yêu cầu tính lợi nhuận `profit` nhưng trong dữ liệu chỉ có giá bán `selling_price` mà không có giá vốn/chi phí mua hàng `cost`/`purchase_price`), TUYỆT ĐỐI KHÔNG đoán mò hay tự giả định chi phí = 0. Hãy viết comment `# Phân tích: [Nêu rõ lý do bằng tiếng Việt vì sao thiếu thông tin]...` và gán `final_result = "unanswerable"`.
   - Nếu schema có cột liên quan nhưng bản ghi thực tế không tồn tại sau khi lọc (hoặc rỗng), hãy gán `final_result = None` (hoặc `"unanswerable"`).
4. XỬ LÝ NGÀY THÁNG TIẾNG VIỆT & as_of_date:
   - Các từ chỉ thời gian tương đối như 'hôm nay', 'hôm qua', 'tuần trước', 'tuần rồi', 'tháng này', 'cùng kỳ năm ngoái', 'Tết', 'quý 2 năm ngoái', 'năm nay':
     Ưu tiên sử dụng các mốc ngày đã được tính toán sẵn trong mục PRE-COMPUTED TIME INTERVALS bên dưới hoặc dùng `calendar_utils.resolve_vietnamese_relative_date("cụm từ", as_of=as_of_date)`.
5. TUÂN THỦ QUY TẮC TỪ RULEBOOK ĐỘNG (RULEBOOK.MD):
   - Mọi định nghĩa kinh doanh BẮT BUỘC phải tuân theo file `rulebook.md` của tập dữ liệu hiện tại:
     * Doanh thu (`revenue` / `doanh số`): loại trừ các đơn hủy, hoàn trả, hoặc thất bại (`cancelled`, `returned`, `failed`, `đã hủy`, `hoàn trả`) trừ khi rulebook quy định khác.
     * Chiết khấu và voucher: áp dụng đúng công thức giảm giá dòng sản phẩm và mã voucher nếu có trong dữ liệu và rulebook.
     * Xếp hạng và hòa điểm (Ranking & Tie-breaking): sắp xếp giảm dần theo tiêu chí chính; nếu giá trị bằng nhau thì tie-break theo mã sản phẩm / đối tượng tăng dần (`ascending=True`).
6. TRUY VẤN TỪ KHÓA TIẾNG VIỆT KHÔNG DẤU / KHÔNG PHÂN BIỆT HOA THƯỜNG:
   - Khi tìm kiếm tên riêng, chi nhánh, sản phẩm bằng tiếng Việt: chuẩn hóa về chữ thường và loại bỏ dấu (hoặc dùng regex) để so khớp linh hoạt, tránh bị lệch do cách gõ dấu (ví dụ: "ha noi" khớp "Hà Nội").
7. ĐỊNH DẠNG SỐ VÀ CHUỖI:
   - Nếu là số thực, làm tròn 2 chữ số thập phân (`round(val, 2)`).
   - Nếu là số đếm / số nguyên, giữ nguyên kiểu số nguyên.
   - Chuỗi ký tự tiếng Việt phải chuẩn Unicode NFC.
8. ĐỊNH DẠNG PHẢN HỒI:
   - CHỈ trả về duy nhất một khối mã Python trong thẻ ```python ... ```.
   - Tuyệt đối không viết thêm lời giải thích bằng lời ở ngoài khối code."""


FEW_SHOT_EXAMPLES = """---
### VÍ DỤ 1 (Lọc dữ liệu thời gian tương đối dựa vào as_of_date):
Câu hỏi: "Có bao nhiêu đơn hàng thành công trong tuần trước?"
Ngày tham chiếu: 2024-06-19

```python
# Phân tích: Dùng calendar_utils để tính khoảng thời gian 'tuần trước' từ as_of_date, sau đó đọc file orders.csv, lọc các đơn hàng có trạng thái 'thành công' nằm trong khoảng thời gian này và đếm số lượng dòng.
import pandas as pd
import calendar_utils

start_date, end_date = calendar_utils.resolve_vietnamese_relative_date("tuần trước", as_of=as_of_date)

csv_path = FILE_MAP.get("orders.csv", "orders.csv")
df = pd.read_csv(csv_path)

df['order_date'] = pd.to_datetime(df['created_at']).dt.date
mask = (
    (df['order_date'] >= start_date) &
    (df['order_date'] <= end_date) &
    (df['status'].str.strip().str.lower() == 'thành công')
)
filtered = df[mask]

final_result = int(len(filtered))
```

---
### VÍ DỤ 2 (Tìm kiếm từ khóa tiếng Việt không dấu / không phân biệt hoa thường):
Câu hỏi: "Tính tổng doanh số của chi nhánh 'da nang' trong năm 2024?"
Ngày tham chiếu: 2024-06-01

```python
# Phân tích: Định nghĩa hàm remove_accents để chuẩn hóa cột branch_name và từ khóa 'da nang' về chữ thường không dấu, sau đó lọc các dòng thuộc năm 2024 trong sales.csv và tính tổng doanh thu cột revenue.
import pandas as pd
import unicodedata
import re

def remove_accents(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize('NFD', s)
    s = re.sub(r'[\u0300-\u036f]', '', s)
    s = s.replace('đ', 'd').replace('Đ', 'D')
    return s.strip().lower()

csv_path = FILE_MAP.get("sales.csv", "sales.csv")
df = pd.read_csv(csv_path)

target_branch = remove_accents("da nang")
df['branch_normalized'] = df['branch_name'].apply(remove_accents)

df['order_date'] = pd.to_datetime(df['order_date'])
mask = (
    (df['branch_normalized'] == target_branch) &
    (df['order_date'].dt.year == 2024)
)
filtered = df[mask]

if filtered.empty:
    final_result = None
else:
    total_rev = filtered['revenue'].sum()
    final_result = round(float(total_rev), 2)
```

---
### VÍ DỤ 3 (Unanswerable - Schema không có bảng hoặc cột thông tin liên quan):
Hỏi: "cho biết tên nhân viên bán được nhiều hàng nhất" (as_of_date: 2026-03-15)
Lược đồ dữ liệu (Schema): Chỉ chứa `orders.csv` với các cột `order_id`, `product_id`, `revenue`.
Đáp:

```python
# Phân tích: Câu hỏi yêu cầu tìm "tên nhân viên", nhưng trong lược đồ dữ liệu chỉ có bảng orders với các cột order_id, product_id, revenue. Không có dữ liệu nào về nhân viên (employee). Do đó không thể trả lời.
final_result = "unanswerable"
```

---
### VÍ DỤ 4 (Đọc file Excel nhiều sheets, tính toán tổng hợp GroupBy và sắp xếp):
Câu hỏi: "Khách hàng nào có tổng chi tiêu cao nhất trong quý 1 năm 2024?"
Ngày tham chiếu: 2024-04-01

```python
# Phân tích: Lấy khoảng ngày quý 1 năm 2024, đọc sheet "Orders" từ file ecommerce.xlsx, lọc các đơn 'completed', nhóm theo customer_name tính tổng total_amount và sắp xếp giảm dần để lấy khách hàng đứng đầu.
import pandas as pd
import calendar_utils

start_date, end_date = calendar_utils.get_quarter_date_range(2024, 1)

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
        prompt_parts = []
        prompt_parts.append("### THÔNG TIN CẤU TRÚC BẢNG VÀ DỮ LIỆU (SCHEMA STRING):")
        prompt_parts.append(schema_context.strip())
        prompt_parts.append("\n" + "=" * 50)

        prompt_parts.append("### THÔNG TIN THAM CHIẾU (REFERENCE):")
        if as_of_date:
            prompt_parts.append(f"- Ngày tham chiếu (as_of_date): `{as_of_date}`")
            try:
                from src.calendar_utils import get_precomputed_intervals_summary
            except ImportError:
                try:
                    from calendar_utils import get_precomputed_intervals_summary
                except ImportError:
                    get_precomputed_intervals_summary = None

            if get_precomputed_intervals_summary is not None:
                cal_path = None
                if file_map:
                    cal_path = file_map.get("calendar.csv") or file_map.get("calendar")
                    if not cal_path:
                        for k, v in file_map.items():
                            if "calendar.csv" in k.lower():
                                cal_path = v
                                break
                intervals_summary = get_precomputed_intervals_summary(as_of_date, calendar_path=cal_path)
                if intervals_summary:
                    prompt_parts.append("\n" + "=" * 50)
                    prompt_parts.append(intervals_summary)
        else:
            prompt_parts.append("- Ngày tham chiếu (as_of_date): Chưa chỉ định (mặc định hôm nay)")

        if file_map:
            file_list = ", ".join(f"`{k}`" for k in file_map.keys())
            prompt_parts.append(f"- Các file dữ liệu có sẵn trong FILE_MAP: {file_list}")

        rb = rulebook_content if rulebook_content is not None else self.rulebook_content
        if rb and rb.strip():
            prompt_parts.append("\n" + "=" * 50)
            prompt_parts.append("### QUY TẮC TẬP DỮ LIỆU HIỆN TẠI (RULEBOOK.MD):")
            prompt_parts.append(rb.strip())

        prompt_parts.append("\n" + "=" * 50)
        prompt_parts.append(f"### CÂU HỎI TIẾNG VIỆT:\n\"{question}\"\n")
        prompt_parts.append(
            "CRITICAL RULES (ERROR-PREVENTION - TUÂN THỦ TUYỆT ĐỐI ĐỂ TRÁNH CRASH HỆ THỐNG):\n"
            "- CHỈ sử dụng ĐÚNG tên bảng và tên cột có trong Schema, TUYỆT ĐỐI KHÔNG bịa đặt cột.\n"
            "- KIỂM TRA sự tồn tại của cột/bảng trước khi tính toán. Nếu dữ liệu kinh doanh thiếu thông tin cốt lõi (ví dụ: tính lợi nhuận nhưng không có cột giá vốn/chi phí), gán `final_result = \"unanswerable\"` kèm comment giải thích lý do bằng tiếng Việt.\n"
            "- Tuân thủ rulebook: Doanh thu loại trừ các đơn hủy/trả (cancelled/returned). Xếp hạng nếu bằng nhau thì ưu tiên theo mã đối tượng/sản phẩm tăng dần (tie-break ascending).\n"
            "- Xử lý an toàn edge cases: dùng `.fillna()`, kiểm tra `if df.empty:` trước khi trích xuất giá trị để tránh KeyError/IndexError.\n"
            "- BẮT BUỘC bắt đầu code bằng comment `# Phân tích: ...`, viết đoạn mã Python (dùng pandas, sqlite3, openpyxl) hợp lệ để tính toán và gán câu trả lời vào biến `final_result` hoặc `FINAL_RESULT` (hoặc gán `final_result = \"unanswerable\"` nếu schema không có thông tin liên quan).\n"
            "- TUYỆT ĐỐI KHÔNG viết lời giải thích hay văn bản hội thoại nào ngoài code block.\n"
            "Chỉ in ra code trong thẻ ```python ... ```, không giải thích."
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
