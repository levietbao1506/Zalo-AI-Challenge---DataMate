"""System Prompt & Few-Shot Instruction Builder for Tabular QA (Bước 3).

Crafts highly optimized prompts tailored for offline code-generation LLMs
(Qwen2.5-Coder-0.5B-Instruct / Qwen2.5-14B-Instruct) answering Vietnamese
tabular questions over CSV, XLSX, and SQLite files with Chain-of-Thought
analysis and unanswerable schema fallback.
"""

from __future__ import annotations

from typing import Dict, Optional


SYSTEM_PROMPT = """Bạn là một Data Analyst chuyên nghiệp. Hãy viết mã Python (dùng thư viện pandas) để trả lời câu hỏi sau. Chỉ in ra code, không giải thích.

MÔI TRƯỜNG THỰC THI (SANDBOX ENVIRONMENT):
1. Mã của bạn sẽ được thực thi tự động bằng exec() trong môi trường Python 3.10+.
2. Các thư viện đã được import sẵn trong global scope:
   - `pd` (pandas: công cụ phân tích và xử lý bảng tính chính)
   - `duckdb` (dùng để truy vấn SQL tốc độ cao trực tiếp trên file CSV, Parquet, hoặc SQLite)
   - `sqlite3`
   - `datetime`, `date`, `timedelta`, `re`, `np`, `math`, `unicodedata`
   - `calendar_utils` (chứa các hàm: `resolve_as_of_date`, `resolve_vietnamese_relative_date`, `get_quarter_date_range`, `get_month_date_range`, `get_tet_date`)
3. Các biến đã có sẵn trong môi trường:
   - `FILE_MAP`: Từ điển ánh xạ tên file sang đường dẫn tuyệt đối (vd: `FILE_MAP['orders.csv']`)
   - `data_dir` / `DATA_DIR`: Đường dẫn tới thư mục chứa dữ liệu
   - `as_of_date`: Đối tượng `datetime.date` làm mốc thời gian tham chiếu (ngày hiện tại / ngày neo).
   - Tên file được biến đổi thành biến trực tiếp (vd: `orders_csv = FILE_MAP['orders.csv']`).

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. SUY LUẬN TỪNG BƯỚC (CHAIN-OF-THOUGHT):
   - Trước khi viết code, bạn BẮT BUỘC phải viết 1-2 dòng comment bắt đầu bằng `# Phân tích: ` để giải thích ngắn gọn cách bạn sẽ lọc dữ liệu và tính toán.
2. GÁN KẾT QUẢ CUỐI CÙNG VÀO BIẾN `final_result`:
   - Ví dụ: `final_result = 15000000.5` hoặc `final_result = 42` hoặc `final_result = "Hà Nội"`.
3. XỬ LÝ TRƯỜNG HỢP KHÔNG THỂ TRẢ LỜI (UNANSWERABLE & PENALIZED ACCURACY):
   - Cuộc thi áp dụng Penalized Accuracy: Trả lời sai bị TRỪ ĐIỂM, không trả lời (Abstain/Unanswerable) được 0 điểm.
   - Nếu lược đồ dữ liệu (schema) hoàn toàn không có bảng hoặc cột thông tin nào liên quan để trả lời câu hỏi, không được đoán mò. Hãy viết comment phân tích và gán `final_result = "unanswerable"`.
   - Nếu schema có cột liên quan nhưng bản ghi thực tế không tồn tại sau khi lọc (hoặc rỗng), hãy gán `final_result = None` (hoặc `"unanswerable"`).
4. XỬ LÝ NGÀY THÁNG TIẾNG VIỆT & as_of_date:
   - Các từ chỉ thời gian tương đối như 'hôm nay', 'hôm qua', 'tuần trước', 'tháng này', 'quý 2 năm ngoái', 'năm nay':
     LUÔN DỰA VÀO `as_of_date`. Dùng `calendar_utils.resolve_vietnamese_relative_date("cụm từ", as_of=as_of_date)` để lấy khoảng thời gian `(start_date, end_date)`.
5. TRUY VẤN TỪ KHÓA TIẾNG VIỆT KHÔNG DẤU / KHÔNG PHÂN BIỆT HOA THƯỜNG:
   - Khi tìm kiếm tên riêng, chi nhánh, sản phẩm bằng tiếng Việt: chuẩn hóa về chữ thường và loại bỏ dấu (hoặc dùng regex) để so khớp linh hoạt, tránh bị lệch do cách gõ dấu (ví dụ: "ha noi" khớp "Hà Nội").
6. ĐỊNH DẠNG SỐ VÀ CHUỖI:
   - Nếu là số thực, làm tròn 2 chữ số thập phân (`round(val, 2)`).
   - Nếu là số đếm / số nguyên, giữ nguyên kiểu số nguyên.
   - Chuỗi ký tự tiếng Việt phải chuẩn Unicode NFC.
7. ĐỊNH DẠNG PHẢN HỒI:
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

xlsx_path = FILE_MAP.get("ecommerce.xlsx", "ecommerce.xlsx")
df_orders = pd.read_excel(xlsx_path, sheet_name="Orders")

df_orders['order_date'] = pd.to_datetime(df_orders['order_date']).dt.date
mask = (
    (df_orders['order_date'] >= start_date) &
    (df_orders['order_date'] <= end_date) &
    (df_orders['status'].str.lower() == 'completed')
)
valid_orders = df_orders[mask]

if valid_orders.empty:
    final_result = None
else:
    top_customer = (
        valid_orders.groupby('customer_name')['total_amount']
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )
    final_result = str(top_customer).strip()
```

---
### VÍ DỤ 5 (Trường hợp dữ liệu không tồn tại -> Abstain để tránh bị trừ điểm):
Câu hỏi: "Tỷ lệ tăng trưởng doanh số của sản phẩm mã 'SKU-9999' so với cùng kỳ năm ngoái?"
Ngày tham chiếu: 2024-04-10

```python
# Phân tích: Đọc inventory.csv và kiểm tra xem mã 'SKU-9999' có nằm trong cột product_id hay không. Vì không có bản ghi nào khớp, gán final_result = None để abstain an toàn.
import pandas as pd

csv_path = FILE_MAP.get("inventory.csv", "inventory.csv")
df = pd.read_csv(csv_path)

if 'SKU-9999' not in df['product_id'].astype(str).values:
    final_result = None
else:
    final_result = None
```"""


class PromptBuilder:
    """Builds system and question prompts for the code generation LLM."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, few_shot_examples: str = FEW_SHOT_EXAMPLES):
        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples

    def build_system_prompt(self) -> str:
        """Return complete system prompt with role declaration and few-shot instructions."""
        return f"{self.system_prompt}\n\n{self.few_shot_examples}"

    def build_user_prompt(
        self,
        question: str,
        schema_context: str,
        as_of_date: Optional[str] = None,
        file_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """Compose user prompt including schema context and specific question.

        Nối: Schema String + as_of_date (ngày hiện tại) + Câu hỏi tiếng Việt.

        Args:
            question: Natural language question in Vietnamese.
            schema_context: Markdown summary of datasets and head(5) previews.
            as_of_date: Date anchor string (e.g., '2024-06-15').
            file_map: Dictionary of available filenames to absolute paths.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = []
        prompt_parts.append("### THÔNG TIN CẤU TRÚC BẢNG VÀ DỮ LIỆU (SCHEMA STRING):")
        prompt_parts.append(schema_context.strip())
        prompt_parts.append("\n" + "=" * 50)

        prompt_parts.append("### THÔNG TIN THAM CHIẾU (REFERENCE):")
        if as_of_date:
            prompt_parts.append(f"- Ngày tham chiếu (as_of_date): `{as_of_date}`")
        else:
            prompt_parts.append("- Ngày tham chiếu (as_of_date): Chưa chỉ định (mặc định hôm nay)")

        if file_map:
            file_list = ", ".join(f"`{k}`" for k in file_map.keys())
            prompt_parts.append(f"- Các file dữ liệu có sẵn trong FILE_MAP: {file_list}")

        prompt_parts.append("\n" + "=" * 50)
        prompt_parts.append(f"### CÂU HỎI TIẾNG VIỆT:\n\"{question}\"\n")
        prompt_parts.append(
            "Hãy bắt đầu bằng comment `# Phân tích: ...`, viết đoạn mã Python (dùng pandas) hợp lệ để tính toán và gán câu trả lời vào biến `final_result` (hoặc gán `final_result = \"unanswerable\"` nếu schema không có thông tin liên quan).\n"
            "Chỉ in ra code trong thẻ ```python ... ```, không giải thích."
        )

        return "\n".join(prompt_parts)

    def build_retry_prompt(
        self,
        question: str,
        previous_code: str,
        error_traceback: str,
    ) -> str:
        """Build self-correction prompt providing error traceback back to LLM.

        Args:
            question: Original question.
            previous_code: Code that triggered runtime or syntax failure.
            error_traceback: Traceback message captured by executor.

        Returns:
            Refinement prompt string.
        """
        # Extract a short error message (e.g. KeyError: 'revenue')
        error_msg = error_traceback.strip().splitlines()[-1] if error_traceback.strip() else "Unknown error"
        if ":" in error_traceback.strip().splitlines()[0]:
            first_line = error_traceback.strip().splitlines()[0]
            if any(first_line.startswith(err) for err in ("KeyError", "IndexError", "TypeError", "ValueError", "SyntaxError", "ZeroDivisionError", "NameError")):
                error_msg = first_line

        # Truncate traceback if overly long to prevent token overflow
        tb_snippet = error_traceback.strip()
        if len(tb_snippet) > 1200:
            tb_snippet = tb_snippet[-1200:]

        return (
            f"Đoạn code trên bị lỗi: {error_msg}. Hãy viết lại code sửa lỗi.\n\n"
            f"**CÂU HỎI:** \"{question}\"\n\n"
            f"**MÃ ĐÃ CHẠY:**\n```python\n{previous_code}\n```\n\n"
            f"**CHI TIẾT LỖI:**\n```\n{tb_snippet}\n```\n\n"
            "Hãy bắt đầu bằng comment `# Phân tích: ...`, sửa lại toàn bộ mã Python (dùng pandas) và gán kết quả vào `final_result`.\n"
            "Nếu sau khi xem xét bạn nhận thấy dữ liệu thực sự không hỗ trợ trả lời câu hỏi, hãy gán `final_result = \"unanswerable\"` hoặc `final_result = None`.\n"
            "Chỉ in ra code trong thẻ ```python ... ```, không giải thích."
        )
