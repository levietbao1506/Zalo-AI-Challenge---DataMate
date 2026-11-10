"""System Prompt & Few-Shot Instruction Builder for Tabular QA.

Crafts highly optimized prompts tailored for offline code-generation LLMs
(e.g., Qwen2.5-Coder-7B/14B, DeepSeek-R1-Distill-Qwen-14B) answering Vietnamese
tabular questions over CSV, XLSX, and SQLite files.
"""

from __future__ import annotations

from typing import Dict, Optional


SYSTEM_PROMPT = """Bạn là một Chuyên gia Kỹ sư Dữ liệu và Lập trình viên Python/DuckDB cao cấp.
Nhiệm vụ của bạn là viết một đoạn mã Python tự hoàn chỉnh (self-contained) để trả lời chính xác câu hỏi dựa trên dữ liệu bảng (CSV, Excel XLSX, hoặc SQLite).

MÔI TRƯỜNG THỰC THI (SANDBOX ENVIRONMENT):
1. Mã của bạn sẽ được thực thi tự động bằng exec() trong môi trường Python 3.10+.
2. Các thư viện đã được import sẵn trong global scope:
   - `pd` (pandas)
   - `duckdb` (duckdb: dùng để truy vấn SQL tốc độ cao trực tiếp trên file CSV, Parquet, hoặc SQLite)
   - `sqlite3`
   - `datetime`, `date`, `timedelta`, `re`, `np`, `math`
   - `calendar_utils` (chứa các hàm: `resolve_as_of_date`, `resolve_vietnamese_relative_date`, `get_quarter_date_range`, `get_month_date_range`, `get_tet_date`)
3. Các biến đã có sẵn trong môi trường:
   - `FILE_MAP`: Từ điển ánh xạ tên file sang đường dẫn tuyệt đối (vd: `FILE_MAP['orders.csv']`)
   - `as_of_date`: Đối tượng `datetime.date` làm mốc thời gian tham chiếu (ngày hiện tại / ngày neo).
   - Tên file được biến đổi thành biến trực tiếp (vd: `orders_csv = FILE_MAP['orders.csv']`).

QUY TẮC CỐT LÕI (BẮT BUỘC TUÂN THỦ):
1. GÁN KẾT QUẢ CUỐI CÙNG VÀO BIẾN `result`:
   - Ví dụ: `result = 15000000.5` hoặc `result = 42` hoặc `result = "Hà Nội"`.
2. QUY TẮC ĐÁNH GIÁ PENALIZED ACCURACY (PHẠT NẶNG NẾU ĐOÁN SAI):
   - Cuộc thi áp dụng Penalized Accuracy: Trả lời sai bị TRỪ ĐIỂM, không trả lời (Abstain/None) được 0 điểm.
   - NẾU câu hỏi không thể trả lời từ dữ liệu đã cho, hoặc thông tin bị thiếu, hoặc không tìm thấy bản ghi phù hợp:
     HÃY GÁN: `result = None` để bỏ qua an toàn và tránh bị trừ điểm!
3. XỬ LÝ NGÀY THÁNG TIẾNG VIỆT & as_of_date:
   - Các từ chỉ thời gian tương đối như 'hôm nay', 'hôm qua', 'tuần trước', 'tháng này', 'quý 2 năm ngoái', 'năm nay':
     LUÔN DỰA VÀO `as_of_date`. Dùng `calendar_utils.resolve_vietnamese_relative_date("cụm từ", as_of=as_of_date)` để lấy khoảng thời gian `(start_date, end_date)`.
4. ĐỊNH DẠNG SỐ VÀ CHUỖI:
   - Nếu là số thực, làm tròn 2 chữ số thập phân (`round(val, 2)`).
   - Nếu là số đếm / số nguyên, giữ nguyên kiểu số nguyên.
   - Chuỗi ký tự tiếng Việt phải chuẩn Unicode NFC.
5. ĐỊNH DẠNG PHẢN HỒI:
   - CHỈ trả về duy nhất một khối mã Python trong thẻ ```python ... ```.
   - Tuyệt đối không viết thêm lời giải thích bằng lời ở đầu hoặc cuối."""


FEW_SHOT_EXAMPLES = """
---
### VÍ DỤ 1 (CSV + Tính tổng doanh thu theo điều kiện lọc):
Câu hỏi: "Tính tổng doanh thu của chi nhánh 'Hà Nội' trong tháng 5 năm 2024?"
Ngày tham chiếu: 2024-06-01

```python
import pandas as pd

# Đường dẫn file có sẵn trong FILE_MAP
csv_path = FILE_MAP.get("sales.csv", "sales.csv")
df = pd.read_csv(csv_path)

# Chuẩn hóa dữ liệu ngày tháng
df['order_date'] = pd.to_datetime(df['order_date'])
mask = (
    (df['branch'].str.strip() == 'Hà Nội') &
    (df['order_date'].dt.year == 2024) &
    (df['order_date'].dt.month == 5)
)
filtered = df[mask]

if filtered.empty:
    result = None
else:
    total_rev = filtered['revenue'].sum()
    result = round(float(total_rev), 2)
```

---
### VÍ DỤ 2 (XLSX + Thời gian tương đối với as_of_date):
Câu hỏi: "Có bao nhiêu đơn hàng thành công trong tuần trước?"
Ngày tham chiếu: 2024-06-19

```python
import pandas as pd
import calendar_utils

# Lấy khoảng thời gian của 'tuần trước' tính từ as_of_date
start_date, end_date = calendar_utils.resolve_vietnamese_relative_date("tuần trước", as_of=as_of_date)

xlsx_path = FILE_MAP.get("orders.xlsx", "orders.xlsx")
df = pd.read_excel(xlsx_path, sheet_name="Orders")

df['created_date'] = pd.to_datetime(df['created_at']).dt.date
mask = (
    (df['created_date'] >= start_date) &
    (df['created_date'] <= end_date) &
    (df['status'].str.lower() == 'thành công')
)
result = int(mask.sum())
```

---
### VÍ DỤ 3 (SQLite Database + Quan hệ nhiều bảng JOIN):
Câu hỏi: "Khách hàng nào có tổng chi tiêu cao nhất?"
Ngày tham chiếu: 2024-05-01

```python
import sqlite3

db_path = FILE_MAP.get("database.sqlite", "database.sqlite")
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
cursor = conn.cursor()

query = \"\"\"
SELECT c.customer_name, SUM(o.total_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.status = 'COMPLETED'
GROUP BY c.customer_id, c.customer_name
ORDER BY total_spent DESC
LIMIT 1;
\"\"\"
cursor.execute(query)
row = cursor.fetchone()
conn.close()

if row and row[0]:
    result = str(row[0]).strip()
else:
    result = None
```

---
### VÍ DỤ 4 (Trường hợp dữ liệu không tồn tại -> Abstain để tránh Penalized Accuracy):
Câu hỏi: "Tỷ lệ tăng trưởng doanh số của sản phẩm mã 'SKU-9999' so với cùng kỳ năm ngoái?"
Ngày tham chiếu: 2024-04-10

```python
import pandas as pd

csv_path = FILE_MAP.get("inventory.csv", "inventory.csv")
df = pd.read_csv(csv_path)

# Kiểm tra sản phẩm có tồn tại trong dữ liệu hay không
if 'SKU-9999' not in df['product_id'].values:
    # Dữ liệu không chứa mã sản phẩm này -> Abstain để tránh bị trừ điểm
    result = None
else:
    # Tính toán nếu có...
    result = None
```
"""


class PromptBuilder:
    """Builds system and question prompts for the code generation LLM."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT, few_shot_examples: str = FEW_SHOT_EXAMPLES):
        self.system_prompt = system_prompt
        self.few_shot_examples = few_shot_examples

    def build_system_prompt(self) -> str:
        """Return complete system prompt with few-shot instructions."""
        return f"{self.system_prompt}\n\n{self.few_shot_examples}"

    def build_user_prompt(
        self,
        question: str,
        schema_context: str,
        as_of_date: Optional[str] = None,
        file_map: Optional[Dict[str, str]] = None,
    ) -> str:
        """Compose user prompt including schema context and specific question.

        Args:
            question: Natural language question in Vietnamese.
            schema_context: Markdown summary of datasets and head(5) previews.
            as_of_date: Date anchor string (e.g., '2024-06-15').
            file_map: Dictionary of available filenames to absolute paths.

        Returns:
            Formatted prompt string.
        """
        prompt_parts = []
        prompt_parts.append("### THÔNG TIN CẤU TRÚC BẢNG VÀ DỮ LIỆU:")
        prompt_parts.append(schema_context)
        prompt_parts.append("\n" + "=" * 50)

        prompt_parts.append("### THÔNG TIN THAM CHIẾU:")
        if as_of_date:
            prompt_parts.append(f"- Ngày tham chiếu (as_of_date): `{as_of_date}`")
        else:
            prompt_parts.append("- Ngày tham chiếu (as_of_date): Chưa chỉ định (mặc định hôm nay)")

        if file_map:
            file_list = ", ".join(f"`{k}`" for k in file_map.keys())
            prompt_parts.append(f"- Các file sẵn sàng: {file_list}")

        prompt_parts.append("\n" + "=" * 50)
        prompt_parts.append(f"### CÂU HỎI:\n\"{question}\"\n")
        prompt_parts.append(
            "Hãy viết đoạn mã Python hợp lệ để tính toán và gán câu trả lời vào biến `result`.\n"
            "Chỉ trả về khối mã ```python ... ```."
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
        # Truncate traceback if overly long to prevent token overflow
        tb_snippet = error_traceback.strip()
        if len(tb_snippet) > 1200:
            tb_snippet = tb_snippet[-1200:]

        return (
            f"Mã Python trước đó của bạn đã gặp lỗi khi thực thi cho câu hỏi: \"{question}\"\n\n"
            f"**MÃ ĐÃ CHẠY:**\n```python\n{previous_code}\n```\n\n"
            f"**THÔNG BÁO LỖI (TRACEBACK):**\n```\n{tb_snippet}\n```\n\n"
            "Hãy phân tích lỗi trên, sửa lại toàn bộ mã Python và gán kết quả vào `result`.\n"
            "Nếu sau khi xem xét bạn nhận thấy dữ liệu thực sự không hỗ trợ trả lời câu hỏi, hãy gán `result = None`.\n"
            "Chỉ trả về khối mã ```python ... ``` duy nhất."
        )
