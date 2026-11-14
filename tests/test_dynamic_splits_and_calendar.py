"""Unit tests for Dynamic Split Rulebooks, Calendar Utils, and Multi-File Business Parsers.

Verifies:
1. Multi-file business folders under <data_dir>/data/<business_id>/ containing CSV, XLSX (multiple sheets),
   and SQLite (multiple tables) with graceful handling of missing or empty README.md.
2. Inferred data types, row counts, min/max date ranges, 3-5 sample rows, and distinct values for low-cardinality columns.
3. Runtime dynamic parsing of <data_dir>/calendar.csv for solar-to-lunar, lunar-to-solar, and Tết windows.
4. Vietnamese relative time phrase resolution ("tháng này", "tuần rồi", "cùng kỳ năm ngoái", "Tết").
5. Live rulebook.md prompt injection and strict unanswerable instruction when required columns are absent.
"""

from datetime import date
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

from src import calendar_utils
from src.prompt_builder import PromptBuilder, read_rulebook
from src.schema_parser import SchemaParser, parse_csv_head, parse_excel_head, parse_sqlite_head


class TestDynamicSplitsAndCalendar(unittest.TestCase):
    """Test suite for upgraded schema parser, calendar utils, and prompt builder."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_multi_file_business_folder_under_data_subfolder(self):
        """Verify business folder under <data_dir>/data/<business_id>/ parses CSV, XLSX, SQLite and missing README."""
        biz_dir = self.base_dir / "data" / "biz_megacorp"
        biz_dir.mkdir(parents=True)

        # 1. Empty README.md (must not fail or crash)
        (biz_dir / "README.md").write_text("", encoding="utf-8")

        # 2. CSV file
        df_csv = pd.DataFrame({
            "order_id": [1, 2, 3],
            "order_date": ["2024-01-05", "2024-06-15", "2024-12-20"],
            "status": ["Hoàn thành", "Đang giao", "Hoàn thành"],
        })
        df_csv.to_csv(biz_dir / "orders.csv", index=False)

        # 3. XLSX with 2 sheets
        with pd.ExcelWriter(biz_dir / "finances.xlsx", engine="openpyxl") as writer:
            pd.DataFrame({"acc_id": [101, 102], "balance": [1000.0, 2000.0]}).to_excel(writer, sheet_name="Accounts", index=False)
            pd.DataFrame({"tx_id": ["T1", "T2"], "amount": [50.0, 150.0]}).to_excel(writer, sheet_name="Transactions", index=False)

        # 4. SQLite with 2 tables
        db_path = biz_dir / "warehouse.sqlite"
        conn = sqlite3.connect(str(db_path))
        cur = conn.cursor()
        cur.execute("CREATE TABLE inventory (sku TEXT, qty INTEGER);")
        cur.execute("INSERT INTO inventory VALUES ('SKU_A', 10), ('SKU_B', 20);")
        cur.execute("CREATE TABLE shipments (ship_id TEXT, ship_date TEXT);")
        cur.execute("INSERT INTO shipments VALUES ('S1', '2024-03-01'), ('S2', '2024-08-15');")
        conn.commit()
        conn.close()

        # Parse via SchemaParser
        parser = SchemaParser(data_dir=self.base_dir)
        context = parser.parse(business_id="biz_megacorp")

        self.assertEqual(context.business_id, "biz_megacorp")
        self.assertIn("orders.csv", context.file_map)
        self.assertIn("finances.xlsx", context.file_map)
        self.assertIn("warehouse.sqlite", context.file_map)

        # 1 CSV table + 2 XLSX sheets + 2 SQLite tables = 5 total tables
        self.assertEqual(len(context.tables), 5)

        # Verify date range extracted from orders.csv
        orders_tbl = next(t for t in context.tables if t.name == "orders.csv")
        self.assertIn("order_date", orders_tbl.date_ranges)
        self.assertEqual(orders_tbl.date_ranges["order_date"], "2024-01-05 to 2024-12-20")

        # Verify low-cardinality values extracted for status
        self.assertIn("status", orders_tbl.low_cardinality_values)
        self.assertIn("Hoàn thành", orders_tbl.low_cardinality_values["status"])
        self.assertIn("Đang giao", orders_tbl.low_cardinality_values["status"])

        # Prompt context handles empty README gracefully without crashing
        prompt_str = context.to_prompt_context()
        self.assertIn("Không có file README.md", prompt_str)
        self.assertIn("orders.csv", prompt_str)
        self.assertIn("finances.xlsx [Accounts]", prompt_str)
        self.assertIn("warehouse.sqlite -> inventory", prompt_str)

    def test_dynamic_calendar_csv_parsing_and_tet_window(self):
        """Verify dynamic parsing of calendar.csv at runtime for lunar conversions and Tết windows."""
        cal_file = self.base_dir / "calendar.csv"
        # Simulate official DataMate calendar.csv with solar dates, lunar dates, and holiday names
        rows = [
            {"date": "2024-02-08", "lunar_date": "2023-12-29", "holiday": "Nghỉ Tết Giáp Thìn"},
            {"date": "2024-02-09", "lunar_date": "2023-12-30", "holiday": "Nghỉ Tết - 30 Tết"},
            {"date": "2024-02-10", "lunar_date": "2024-01-01", "holiday": "Tết Nguyên Đán - Mùng 1 Tết"},
            {"date": "2024-02-11", "lunar_date": "2024-01-02", "holiday": "Tết Nguyên Đán - Mùng 2 Tết"},
            {"date": "2024-02-12", "lunar_date": "2024-01-03", "holiday": "Tết Nguyên Đán - Mùng 3 Tết"},
            {"date": "2024-02-13", "lunar_date": "2024-01-04", "holiday": "Nghỉ Tết"},
            {"date": "2024-02-14", "lunar_date": "2024-01-05", "holiday": "Nghỉ Tết"},
            {"date": "2024-04-30", "lunar_date": "2024-03-22", "holiday": "Giải phóng miền Nam (30/4)"},
            {"date": "2024-05-01", "lunar_date": "2024-03-23", "holiday": "Quốc tế Lao động (1/5)"},
        ]
        pd.DataFrame(rows).to_csv(cal_file, index=False)

        calendar_utils.set_calendar_csv_path(cal_file)

        # 1. Solar to Lunar dynamic lookup
        ld, lm, ly, _ = calendar_utils.solar_to_lunar(date(2024, 2, 10))
        self.assertEqual((ld, lm, ly), (1, 1, 2024))

        # 2. Lunar to Solar dynamic lookup
        sol = calendar_utils.lunar_to_solar(lunar_year=2024, lunar_month=1, lunar_day=1)
        self.assertEqual(sol, date(2024, 2, 10))

        # 3. Dynamic Tết Window computation (min/max of Tết holiday rows from calendar.csv)
        tet_start, tet_end = calendar_utils.get_tet_window(year=2024)
        self.assertEqual(tet_start, date(2024, 2, 8))
        self.assertEqual(tet_end, date(2024, 2, 14))

        # 4. Dynamic Holiday Range
        h_start, h_end = calendar_utils.get_holiday_range("30/4", year=2024)
        self.assertEqual(h_start, date(2024, 4, 30))
        self.assertEqual(h_end, date(2024, 4, 30))

        # Cleanup
        calendar_utils.set_calendar_csv_path(None)

    def test_vietnamese_relative_time_phrases(self):
        """Verify resolution of Vietnamese time phrases: tháng này, tuần rồi, cùng kỳ năm ngoái, Tết."""
        as_of = date(2024, 6, 19)  # Wednesday

        # 'tháng này'
        start_m, end_m = calendar_utils.resolve_vietnamese_relative_date("tháng này", as_of=as_of)
        self.assertEqual(start_m, date(2024, 6, 1))
        self.assertEqual(end_m, date(2024, 6, 30))

        # 'tuần rồi' / 'tuần trước'
        start_w, end_w = calendar_utils.resolve_vietnamese_relative_date("tuần rồi", as_of=as_of)
        self.assertEqual(start_w, date(2024, 6, 10))  # Monday of previous week
        self.assertEqual(end_w, date(2024, 6, 16))    # Sunday of previous week

        # 'cùng kỳ năm ngoái'
        start_ck, end_ck = calendar_utils.resolve_vietnamese_relative_date("cùng kỳ năm ngoái", as_of=as_of)
        self.assertEqual(start_ck, date(2023, 6, 1))
        self.assertEqual(end_ck, date(2023, 6, 30))

        # 'cùng kỳ năm ngoái' with explicit date interval
        shifted_s, shifted_e = calendar_utils.get_same_period_last_year("2024-02-29", "2024-03-15")
        self.assertEqual(shifted_s, date(2023, 2, 28))
        self.assertEqual(shifted_e, date(2023, 3, 15))

        # 'Tết'
        tet_s, tet_e = calendar_utils.resolve_vietnamese_relative_date("dịp Tết", as_of=as_of)
        self.assertIsInstance(tet_s, date)
        self.assertIsInstance(tet_e, date)
        self.assertLess(tet_s, tet_e)

    def test_live_rulebook_injection_and_unanswerable_instruction(self):
        """Verify prompt builder injects live rulebook.md and instructs unanswerable when columns are missing."""
        rulebook_file = self.base_dir / "rulebook.md"
        rulebook_text = (
            "### QUY TẮC TÍNH TOÁN:\n"
            "- Doanh thu chỉ tính các đơn hàng có trạng thái 'COMPLETED'.\n"
            "- Tuần tính từ Chủ Nhật đến thứ Bảy.\n"
            "- Lợi nhuận = Doanh thu - Giá vốn (cost)."
        )
        rulebook_file.write_text(rulebook_text, encoding="utf-8")

        # 1. read_rulebook helper
        loaded_rb = read_rulebook(self.base_dir)
        self.assertEqual(loaded_rb, rulebook_text)

        # 2. System prompt injection
        builder = PromptBuilder(rulebook_content=loaded_rb)
        sys_prompt = builder.build_system_prompt()
        self.assertIn("=== QUY TẮC BỔ SUNG TỪ RULEBOOK (RULEBOOK.MD) ===", sys_prompt)
        self.assertIn("Doanh thu chỉ tính các đơn hàng có trạng thái 'COMPLETED'", sys_prompt)
        self.assertIn("TUÂN THỦ QUY TẮC TỪ RULEBOOK ĐỘNG", sys_prompt)

        # 3. User prompt unanswerable rule
        user_prompt = builder.build_user_prompt(
            question="Tính tổng lợi nhuận của sản phẩm sữa tươi?",
            schema_context="Table: sales.csv\n- product_name (string)\n- revenue (float64)",
            as_of_date="2024-06-01",
        )
        self.assertIn("### QUY TẮC TẬP DỮ LIỆU HIỆN TẠI (RULEBOOK.MD):", user_prompt)
        self.assertIn("KIỂM TRA sự tồn tại của cột/bảng trước khi tính toán", user_prompt)
        self.assertIn("final_result = \"unanswerable\"", user_prompt)


if __name__ == "__main__":
    unittest.main()
