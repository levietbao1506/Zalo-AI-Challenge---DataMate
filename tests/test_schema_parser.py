"""Unit tests for SchemaParser module (DataMate 2026 - Bước 2)."""

import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

import pandas as pd

try:
    import openpyxl  # noqa: F401
    _HAS_OPENPYXL = True
except ImportError:
    _HAS_OPENPYXL = False

from src.schema_parser import (
    DatasetContext,
    SchemaParser,
    TableSchema,
    build_schema_string,
    clean_sample_records,
    extract_as_of_date,
    extract_columns_and_types,
    format_table_schema,
    parse_csv_head,
    parse_excel_head,
    parse_sqlite_head,
    parse_table_head,
    read_business_readme,
    sample_unique_values,
    strip_dataframe_whitespace,
)


class TestSchemaParser(unittest.TestCase):
    """Test suite for schema parsing, README extraction, and prompt formatting."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_extract_columns_and_types(self):
        """Ensure Pandas DataFrame dtypes are cleanly mapped to readable type strings."""
        df = pd.DataFrame({
            "order_id": [1, 2, 3],
            "customer_name": ["Nguyễn Văn A", "Trần Thị B", "Lê C"],
            "revenue": [150000.5, 230000.0, 99000.25],
            "is_active": [True, False, True],
            "order_date": pd.to_datetime(["2024-05-01", "2024-05-02", "2024-05-03"]),
        })

        col_types = extract_columns_and_types(df)
        self.assertEqual(col_types["order_id"], "int64")
        self.assertEqual(col_types["customer_name"], "string")
        self.assertEqual(col_types["revenue"], "float64")
        self.assertEqual(col_types["is_active"], "bool")
        self.assertEqual(col_types["order_date"], "datetime")

    def test_clean_sample_records_and_head_limit(self):
        """Ensure clean_sample_records takes only top 5 rows and handles NaN/NaT cleanly."""
        data = {
            "id": list(range(10)),
            "val": [10.5, float("nan"), 30.0, None, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        }
        df = pd.DataFrame(data)
        records = clean_sample_records(df, n_rows=5)

        self.assertEqual(len(records), 5)
        self.assertEqual(records[0]["id"], 0)
        self.assertIsNone(records[1]["val"])  # NaN converted to None
        self.assertIsNone(records[3]["val"])  # None preserved

    def test_read_business_readme(self):
        """Ensure read_business_readme loads README.md from business subfolder or root."""
        # Create business_01 with README.md
        biz_dir = self.base_dir / "biz_retail"
        biz_dir.mkdir(parents=True)
        readme_file = biz_dir / "README.md"
        readme_file.write_text(
            "# Doanh nghiệp Bán lẻ ABC\n\nas_of_date: 2024-06-15\n\nMô tả: Chuyên bán lẻ tạp hóa.",
            encoding="utf-8",
        )

        content = read_business_readme("biz_retail", data_dir=self.base_dir)
        self.assertIn("Doanh nghiệp Bán lẻ ABC", content)
        self.assertIn("as_of_date: 2024-06-15", content)

        # Test as_of_date extraction
        as_of = extract_as_of_date(content)
        self.assertEqual(as_of, "2024-06-15")

    def test_parse_csv_head_5_rows(self):
        """Ensure parse_csv_head reads exactly 5 rows using Pandas and does not load full table."""
        # Generate 100 rows CSV
        csv_file = self.base_dir / "transactions.csv"
        df_large = pd.DataFrame({
            "tx_id": [f"TX_{i}" for i in range(100)],
            "amount": [i * 1000.0 for i in range(100)],
            "branch": ["Hà Nội" if i % 2 == 0 else "TP HCM" for i in range(100)],
        })
        df_large.to_csv(csv_file, index=False, encoding="utf-8-sig")

        schema = parse_csv_head(csv_file, n_rows=5)
        self.assertEqual(schema.name, "transactions.csv")
        self.assertEqual(schema.source_type, "csv")
        self.assertEqual(schema.columns, ["tx_id", "amount", "branch"])
        self.assertEqual(len(schema.sample_rows), 5)
        self.assertEqual(schema.sample_rows[0]["tx_id"], "TX_0")
        self.assertEqual(schema.sample_rows[4]["tx_id"], "TX_4")
        self.assertEqual(schema.total_rows, 100)

    def test_parse_excel_head_multi_sheet(self):
        """Ensure parse_excel_head reads 5 rows from each sheet in an Excel workbook."""
        if not _HAS_OPENPYXL:
            self.skipTest("openpyxl is not installed in this environment.")
        xlsx_file = self.base_dir / "company_data.xlsx"
        df_orders = pd.DataFrame({
            "order_id": [101, 102, 103, 104, 105, 106],
            "price": [50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        })
        df_customers = pd.DataFrame({
            "cust_id": ["C1", "C2", "C3", "C4", "C5", "C6"],
            "city": ["Đà Nẵng", "Cần Thơ", "Hà Nội", "Huế", "Nha Trang", "Hải Phòng"],
        })

        with pd.ExcelWriter(xlsx_file, engine="openpyxl") as writer:
            df_orders.to_excel(writer, sheet_name="Orders", index=False)
            df_customers.to_excel(writer, sheet_name="Customers", index=False)

        schemas = parse_excel_head(xlsx_file, n_rows=5)
        self.assertEqual(len(schemas), 2)

        orders_schema = next(s for s in schemas if "Orders" in s.name)
        self.assertEqual(len(orders_schema.sample_rows), 5)
        self.assertEqual(orders_schema.columns, ["order_id", "price"])

        cust_schema = next(s for s in schemas if "Customers" in s.name)
        self.assertEqual(len(cust_schema.sample_rows), 5)
        self.assertEqual(cust_schema.columns, ["cust_id", "city"])

    def test_parse_sqlite_head(self):
        """Ensure parse_sqlite_head extracts tables, column definitions, and 5 sample rows."""
        db_file = self.base_dir / "inventory.sqlite"
        conn = sqlite3.connect(str(db_file))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE products (item_id INTEGER, name TEXT, stock INTEGER);")
        for i in range(10):
            cursor.execute("INSERT INTO products VALUES (?, ?, ?);", (i, f"Sản phẩm {i}", i * 5))
        conn.commit()
        conn.close()

        schemas = parse_sqlite_head(db_file, n_rows=5)
        self.assertEqual(len(schemas), 1)
        self.assertEqual(schemas[0].name, "inventory.sqlite -> products")
        self.assertEqual(len(schemas[0].sample_rows), 5)
        self.assertEqual(schemas[0].columns, ["item_id", "name", "stock"])

    def test_schema_parser_end_to_end(self):
        """Ensure SchemaParser integrates business README, CSV, and XLSX into prompt string."""
        biz_dir = self.base_dir / "retail_01"
        biz_dir.mkdir(parents=True)

        # 1. README
        (biz_dir / "README.md").write_text(
            "# Hệ Thống Siêu Thị Retail 01\nNgày tham chiếu: 2024-07-01\nBảng dữ liệu lưu thông tin bán hàng.",
            encoding="utf-8",
        )

        # 2. CSV
        pd.DataFrame({
            "sku": ["SKU1", "SKU2", "SKU3"],
            "qty": [10, 20, 30],
        }).to_csv(biz_dir / "inventory.csv", index=False)

        parser = SchemaParser(data_dir=self.base_dir)
        context = parser.parse(business_id="retail_01")

        self.assertEqual(context.business_id, "retail_01")
        self.assertEqual(context.as_of_date, "2024-07-01")
        self.assertEqual(len(context.tables), 1)
        self.assertEqual(context.tables[0].name, "inventory.csv")

        # 3. Schema Prompt String
        prompt_str = context.to_prompt_context()
        self.assertIn("Hệ Thống Siêu Thị Retail 01", prompt_str)
        self.assertIn("retail_01", prompt_str)
        self.assertIn("2024-07-01", prompt_str)
        self.assertIn("`sku` (string)", prompt_str)
        self.assertIn("`qty` (int64)", prompt_str)
        self.assertIn("inventory.csv", prompt_str)

        # 4. Helper function test
        direct_str = build_schema_string("retail_01", data_dir=self.base_dir)
        self.assertEqual(prompt_str, direct_str)

    def test_strip_dataframe_whitespace(self):
        """Ensure strip_dataframe_whitespace trims leading/trailing whitespace on string/object columns."""
        df = pd.DataFrame({
            "order_id": [1, 2, 3],
            "product_name": ["  Trà sữa trân châu  ", " Cà phê sữa", None],
            "status": ["Hoàn thành  ", "  Đang giao ", "Đã hủy"],
            "revenue": [35000.0, 25000.0, 30000.0],
        })
        cleaned = strip_dataframe_whitespace(df)
        self.assertEqual(cleaned["product_name"].iloc[0], "Trà sữa trân châu")
        self.assertEqual(cleaned["product_name"].iloc[1], "Cà phê sữa")
        self.assertTrue(pd.isna(cleaned["product_name"].iloc[2]))
        self.assertEqual(cleaned["status"].tolist(), ["Hoàn thành", "Đang giao", "Đã hủy"])

    def test_sample_unique_values_and_enriched_formatting(self):
        """Ensure unique values are ranked by frequency (dropping NaNs) and formatted cleanly."""
        csv_file = self.base_dir / "orders.csv"
        rows = [
            {"order_id": 1, "product_name": "  Trà sữa trân châu  ", "status": " Hoàn thành", "revenue": 45000.0},
            {"order_id": 2, "product_name": "Trà sữa trân châu", "status": "Hoàn thành ", "revenue": 45000.0},
            {"order_id": 3, "product_name": "Cà phê sữa ", "status": "Hoàn thành", "revenue": 29000.0},
            {"order_id": 4, "product_name": "Cà phê sữa", "status": "Đang giao", "revenue": 29000.0},
            {"order_id": 5, "product_name": " Bạc xỉu ", "status": "Đang giao", "revenue": 32000.0},
            {"order_id": 6, "product_name": None, "status": "Đã hủy ", "revenue": 0.0},
            {"order_id": 7, "product_name": "Trà đào", "status": None, "revenue": 35000.0},
        ]
        pd.DataFrame(rows).to_csv(csv_file, index=False, encoding="utf-8-sig")

        schema = parse_csv_head(csv_file, n_rows=5, top_n_examples=3)
        # Whitespace was stripped in sample_rows as well
        self.assertEqual(schema.sample_rows[0]["product_name"], "Trà sữa trân châu")
        self.assertEqual(schema.sample_rows[0]["status"], "Hoàn thành")

        # Top 3 unique values ordered by frequency, NaNs dropped
        self.assertEqual(schema.column_examples["product_name"], ["Trà sữa trân châu", "Cà phê sữa", "Bạc xỉu"])
        self.assertEqual(schema.column_examples["status"], ["Hoàn thành", "Đang giao", "Đã hủy"])
        self.assertNotIn("order_id", schema.column_examples)
        self.assertNotIn("revenue", schema.column_examples)

        enriched_text = format_table_schema(schema)
        self.assertIn("Table: orders.csv", enriched_text)
        self.assertIn("- order_id (int64)", enriched_text)
        self.assertIn("[Examples: 'Trà sữa trân châu', 'Cà phê sữa', 'Bạc xỉu']", enriched_text)
        self.assertIn("[Examples: 'Hoàn thành', 'Đang giao', 'Đã hủy']", enriched_text)
        self.assertIn("- revenue (float64)", enriched_text)


if __name__ == "__main__":
    unittest.main()
