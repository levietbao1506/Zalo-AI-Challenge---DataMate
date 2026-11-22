"""Unit tests for automatic business data ingestion into DuckDB."""

from pathlib import Path
from datetime import date
import sqlite3
import tempfile
import unittest

from openpyxl import Workbook

from src.duckdb_store import BusinessDuckDBStore
from src.schema_parser import build_sql_schema_card


class TestBusinessDuckDBStore(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.business_dir = self.root / "doanh_nghiep_01"
        self.business_dir.mkdir()
        self.cache_root = self.root / "cache"
        self.calendar_path = self.root / "calendar.csv"
        self.calendar_path.write_text(
            "date,event\n2024-01-01,Tết Dương lịch\n2024-02-10,Tết Nguyên đán\n",
            encoding="utf-8-sig",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_csv(self) -> None:
        rows = [
            "Mã đơn;Ngày;Số tiền;Trạng thái",
            "DH001;01/01/2024;1.250.000 đ; Hoàn thành ",
            "DH002;02/01/2024;500k;Đang xử lý",
            "DH003;03/01/2024;12,5%;Hoàn thành",
            "DH004;04/01/2024;200000;Đã hủy",
            "DH005;05/01/2024;không rõ;Hoàn thành",
            "Tổng cộng;;1.950.000 đ;",
        ]
        (self.business_dir / "orders.csv").write_text("\n".join(rows), encoding="utf-8-sig")

    def _write_excel(self) -> None:
        workbook = Workbook()
        revenue = workbook.active
        revenue.title = "Doanh thu"
        revenue.merge_cells("A1:B1")
        revenue.merge_cells("C1:D1")
        revenue["A1"] = "Báo cáo"
        revenue["C1"] = "Tháng 1"
        revenue.append(["Mã đơn", "Ngày", "Số tiền", "Chi nhánh"])
        revenue.append(["X01", "2024-01-01", "1.250.000 đ", " Hà Nội "])
        revenue.append(["X02", "2024-01-02", "500k", "Đà Nẵng"])
        revenue.append(["Tổng cộng", "", "1.750.000 đ", ""])
        inventory = workbook.create_sheet("Tồn kho")
        inventory.append(["Danh mục tồn kho", "", ""])
        inventory.append(["Mã SP", "Kho", "Kho"])
        inventory.append(["SP01", "Miền Bắc", "K1"])
        inventory.append(["SP02", "Miền Nam", "K2"])
        workbook.create_sheet("Rỗng")
        workbook.save(self.business_dir / "bao cao.xlsx")

    def _write_sqlite(self) -> None:
        path = self.business_dir / "crm.sqlite"
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE customers (customer_id TEXT, city TEXT, join_date TEXT)")
        connection.executemany(
            "INSERT INTO customers VALUES (?, ?, ?)",
            [("C01", "Hà Nội", "2024-01-01"), ("C02", "Đà Nẵng", "2024-02-01")],
        )
        connection.commit()
        connection.close()

    def _build_store(self) -> BusinessDuckDBStore:
        self._write_csv()
        self._write_excel()
        self._write_sqlite()
        return BusinessDuckDBStore(
            self.business_dir,
            self.calendar_path,
            cache_root=self.cache_root,
        )

    def test_loads_csv_with_vietnamese_types_and_removes_summary(self) -> None:
        store = self._build_store()
        try:
            self.assertEqual(store.database_path, self.cache_root / "doanh_nghiep_01.duckdb")
            rows = store.connection.execute(
                "SELECT ma_don, ngay, so_tien, so_tien__raw, trang_thai FROM orders ORDER BY ma_don"
            ).fetchall()
            self.assertEqual(len(rows), 5)
            self.assertEqual(rows[0][2], 1250000)
            self.assertEqual(rows[1][2], 500000)
            self.assertEqual(rows[2][2], 0.125)
            self.assertIsNone(rows[4][2])
            self.assertEqual(rows[4][3], "không rõ")
            self.assertEqual(rows[0][4], "Hoàn thành")
            metadata = store.tables["orders"]
            self.assertEqual(metadata.coerced_columns["so_tien"]["target_type"], "DOUBLE")
            self.assertEqual(metadata.coerced_columns["ngay"]["target_type"], "DATE")
            self.assertTrue(any(item.startswith("partial_type_parse:so_tien") for item in metadata.etl_warnings))
        finally:
            database_path = store.database_path
            store.close_and_cleanup()
        self.assertFalse(database_path.exists())

    def test_loads_nonempty_excel_sheets_with_unique_ascii_names(self) -> None:
        store = self._build_store()
        try:
            self.assertIn("bao_cao__doanh_thu", store.tables)
            self.assertIn("bao_cao__ton_kho", store.tables)
            self.assertNotIn("bao_cao__rong", store.tables)
            inventory_columns = [
                row[0]
                for row in store.connection.execute("DESCRIBE bao_cao__ton_kho").fetchall()
            ]
            self.assertEqual(inventory_columns, ["ma_sp", "kho", "kho_2"])
            revenue = store.connection.execute(
                "SELECT ma_don, so_tien, chi_nhanh FROM bao_cao__doanh_thu ORDER BY ma_don"
            ).fetchall()
            self.assertEqual(revenue, [("X01", 1250000, "Hà Nội"), ("X02", 500000, "Đà Nẵng")])
            metadata = store.tables["bao_cao__doanh_thu"]
            self.assertEqual(metadata.source_sheet, "Doanh thu")
            self.assertEqual(metadata.original_columns["so_tien"], "Số tiền")
            self.assertIn("shifted_header_detected", metadata.etl_warnings)
            self.assertIn("complex_merged_cells", metadata.etl_warnings)
        finally:
            store.close_and_cleanup()

    def test_loads_sqlite_and_shared_calendar(self) -> None:
        store = self._build_store()
        try:
            self.assertEqual(
                store.connection.execute("SELECT COUNT(*) FROM crm__customers").fetchone()[0],
                2,
            )
            self.assertEqual(
                store.connection.execute("SELECT MIN(date), MAX(date) FROM _calendar").fetchone(),
                (date(2024, 1, 1), date(2024, 2, 10)),
            )
            self.assertEqual(store.tables["crm__customers"].source_table, "customers")
        finally:
            store.close_and_cleanup()

    def test_schema_card_is_deterministic_and_enriched(self) -> None:
        store = self._build_store()
        try:
            first = build_sql_schema_card(store, " Doanh nghiệp thử nghiệm\n bán lẻ ")
            second = build_sql_schema_card(store, " Doanh nghiệp thử nghiệm\n bán lẻ ")
            self.assertEqual(first, second)
            self.assertIn("README: Doanh nghiệp thử nghiệm bán lẻ", first)
            self.assertIn("TABLE orders", first)
            self.assertIn('original="Số tiền"', first)
            self.assertIn('values=["Hoàn thành","Đang xử lý","Đã hủy"]', first)
            self.assertIn("range=0.125..1250000.0", first)
            self.assertIn("ROW {", first)
            self.assertIn("WARNINGS", first)
            self.assertIn("TABLE _calendar rows=2", first)
            self.assertIn("date=2024-01-01..2024-02-10", first)
        finally:
            store.close_and_cleanup()


if __name__ == "__main__":
    unittest.main()
