from __future__ import annotations

import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from src import calendar_utils
from src.duckdb_store import BusinessDuckDBStore


class TestCalendarSingleSourceOfTruth(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)

        rows = []
        cur = date(2023, 1, 1)
        end_d = date(2024, 12, 31)
        while cur <= end_d:
            holiday_val = ""
            if date(2024, 2, 8) <= cur <= date(2024, 2, 14):
                holiday_val = "Nghỉ Tết Nguyên Đán"
            elif date(2023, 1, 20) <= cur <= date(2023, 1, 26):
                holiday_val = "Nghỉ Tết Quý Mão"

            if cur == date(2024, 2, 2):
                l_year, l_month, l_day = 2023, 12, 23
            elif cur == date(2024, 2, 19):
                l_year, l_month, l_day = 2024, 1, 10
            elif cur == date(2024, 2, 10):
                l_year, l_month, l_day = 2024, 1, 1
            else:
                l_year, l_month, l_day = cur.year, 6, cur.day

            rows.append({
                "date": cur.isoformat(),
                "lunar_date": f"{l_year:04d}-{l_month:02d}-{l_day:02d}",
                "holiday": holiday_val,
            })
            cur += timedelta(days=1)

        self.cal_path = self.base_dir / "calendar.csv"
        pd.DataFrame(rows).to_csv(self.cal_path, index=False)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_split_1_monday_start_mtd_and_event_tet(self) -> None:
        rulebook_1 = (
            "Tuần bắt đầu từ thứ Hai. "
            "Tháng này tính từ đầu tháng đến ngày as_of_date. "
            "Dịp Tết tính theo các ngày nghỉ Tết trong lịch."
        )
        time_rules = calendar_utils.parse_time_rules_from_rulebook(rulebook_1)
        self.assertEqual(time_rules["week_start"], "monday")
        self.assertEqual(time_rules["this_month_definition"], "mtd")

        cal_df = calendar_utils.load_normalized_calendar_df(self.cal_path)
        self.assertIn("solar_date", cal_df.columns)
        self.assertIn("iso_weekday", cal_df.columns)
        self.assertIn("weekday_vi", cal_df.columns)
        self.assertIn("lunar_year", cal_df.columns)

        biz_dir = self.base_dir / "biz_1"
        biz_dir.mkdir()
        (biz_dir / "sales.csv").write_text("id,amount\n1,100\n", encoding="utf-8")
        store = BusinessDuckDBStore(
            biz_dir,
            calendar_csv_path=self.cal_path,
            cache_root=self.base_dir / "cache_1",
        )

        try:
            as_of_date = "2024-06-19"

            py_this_month = calendar_utils.resolve_relative_period("this_month", as_of_date, cal_df, time_rules)
            sql_this_month = store.connection.execute(
                "SELECT MIN(solar_date)::TEXT, MAX(solar_date)::TEXT "
                "FROM _calendar WHERE solar_year = 2024 AND solar_month = 6 AND solar_date <= '2024-06-19'"
            ).fetchone()
            self.assertEqual(py_this_month, ("2024-06-01", "2024-06-19"))
            self.assertEqual(py_this_month, sql_this_month)

            py_last_week = calendar_utils.resolve_relative_period("tuần rồi", as_of_date, cal_df, time_rules)
            sql_last_week = store.connection.execute(
                "SELECT MIN(solar_date)::TEXT, MAX(solar_date)::TEXT "
                "FROM _calendar WHERE solar_date BETWEEN '2024-06-10' AND '2024-06-16'"
            ).fetchone()
            self.assertEqual(py_last_week, ("2024-06-10", "2024-06-16"))
            self.assertEqual(py_last_week, sql_last_week)

            py_sp_ly = calendar_utils.resolve_relative_period("cùng kỳ năm ngoái", as_of_date, cal_df, time_rules)
            sql_sp_ly = store.connection.execute(
                "SELECT MIN(solar_date)::TEXT, MAX(solar_date)::TEXT "
                "FROM _calendar WHERE solar_year = 2023 AND solar_month = 6 AND solar_date <= '2023-06-19'"
            ).fetchone()
            self.assertEqual(py_sp_ly, ("2023-06-01", "2023-06-19"))
            self.assertEqual(py_sp_ly, sql_sp_ly)

            py_tet = calendar_utils.resolve_relative_period("tet_current_year", as_of_date, cal_df, time_rules)
            sql_tet = store.connection.execute(
                "SELECT MIN(solar_date)::TEXT, MAX(solar_date)::TEXT "
                "FROM _calendar WHERE solar_year = 2024 AND holiday ILIKE '%tết%'"
            ).fetchone()
            self.assertEqual(py_tet, ("2024-02-08", "2024-02-14"))
            self.assertEqual(py_tet, sql_tet)

        finally:
            store.close_and_cleanup()

    def test_split_2_sunday_start_full_month_and_lunar_window_tet(self) -> None:
        rulebook_2 = (
            "Tuần bắt đầu từ Chủ Nhật đến thứ Bảy. "
            "Tháng này tính toàn bộ tháng. "
            "Dịp Tết tính từ 23 tháng Chạp đến mùng 10 tháng Giêng."
        )
        time_rules = calendar_utils.parse_time_rules_from_rulebook(rulebook_2)
        self.assertEqual(time_rules["week_start"], "sunday")
        self.assertEqual(time_rules["this_month_definition"], "full_month")
        self.assertEqual(time_rules["tet_lunar_start"], (12, 23))
        self.assertEqual(time_rules["tet_lunar_end"], (1, 10))

        cal_df = calendar_utils.load_normalized_calendar_df(self.cal_path)
        biz_dir = self.base_dir / "biz_2"
        biz_dir.mkdir()
        (biz_dir / "sales.csv").write_text("id,amount\n1,200\n", encoding="utf-8")
        store = BusinessDuckDBStore(
            biz_dir,
            calendar_csv_path=self.cal_path,
            cache_root=self.base_dir / "cache_2",
        )

        try:
            as_of_date = "2024-06-19"

            py_this_month = calendar_utils.resolve_relative_period("this_month", as_of_date, cal_df, time_rules)
            sql_this_month = store.connection.execute(
                "SELECT MIN(solar_date)::TEXT, MAX(solar_date)::TEXT "
                "FROM _calendar WHERE solar_year = 2024 AND solar_month = 6"
            ).fetchone()
            self.assertEqual(py_this_month, ("2024-06-01", "2024-06-30"))
            self.assertEqual(py_this_month, sql_this_month)

            py_last_week = calendar_utils.resolve_relative_period("tuần rồi", as_of_date, cal_df, time_rules)
            sql_last_week = store.connection.execute(
                "SELECT MIN(solar_date)::TEXT, MAX(solar_date)::TEXT "
                "FROM _calendar WHERE solar_date BETWEEN '2024-06-09' AND '2024-06-15'"
            ).fetchone()
            self.assertEqual(py_last_week, ("2024-06-09", "2024-06-15"))
            self.assertEqual(py_last_week, sql_last_week)

            py_sp_ly = calendar_utils.resolve_relative_period("cùng kỳ năm ngoái", as_of_date, cal_df, time_rules)
            sql_sp_ly = store.connection.execute(
                "SELECT MIN(solar_date)::TEXT, MAX(solar_date)::TEXT "
                "FROM _calendar WHERE solar_year = 2023 AND solar_month = 6"
            ).fetchone()
            self.assertEqual(py_sp_ly, ("2023-06-01", "2023-06-30"))
            self.assertEqual(py_sp_ly, sql_sp_ly)

            py_tet = calendar_utils.resolve_relative_period("tet_current_year", as_of_date, cal_df, time_rules)
            sql_tet = store.connection.execute(
                "SELECT MIN(solar_date)::TEXT, MAX(solar_date)::TEXT "
                "FROM _calendar WHERE (lunar_year = 2023 AND lunar_month = 12 AND lunar_day >= 23) "
                "OR (lunar_year = 2024 AND lunar_month = 1 AND lunar_day <= 10)"
            ).fetchone()
            self.assertEqual(py_tet, ("2024-02-02", "2024-02-19"))
            self.assertEqual(py_tet, sql_tet)

        finally:
            store.close_and_cleanup()

    def test_duckdb_calendar_columns_and_weekday_mappings(self) -> None:
        cal_df = calendar_utils.load_normalized_calendar_df(self.cal_path)
        biz_dir = self.base_dir / "biz_3"
        biz_dir.mkdir()
        (biz_dir / "sales.csv").write_text("id,amount\n1,300\n", encoding="utf-8")
        store = BusinessDuckDBStore(
            biz_dir,
            calendar_csv_path=self.cal_path,
            cache_root=self.base_dir / "cache_3",
        )

        try:
            sample = store.connection.execute(
                "SELECT solar_date, iso_weekday, weekday_vi FROM _calendar WHERE solar_date = '2024-06-17'"
            ).fetchone()
            self.assertEqual(sample[0], date(2024, 6, 17))
            self.assertEqual(sample[1], 1)
            self.assertEqual(sample[2], "Thứ 2")

            sunday_sample = store.connection.execute(
                "SELECT solar_date, iso_weekday, weekday_vi FROM _calendar WHERE solar_date = '2024-06-16'"
            ).fetchone()
            self.assertEqual(sunday_sample[0], date(2024, 6, 16))
            self.assertEqual(sunday_sample[1], 7)
            self.assertEqual(sunday_sample[2], "Chủ nhật")

        finally:
            store.close_and_cleanup()


if __name__ == "__main__":
    unittest.main()
