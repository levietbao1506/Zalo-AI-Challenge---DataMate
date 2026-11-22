"""Tests for Join-Key Profiling, BusinessEntityIndex probing, and Adaptive Hybrid integration."""

from __future__ import annotations

from pathlib import Path
import tempfile
import time
import unittest

import pandas as pd

from src.agent import CodeAgent, execute_adaptive_hybrid
from src.duckdb_store import BusinessDuckDBStore
from src.entity_probe import BusinessEntityIndex
from src.prompt_builder import (
    PromptBuilder,
    build_business_prefix,
    build_python_prompt,
    build_sql_prompt,
)
from src.schema_parser import build_sql_schema_card
from src.sql_executor import DuckDBSQLExecutor


class TestEntityProbeAndJoins(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir_obj = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp_dir_obj.name)
        self.cache_dir = self.tmp_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        orders_df = pd.DataFrame({
            "order_id": [1, 2, 3, 4],
            "product_id": ["P01", "P01", "P02", "P02"],
            "so_luong": [10, 5, 2, 1],
        })
        orders_df.to_csv(self.tmp_dir / "orders.csv", index=False)

        products_df = pd.DataFrame({
            "product_id": ["P01", "P02", "P03"],
            "product_name": [
                "Xi măng Hà Tiên PCB40 bao 50kg",
                "Sting dâu 330ml",
                "Bia Tiger Crystal lon 330ml",
            ],
            "danh_muc": ["Vật liệu", "Nước giải khát", "Đồ uống"],
        })
        products_df.to_excel(self.tmp_dir / "products.xlsx", index=False)

    def tearDown(self) -> None:
        self.tmp_dir_obj.cleanup()

    def test_many_to_one_join_detection_and_schema_card(self) -> None:
        store = BusinessDuckDBStore(
            business_dir=self.tmp_dir,
            calendar_csv_path=None,
            cache_root=self.cache_dir,
        )
        try:
            self.assertIn("orders", store.candidate_unique_keys)
            self.assertIn("order_id", store.candidate_unique_keys["orders"])
            self.assertNotIn("product_id", store.candidate_unique_keys["orders"])

            products_table_key = [t for t in store.tables if "products" in t][0]
            self.assertIn("product_id", store.candidate_unique_keys[products_table_key])

            joins = store.join_hints
            matching = [
                j for j in joins
                if j["left_table"] == "orders"
                and j["left_col"] == "product_id"
                and j["right_table"] == products_table_key
                and j["right_col"] == "product_id"
            ]
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]["relationship_type"], "many_to_one")
            self.assertGreaterEqual(matching[0]["overlap_ratio"], 0.99)

            card = build_sql_schema_card(store, "README context")
            self.assertIn("## Key & Join Hints", card)
            self.assertIn("Unique Keys:", card)
            self.assertIn("order_id", card)
            self.assertIn("many_to_one", card)
            self.assertNotIn("[WARN: many-to-many, aggregate before join]", card)
        finally:
            store.close_and_cleanup()

    def test_many_to_many_fan_out_warning(self) -> None:
        dup_dir_obj = tempfile.TemporaryDirectory()
        dup_dir = Path(dup_dir_obj.name)
        dup_cache = dup_dir / "cache"
        dup_cache.mkdir(parents=True, exist_ok=True)
        try:
            tbl_a = pd.DataFrame({
                "item_code": ["A1", "A1", "A2"],
                "qty": [1, 2, 3],
            })
            tbl_b = pd.DataFrame({
                "item_code": ["A1", "A1", "A2"],
                "price": [100, 105, 200],
            })
            tbl_a.to_csv(dup_dir / "sales.csv", index=False)
            tbl_b.to_csv(dup_dir / "prices.csv", index=False)

            store = BusinessDuckDBStore(
                business_dir=dup_dir,
                calendar_csv_path=None,
                cache_root=dup_cache,
            )
            try:
                card = build_sql_schema_card(store, "")
                self.assertIn("## Key & Join Hints", card)
                self.assertIn("many_to_many", card)
                self.assertIn("[WARN: many-to-many, aggregate before join]", card)
            finally:
                store.close_and_cleanup()
        finally:
            dup_dir_obj.cleanup()

    def test_entity_index_probe_accurate_and_empty_on_general(self) -> None:
        store = BusinessDuckDBStore(
            business_dir=self.tmp_dir,
            calendar_csv_path=None,
            cache_root=self.cache_dir,
        )
        try:
            index = BusinessEntityIndex(store)

            probed_xm = index.probe("xi mang ha tien ban chay nhat")
            self.assertGreater(len(probed_xm), 0)
            self.assertEqual(probed_xm[0]["value"], "Xi măng Hà Tiên PCB40 bao 50kg")
            self.assertIn("products", probed_xm[0]["table"])

            probed_sting = index.probe("sting dau")
            self.assertGreater(len(probed_sting), 0)
            self.assertEqual(probed_sting[0]["value"], "Sting dâu 330ml")

            probed_abbrev = index.probe("xi mang ht")
            self.assertGreater(len(probed_abbrev), 0)
            self.assertEqual(probed_abbrev[0]["value"], "Xi măng Hà Tiên PCB40 bao 50kg")

            general_1 = index.probe("Tong doanh thu cua tat ca cac don hang trong nam la bao nhieu?")
            self.assertEqual(general_1, [])

            general_2 = index.probe("Thang nao co so luong ban chay nhat?")
            self.assertEqual(general_2, [])
        finally:
            store.close_and_cleanup()

    def test_prefix_caching_byte_by_byte_identity(self) -> None:
        store = BusinessDuckDBStore(
            business_dir=self.tmp_dir,
            calendar_csv_path=None,
            cache_root=self.cache_dir,
        )
        try:
            index = BusinessEntityIndex(store)
            schema_card = build_sql_schema_card(store, "Sample README")
            rulebook = "Rulebook sample text."
            readme = "Sample README"

            prefix_q1 = build_business_prefix(rulebook, readme, schema_card)
            prefix_q2 = build_business_prefix(rulebook, readme, schema_card)

            hints1 = index.probe("xi mang ha tien ban chay nhat")
            hints2 = index.probe("sting dau")
            self.assertNotEqual(hints1, hints2)

            sql_prompt1 = build_sql_prompt("xi mang ha tien ban chay nhat", "2024-06-15", entity_hints=hints1)
            sql_prompt2 = build_sql_prompt("sting dau", "2024-06-15", entity_hints=hints2)

            self.assertIn("Matched Entity Hints (exact values found in tables):", sql_prompt1)
            self.assertIn("Xi măng Hà Tiên PCB40 bao 50kg", sql_prompt1)
            self.assertIn("Sting dâu 330ml", sql_prompt2)

            self.assertEqual(prefix_q1.encode("utf-8"), prefix_q2.encode("utf-8"))
            self.assertEqual(len(prefix_q1.encode("utf-8")), len(prefix_q2.encode("utf-8")))

            pb = PromptBuilder(rulebook_content=rulebook)
            py_prompt1 = pb.build_python_prompt(
                question="xi mang ha tien",
                as_of_date="2024-06-15",
                business_id="test_biz",
                entity_hints=hints1,
            )
            self.assertIn("Matched Entity Hints (exact values found in tables):", py_prompt1)
        finally:
            store.close_and_cleanup()

    def test_profiler_and_probe_performance(self) -> None:
        store = BusinessDuckDBStore(
            business_dir=self.tmp_dir,
            calendar_csv_path=None,
            cache_root=self.cache_dir,
        )
        try:
            t0 = time.perf_counter()
            hints = store.profile_keys_and_joins()
            t_profile_ms = (time.perf_counter() - t0) * 1000
            self.assertLess(t_profile_ms, 300.0)

            index = BusinessEntityIndex(store)
            t1 = time.perf_counter()
            for _ in range(10):
                index.probe("xi mang ha tien ban chay nhat")
            t_probe_avg_ms = ((time.perf_counter() - t1) / 10.0) * 1000
            self.assertLess(t_probe_avg_ms, 5.0)
        finally:
            store.close_and_cleanup()

    def test_execute_adaptive_hybrid_records_entity_hints_count(self) -> None:
        store = BusinessDuckDBStore(
            business_dir=self.tmp_dir,
            calendar_csv_path=None,
            cache_root=self.cache_dir,
        )
        sql_exec = DuckDBSQLExecutor(store)
        try:
            index = BusinessEntityIndex(store)
            card = build_sql_schema_card(store, "")
            meta = getattr(store, "schema_meta", {})
            agent = CodeAgent(dry_run=True)

            q_record = {
                "id": "q1",
                "question": "xi mang ha tien ban chay nhat",
                "as_of_date": "2024-06-15",
                "type": "string",
            }
            res = execute_adaptive_hybrid(
                question_record=q_record,
                duckdb_store=store,
                sql_executor=sql_exec,
                schema_card=card,
                schema_meta=meta,
                rulebook_text="",
                agent=agent,
                entity_index=index,
            )
            route_trace = res["route_trace"]
            self.assertIn("entity_hints_count", route_trace)
            self.assertGreater(route_trace["entity_hints_count"], 0)
        finally:
            sql_exec.close()
            store.close_and_cleanup()


if __name__ == "__main__":
    unittest.main()
