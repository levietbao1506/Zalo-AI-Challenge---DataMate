"""Unit tests for safe read-only DuckDB SQL execution."""

from datetime import date
from decimal import Decimal
from pathlib import Path
import tempfile
import time
import unittest

import duckdb

from src.duckdb_store import BusinessDuckDBStore
from src.sql_executor import DuckDBSQLExecutor


class TestDuckDBSQLExecutor(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.cache_root = Path(self.temporary.name)
        self.database_path = self.cache_root / "business_01.duckdb"
        connection = duckdb.connect(str(self.database_path))
        connection.execute("CREATE TABLE products(name VARCHAR, amount INTEGER)")
        connection.executemany(
            "INSERT INTO products VALUES (?, ?)",
            [("Cà phê sữa", 10), ("Trà đào", 20), ("Bánh mì", 30)],
        )
        connection.close()
        self.executor = DuckDBSQLExecutor(
            "business_01",
            cache_root=self.cache_root,
            memory_limit="128MB",
            threads=1,
            max_rows=2,
            timeout_seconds=0.1,
        )

    def tearDown(self) -> None:
        self.executor.close()
        self.temporary.cleanup()

    def test_blocks_writes_ddl_and_external_file_functions(self) -> None:
        queries = [
            "INSERT INTO products VALUES ('x', 1)",
            "UPDATE products SET amount = 0",
            "DELETE FROM products",
            "DROP TABLE products",
            "COPY products TO '/tmp/products.csv'",
            "ATTACH '/tmp/other.duckdb' AS other",
            "LOAD httpfs",
            "SELECT * FROM read_csv('/data/private.csv')",
            "SELECT * FROM read_parquet('/data/private.parquet')",
            "SELECT * FROM '/data/private.csv'",
            "SELECT * FROM query('SELECT * FROM read_csv(''/data/private.csv'')')",
            "SELECT 1; DROP TABLE products",
            "WITH deleted AS (DELETE FROM products RETURNING *) SELECT * FROM deleted",
        ]
        for query in queries:
            with self.subTest(query=query):
                result = self.executor.execute(query)
                self.assertFalse(result.success)
                self.assertIn("UnsafeSQL", result.error or "")
        result = self.executor.execute("SELECT COUNT(*) FROM products")
        self.assertTrue(result.success)
        self.assertEqual(result.rows, [[3]])

    def test_timeout_interrupts_hanging_query_and_connection_recovers(self) -> None:
        query = (
            "WITH RECURSIVE endless(x) AS ("
            "SELECT 1 UNION ALL SELECT x + 1 FROM endless"
            ") SELECT SUM(x) FROM endless"
        )
        started = time.perf_counter()
        result = self.executor.execute(query, timeout_seconds=0.05)
        elapsed = time.perf_counter() - started
        self.assertFalse(result.success)
        self.assertIn("TimeoutError", result.error or "")
        self.assertLess(elapsed, 2.0)
        recovered = self.executor.execute("SELECT 42 AS answer")
        self.assertTrue(recovered.success)
        self.assertEqual(recovered.rows, [[42]])

    def test_vietnamese_normalization_functions(self) -> None:
        result = self.executor.execute(
            "SELECT vn_norm('  Cà phê, SỮA!!!  ') AS normalized, "
            "vn_unaccent('Cà phê sữa') AS unaccented"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.rows, [["cà phê sữa", "ca phe sua"]])
        match = self.executor.execute(
            "SELECT name FROM products WHERE vn_unaccent(name) = 'ca phe sua'"
        )
        self.assertEqual(match.rows, [["Cà phê sữa"]])

    def test_detects_all_result_anomalies(self) -> None:
        empty = self.executor.execute("SELECT * FROM products WHERE FALSE")
        self.assertEqual(empty.anomalies, ["empty_result"])
        all_null = self.executor.execute("SELECT NULL AS value")
        self.assertEqual(all_null.anomalies, ["all_null"])
        zero = self.executor.execute("SELECT 0.0 AS value")
        self.assertEqual(zero.anomalies, ["zero_scalar"])
        extreme = self.executor.execute(
            "SELECT 1200.0 AS percentage, 'NaN'::DOUBLE AS other_value"
        )
        self.assertIn("suspicious_extreme", extreme.anomalies)
        self.assertEqual(extreme.rows, [[1200.0, None]])
        truncated = self.executor.execute("SELECT range AS id FROM range(5) ORDER BY range")
        self.assertTrue(truncated.truncated)
        self.assertEqual(truncated.row_count, 2)
        self.assertEqual(truncated.rows, [[0], [1]])
        self.assertIn("truncated_rows", truncated.anomalies)

    def test_converts_decimal_date_datetime_and_numpy_scalars(self) -> None:
        result = self.executor.execute(
            "SELECT 12.50::DECIMAL(10,2) AS money, "
            "DATE '2024-01-02' AS business_date, "
            "TIMESTAMP '2024-01-02 03:04:05' AS created_at"
        )
        self.assertTrue(result.success)
        self.assertEqual(result.rows, [[12.5, "2024-01-02", "2024-01-02T03:04:05"]])
        self.assertIsInstance(result.rows[0][0], float)
        self.assertEqual(
            self.executor._to_json_value(Decimal("7")),
            7,
        )
        self.assertEqual(
            self.executor._to_json_value(date(2024, 2, 3)),
            "2024-02-03",
        )

    def test_opens_alongside_completed_business_store(self) -> None:
        business_dir = self.cache_root / "integrated_business"
        business_dir.mkdir()
        (business_dir / "orders.csv").write_text(
            "order_id,amount\nA01,100\nA02,200\n",
            encoding="utf-8",
        )
        store = BusinessDuckDBStore(business_dir, None, cache_root=self.cache_root)
        executor = DuckDBSQLExecutor("integrated_business", cache_root=self.cache_root)
        try:
            result = executor.execute("WITH totals AS (SELECT SUM(amount) AS value FROM orders) SELECT value FROM totals")
            self.assertTrue(result.success)
            self.assertEqual(result.rows, [[300]])
        finally:
            executor.close()
            store.close_and_cleanup()


if __name__ == "__main__":
    unittest.main()
