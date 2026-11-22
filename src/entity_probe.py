"""In-memory entity indexing and fast local entity probing for tabular QA."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from src.sql_executor import vn_norm, vn_unaccent

VIETNAMESE_STOPWORDS = {
    "thang", "tuan", "nam", "ngay", "nay", "truoc", "sau", "qua", "kia",
    "bao", "nhieu", "doanh", "thu", "ban", "chay", "nhat", "la", "gi",
    "cua", "cac", "nhung", "co", "tat", "ca", "tong", "so", "luong",
    "gia", "tien", "loi", "nhuan", "chi", "phi", "trung", "binh",
    "don", "hang", "khach", "hang", "san", "pham", "mat", "hang",
    "thoi", "gian", "tu", "den", "trong", "cho", "biet", "hay", "tinh",
    "xem", "top", "cao", "thap", "tang", "giam", "ti", "ty", "le",
    "phan", "tram", "don", "gia", "thanh", "tien", "nao", "may", "va", "hoac",
    "ai", "o", "dau", "khi", "luc", "theo", "voi", "ve", "ra", "sao",
}


@dataclass
class EntityRecord:
    table: str
    column: str
    value: str
    code: Optional[str]
    norm_val: str
    unaccent_val: str
    tokens: List[str]
    token_set: Set[str]
    initials: str


class BusinessEntityIndex:
    def __init__(self, store: Any) -> None:
        self.store = store
        self.records: List[EntityRecord] = []
        self.inverted_index: Dict[str, List[int]] = {}
        self.initials_index: Dict[str, List[int]] = {}
        self._build_index()

    def _build_index(self) -> None:
        connection = getattr(self.store, "connection", None)
        if connection is None:
            return

        tables = getattr(self.store, "table_metadata", {})
        string_types = ("VARCHAR", "CHAR", "TEXT", "STRING")
        code_tokens = ("id", "ma", "code", "sku")
        id_keywords = (
            "ten", "name", "sp", "product", "sku", "ma", "code",
            "khach", "customer", "hang", "item", "brand",
            "danh_muc", "category", "branch", "chi_nhanh", "ncc", "supplier"
        )

        for table_name in sorted(tables):
            if table_name == "_calendar":
                continue
            quoted_table = '"' + table_name.replace('"', '""') + '"'
            try:
                described = connection.execute(f"DESCRIBE SELECT * FROM {quoted_table}").fetchall()
            except Exception:
                continue

            str_cols = [str(r[0]) for r in described if any(t in str(r[1]).upper() for t in string_types)]
            if not str_cols:
                continue

            all_cols = [str(r[0]) for r in described]
            candidate_code_cols = [c for c in all_cols if any(k in c.lower() for k in code_tokens)]

            for col in str_cols:
                quoted_col = '"' + col.replace('"', '""') + '"'
                try:
                    cardinality_row = connection.execute(
                        f"SELECT COUNT(DISTINCT {quoted_col}) FROM {quoted_table} WHERE {quoted_col} IS NOT NULL"
                    ).fetchone()
                    cardinality = cardinality_row[0] if cardinality_row else 0
                except Exception:
                    cardinality = 0

                col_lower = col.lower()
                has_id_kw = any(k in col_lower for k in id_keywords)
                is_eligible = (30 < cardinality <= 20000) or (has_id_kw and 2 <= cardinality <= 20000)
                if not is_eligible:
                    continue

                linked_code_col: Optional[str] = None
                if col in candidate_code_cols:
                    linked_code_col = col
                else:
                    for cc in candidate_code_cols:
                        if cc != col:
                            linked_code_col = cc
                            break

                try:
                    if linked_code_col and linked_code_col != col:
                        quoted_code = '"' + linked_code_col.replace('"', '""') + '"'
                        query = (
                            f"SELECT DISTINCT {quoted_col}, {quoted_code} FROM {quoted_table} "
                            f"WHERE {quoted_col} IS NOT NULL AND TRIM(CAST({quoted_col} AS VARCHAR)) != ''"
                        )
                        rows = connection.execute(query).fetchall()
                    else:
                        query = (
                            f"SELECT DISTINCT {quoted_col} FROM {quoted_table} "
                            f"WHERE {quoted_col} IS NOT NULL AND TRIM(CAST({quoted_col} AS VARCHAR)) != ''"
                        )
                        rows = [(r[0], None) for r in connection.execute(query).fetchall()]
                except Exception:
                    continue

                for row in rows:
                    val = str(row[0]).strip()
                    if not val:
                        continue
                    code = str(row[1]).strip() if len(row) > 1 and row[1] is not None else None
                    norm_v = vn_norm(val) or ""
                    unaccent_v = vn_unaccent(val) or ""
                    tokens = [t for t in unaccent_v.split() if t]
                    token_set = set(tokens)
                    initials = "".join(t[0] for t in tokens if t and t[0].isalpha())

                    record_id = len(self.records)
                    record = EntityRecord(
                        table=table_name,
                        column=col,
                        value=val,
                        code=code,
                        norm_val=norm_v,
                        unaccent_val=unaccent_v,
                        tokens=tokens,
                        token_set=token_set,
                        initials=initials,
                    )
                    self.records.append(record)

                    for t in token_set:
                        if len(t) >= 2:
                            self.inverted_index.setdefault(t, []).append(record_id)
                    if len(initials) >= 2:
                        self.initials_index.setdefault(initials, []).append(record_id)
                    if code:
                        code_u = vn_unaccent(code) or ""
                        for ct in code_u.split():
                            if len(ct) >= 2:
                                self.inverted_index.setdefault(ct, []).append(record_id)

    def probe(self, question: str, top_k: int = 8) -> List[Dict[str, Any]]:
        q_unaccent = vn_unaccent(question) or ""
        words = [w for w in q_unaccent.split() if w]
        non_stopwords = [
            w for w in words
            if (w not in VIETNAMESE_STOPWORDS or any(ch.isdigit() for ch in w)) and len(w) >= 2
        ]
        if not non_stopwords:
            return []

        candidate_ids: Set[int] = set()
        for w in non_stopwords:
            if w in self.inverted_index:
                candidate_ids.update(self.inverted_index[w])
            if w in self.initials_index:
                candidate_ids.update(self.initials_index[w])

        if len(non_stopwords) >= 2:
            initials_candidate = "".join(w[0] for w in non_stopwords if w and w[0].isalpha())
            if initials_candidate in self.initials_index:
                candidate_ids.update(self.initials_index[initials_candidate])

        if not candidate_ids:
            return []

        q_phrase = " ".join(non_stopwords)
        q_tokens_set = set(non_stopwords)
        scored_candidates: List[Dict[str, Any]] = []

        for rec_id in candidate_ids:
            rec = self.records[rec_id]
            e_phrase = rec.unaccent_val
            e_tokens = rec.tokens
            e_tokens_set = rec.token_set

            score = 0.0

            if e_phrase and e_phrase in q_unaccent:
                score = max(score, 1.0)
            elif q_phrase and q_phrase in e_phrase:
                char_cov = len(q_phrase) / max(1, len(e_phrase))
                score = max(score, 0.85 + 0.15 * char_cov)
            else:
                for n in (3, 2):
                    if len(non_stopwords) >= n:
                        for i in range(len(non_stopwords) - n + 1):
                            sub = " ".join(non_stopwords[i:i + n])
                            if sub in e_phrase:
                                sub_cov = len(sub) / max(1, len(e_phrase))
                                score = max(score, 0.70 + 0.15 * sub_cov)

            abbrev_matched: Set[str] = set()
            matched_entity_tokens: Set[str] = set()

            for t in non_stopwords:
                if t in e_tokens_set:
                    abbrev_matched.add(t)
                    matched_entity_tokens.add(t)
                elif len(t) >= 2:
                    t_len = len(t)
                    for i in range(len(e_tokens) - t_len + 1):
                        sub_init = "".join(e_tokens[i + k][0] for k in range(t_len) if e_tokens[i + k])
                        if t == sub_init:
                            abbrev_matched.add(t)
                            for k in range(t_len):
                                matched_entity_tokens.add(e_tokens[i + k])
                            break

            if abbrev_matched:
                query_cov = len(abbrev_matched) / len(non_stopwords)
                entity_cov = len(matched_entity_tokens) / max(1, len(e_tokens_set))
                f1 = (2 * query_cov * entity_cov) / (query_cov + entity_cov) if (query_cov + entity_cov) > 0 else 0.0
                token_score = 0.55 * query_cov + 0.35 * f1 + 0.10 * entity_cov
                score = max(score, token_score)
                if query_cov >= 0.75:
                    score = max(score, 0.75 + 0.20 * query_cov)

            if rec.code:
                code_clean = vn_unaccent(rec.code) or ""
                if code_clean and code_clean in q_unaccent:
                    score = max(score, 0.98)

            if score >= 0.65:
                scored_candidates.append({
                    "table": rec.table,
                    "column": rec.column,
                    "value": rec.value,
                    "code": rec.code,
                    "score": round(score, 4),
                })

        if not scored_candidates:
            return []

        best_by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        for item in scored_candidates:
            key = (item["table"], item["column"], item["value"])
            if key not in best_by_key or item["score"] > best_by_key[key]["score"]:
                best_by_key[key] = item

        deduped = list(best_by_key.values())
        deduped.sort(key=lambda x: (-x["score"], x["table"], x["column"], x["value"]))
        return deduped[:top_k]
