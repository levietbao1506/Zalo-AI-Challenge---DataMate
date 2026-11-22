"""Deterministic routing rules for the Adaptive Hybrid execution architecture."""

from __future__ import annotations

from enum import Enum
import math
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


class RouteDecision(str, Enum):
    SQL_FAST = "SQL_FAST"
    HYBRID_DUAL = "HYBRID_DUAL"
    PYTHON_FIRST = "PYTHON_FIRST"


_SEQUENTIAL_PATTERNS = (
    r"\bbinh quan gia quyen\b",
    r"\bweighted average(?: cost)?\b",
    r"\b(?:fifo|lifo|wac)\b",
    r"\bton kho luy ke\b",
    r"\brunning (?:stock|inventory|balance)\b",
    r"\btheo tung lo (?:nhap|xuat)\b",
    r"\bphan bo chi phi\b",
    r"\bcost allocation\b",
)
_PROFIT_COST_PATTERNS = (
    r"\bloi nhuan\b",
    r"\blai gop\b",
    r"\bbien loi nhuan\b",
    r"\bgia von\b",
    r"\bcogs\b",
    r"\bgvhb\b",
    r"\bln(?:g)?\b",
    r"\blai\b",
    r"\bprofit(?: margin)?\b",
    r"\bgross (?:profit|margin)\b",
)
_OTHER_RISKY_BUSINESS_PATTERNS = (
    r"\bhoan tra\b",
    r"\bdoi tra\b",
    r"\btra hang\b",
    r"\brefund\b",
    r"\breturn(?:ed|s)?\b",
    r"\bcong no qua han\b",
    r"\boverdue\b",
    r"\btang truong\b",
    r"\bgrowth\b",
    r"\bphan tram\b",
    r"\bpercent(?:age)?\b",
    r"\bso voi (?:tuan|thang|quy|nam) truoc\b",
    r"\bnam ngoai\b",
)
_DYNAMIC_CALENDAR_PATTERNS = (
    r"\btet\b",
    r"\bam lich\b",
    r"\btrung thu\b",
    r"\bmung\s+\d+\b",
    r"\bthang chap\b",
    r"\bcung ky\b",
    r"\blunar\b",
)
_RANKING_PATTERNS = (
    r"\btop\s*-?\s*\d+\b",
    r"\bxep hang\b",
    r"\bcao nhat\b",
    r"\bthap nhat\b",
    r"\bnhieu nhat\b",
    r"\bit nhat\b",
    r"\blon nhat\b",
    r"\bnho nhat\b",
)
_TABLE_PATTERNS = (
    r"\bdang bang\b",
    r"\blap bang\b",
    r"\bbang ket qua\b",
    r"\boutput table\b",
    r"\btra ve table\b",
    r"\btra ve (?:dang )?bang\b",
)
_NESTED_FILTER_PATTERNS = (
    r"\btrong so\b",
    r"\bsau khi\b",
    r"\bdong thoi\b",
    r"\bngoai tru\b",
    r"\bchi nhung\b",
    r"\bvua .+ vua\b",
)
_TIME_PATTERNS = (
    r"\b\d{1,2}/\d{1,2}/\d{2,4}\b",
    r"\b\d{4}-\d{1,2}-\d{1,2}\b",
    r"\bthang\s+\d{1,2}(?:/|\s+nam\s+)\d{4}\b",
    r"\bquy\s+[1-4](?:/|\s+nam\s+)\d{4}\b",
    r"\b(?:tuan|thang|quy|nam)\s+(?:nay|truoc|roi|ngoai)\b",
    r"\bhom (?:nay|qua)\b",
    r"\bcung ky\b",
    r"\btu\s+.+?\s+den\s+.+",
)


def classify_question_route(
    question: str,
    as_of_date: str,
    schema_meta: dict,
    rulebook_text: str,
    time_pressure: bool = False,
) -> Tuple[RouteDecision, List[str]]:
    normalized = _normalize(question)
    warnings = _extract_etl_warnings(schema_meta)
    severe_warnings = _severe_etl_warnings(warnings)
    if severe_warnings:
        reasons = [f"severe_etl_warning:{warning}" for warning in severe_warnings]
        return RouteDecision.PYTHON_FIRST, reasons
    sequential_reason = _first_match_reason(normalized, _SEQUENTIAL_PATTERNS)
    if sequential_reason:
        return RouteDecision.PYTHON_FIRST, [f"sequential_business_logic:{sequential_reason}"]

    hybrid_reasons: List[str] = []
    retained_under_pressure: Set[str] = set()
    profit_cost = _matches_any(normalized, _PROFIT_COST_PATTERNS)
    other_business_risk = _matches_any(normalized, _OTHER_RISKY_BUSINESS_PATTERNS) or "%" in question
    if profit_cost:
        hybrid_reasons.append("high_risk_business_metric:profit_or_cogs")
        retained_under_pressure.add("high_risk_business_metric:profit_or_cogs")
    if other_business_risk:
        hybrid_reasons.append("high_risk_business_metric:rulebook_formula")
    if (profit_cost or other_business_risk) and not str(rulebook_text or "").strip():
        hybrid_reasons.append("missing_rulebook_for_sensitive_metric")
    if _matches_any(normalized, _DYNAMIC_CALENDAR_PATTERNS):
        hybrid_reasons.append("dynamic_calendar_logic")
    if _has_relative_time(normalized) and not str(as_of_date or "").strip():
        hybrid_reasons.append("missing_as_of_date")

    missing_concepts = _missing_schema_concepts(normalized, schema_meta)
    for concept in missing_concepts:
        reason = f"schema_missing:{concept}"
        hybrid_reasons.append(reason)
        retained_under_pressure.add(reason)

    ranking = _matches_any(normalized, _RANKING_PATTERNS)
    nested_filter_count = sum(bool(re.search(pattern, normalized)) for pattern in _NESTED_FILTER_PATTERNS)
    conjunction_count = len(re.findall(r"\b(?:va|dong thoi|nhung|voi dieu kien)\b", normalized))
    if _matches_any(normalized, _TABLE_PATTERNS):
        hybrid_reasons.append("complex_table_output")
    multi_metric_ranking = ranking and conjunction_count >= 1 and bool(re.search(r"\btheo\b", normalized))
    if ranking and (nested_filter_count + conjunction_count >= 2 or multi_metric_ranking):
        hybrid_reasons.append("complex_multi_condition_ranking")
    if _count_time_constraints(normalized) >= 2:
        hybrid_reasons.append("multiple_time_constraints")
    if nested_filter_count >= 2:
        hybrid_reasons.append("nested_filters")
    if warnings:
        hybrid_reasons.append("etl_warning_present")
        retained_under_pressure.add("etl_warning_present")

    hybrid_reasons = _unique(hybrid_reasons)
    if hybrid_reasons and time_pressure:
        retained = [reason for reason in hybrid_reasons if reason in retained_under_pressure]
        if retained:
            return RouteDecision.HYBRID_DUAL, [*retained, "time_pressure_retained_high_penalty_risk"]
        return RouteDecision.SQL_FAST, [*hybrid_reasons, "time_pressure_downgrade_to_sql_fast"]
    if hybrid_reasons:
        return RouteDecision.HYBRID_DUAL, hybrid_reasons
    return RouteDecision.SQL_FAST, ["clean_schema_sql_candidate"]


def should_escalate_sql_result(
    sql_outcome: dict,
    question: str,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    success = bool(_outcome_value(sql_outcome, "success", False))
    if not success:
        reasons.append("sql_execution_failed_after_repair")
    anomalies = _as_string_list(_outcome_value(sql_outcome, "anomalies", []))
    boolean_question = _is_boolean_question(question, sql_outcome)
    for anomaly in ("empty_result", "all_null", "zero_scalar", "suspicious_extreme", "truncated_rows"):
        if anomaly in anomalies and not (anomaly == "zero_scalar" and boolean_question):
            reasons.append(f"sql_anomaly:{anomaly}")
    if _contains_abstain_signal(sql_outcome):
        reasons.append("sql_branch_abstain_or_unanswerable")
    if _percent_scale_suspected(sql_outcome, question):
        reasons.append("percent_scale_suspected")
    reasons.extend(_ranking_escalation_reasons(sql_outcome, question))
    is_time_question = bool(
        re.search(
            r"thang|tuan|nam|ngay|ky|giai doan|khoang thoi gian|hom nay|hom qua|thoi gian",
            _normalize(question),
        )
    )
    is_unans = bool(
        _outcome_value(sql_outcome, "is_unanswerable", False)
        or _contains_abstain_signal(sql_outcome)
    )
    is_empty = "empty_result" in anomalies or not _outcome_value(sql_outcome, "rows", [])
    if is_time_question and (is_unans or is_empty):
        reasons.append("check_date_coverage_in_python")
    reasons = _unique(reasons)
    return bool(reasons), reasons


def _normalize(value: Any) -> str:
    text = unicodedata.normalize("NFD", str(value or "").lower().replace("đ", "d"))
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9_%]+", " ", text)
    return " ".join(text.split())


def _matches_any(text: str, patterns: Iterable[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _first_match_reason(text: str, patterns: Iterable[str]) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).replace(" ", "_")
    return None


def _extract_etl_warnings(schema_meta: Any) -> List[str]:
    warnings: List[str] = []
    seen: Set[int] = set()

    def visit(value: Any) -> None:
        identity = id(value)
        if isinstance(value, (dict, list, tuple, set)):
            if identity in seen:
                return
            seen.add(identity)
        if isinstance(value, dict):
            for key, nested in value.items():
                if _normalize(key) == "etl_warnings":
                    warnings.extend(_as_string_list(nested))
                else:
                    visit(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested)

    visit(schema_meta)
    return sorted(set(warning.strip() for warning in warnings if warning.strip()))


def _severe_etl_warnings(warnings: List[str]) -> List[str]:
    severe: List[str] = []
    partial_parse_count = 0
    parse_error_count = 0
    for warning in warnings:
        normalized = _normalize(warning)
        if "partial_type_parse" in warning:
            partial_parse_count += 1
        if "parse" in normalized and ("error" in normalized or "loi" in normalized):
            parse_error_count += 1
        if any(
            marker in normalized
            for marker in (
                "complex_merged_cells",
                "horizontal_pivot",
                "wide_table",
                "high_type_parse_errors",
                "many_parse_errors",
            )
        ) or ("excel" in normalized and any(marker in normalized for marker in ("irregular", "complex", "merged", "pivot"))):
            severe.append(warning)
    if partial_parse_count >= 2:
        severe.extend(warning for warning in warnings if "partial_type_parse" in warning)
    if parse_error_count >= 2:
        severe.extend(
            warning
            for warning in warnings
            if "parse" in _normalize(warning) and ("error" in _normalize(warning) or "loi" in _normalize(warning))
        )
    return sorted(set(severe))


def _extract_schema_columns(schema_meta: Any) -> Set[str]:
    columns: Set[str] = set()
    seen: Set[int] = set()
    column_keys = {"columns", "column_names", "original_columns", "column_name_maps", "coerced_columns"}

    def consume(value: Any) -> None:
        if isinstance(value, str):
            columns.add(_normalize(value).replace(" ", "_"))
        elif isinstance(value, dict):
            for key, nested in value.items():
                columns.add(_normalize(key).replace(" ", "_"))
                if isinstance(nested, str):
                    columns.add(_normalize(nested).replace(" ", "_"))
                elif isinstance(nested, dict):
                    for field in ("name", "sql_name", "original_name", "original"):
                        if isinstance(nested.get(field), str):
                            columns.add(_normalize(nested[field]).replace(" ", "_"))
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                consume(item)

    def visit(value: Any) -> None:
        identity = id(value)
        if isinstance(value, (dict, list, tuple, set)):
            if identity in seen:
                return
            seen.add(identity)
        if isinstance(value, dict):
            for key, nested in value.items():
                normalized_key = _normalize(key).replace(" ", "_")
                if normalized_key in column_keys:
                    consume(nested)
                visit(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested)

    visit(schema_meta)
    return {column for column in columns if column}


def _missing_schema_concepts(question: str, schema_meta: dict) -> List[str]:
    columns = _extract_schema_columns(schema_meta)
    joined = " ".join(sorted(columns))
    missing: List[str] = []
    cost_present = bool(re.search(r"(?:cost|cogs|gia_von|von_hang|unit_cost|purchase_price)", joined))
    revenue_present = bool(re.search(r"(?:revenue|doanh_thu|amount|total|thanh_tien|sales)", joined))
    profit_present = bool(re.search(r"(?:profit|loi_nhuan|lai_gop|gross_profit)", joined))
    if _matches_any(question, _PROFIT_COST_PATTERNS):
        asks_cost_only = bool(re.search(r"\b(?:gia von|cogs|gvhb)\b", question)) and not bool(
            re.search(r"\b(?:loi nhuan|lai gop|bien loi nhuan|profit|margin)\b", question)
        )
        if asks_cost_only and not cost_present:
            missing.append("cost")
        elif not asks_cost_only and not (profit_present or (cost_present and revenue_present)):
            missing.append("profit_inputs")
    concepts = (
        ("customer", r"\b(?:khach hang|customer|client)\b", r"(?:customer|khach_hang|client|buyer)"),
        ("employee", r"\b(?:nhan vien|employee|staff|salesperson)\b", r"(?:employee|nhan_vien|staff|salesperson)"),
        ("branch", r"\b(?:chi nhanh|branch|store|cua hang)\b", r"(?:branch|chi_nhanh|store|cua_hang)"),
    )
    for concept, question_pattern, column_pattern in concepts:
        if re.search(question_pattern, question) and not re.search(column_pattern, joined):
            missing.append(concept)
    return _unique(missing)


def _has_relative_time(question: str) -> bool:
    return bool(re.search(r"\b(?:hom nay|hom qua|tuan nay|thang nay|quy nay|nam nay|tuan truoc|thang truoc|nam ngoai)\b", question))


def _count_time_constraints(question: str) -> int:
    spans: List[Tuple[int, int]] = []
    for pattern in _TIME_PATTERNS:
        spans.extend(match.span() for match in re.finditer(pattern, question))
    unique_spans: List[Tuple[int, int]] = []
    for span in sorted(spans):
        if not any(span[0] >= existing[0] and span[1] <= existing[1] for existing in unique_spans):
            unique_spans.append(span)
    return len(unique_spans)


def _outcome_value(outcome: Any, key: str, default: Any = None) -> Any:
    if isinstance(outcome, dict):
        return outcome.get(key, default)
    return getattr(outcome, key, default)


def _as_string_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _contains_abstain_signal(value: Any) -> bool:
    signals = {"abstain", "unanswerable"}
    seen: Set[int] = set()

    def visit(item: Any, key: str = "") -> bool:
        identity = id(item)
        if isinstance(item, (dict, list, tuple, set)):
            if identity in seen:
                return False
            seen.add(identity)
        if isinstance(item, dict):
            return any(visit(nested, str(nested_key)) for nested_key, nested in item.items())
        if isinstance(item, (list, tuple, set)):
            return any(visit(nested, key) for nested in item)
        if isinstance(item, str) and _normalize(key) in {"type", "status", "answer", "result", "decision"}:
            return item.strip().lower() in signals
        return False

    return visit(value)


def _is_boolean_question(question: str, outcome: Any) -> bool:
    declared_type = str(
        _outcome_value(outcome, "question_type", _outcome_value(outcome, "answer_type", ""))
    ).lower()
    if declared_type == "boolean":
        return True
    normalized = _normalize(question)
    if "bao nhieu" in normalized:
        return False
    return bool(
        re.search(r"\bco\b.+\bkhong\b", normalized)
        or re.search(r"\bco phai\b", normalized)
        or re.search(r"\b(?:dung hay sai|yes or no|true or false)\b", normalized)
    )


def _percent_scale_suspected(outcome: Any, question: str) -> bool:
    normalized = _normalize(question)
    columns = _as_string_list(_outcome_value(outcome, "columns", []))
    percent_context = bool(
        "%" in question
        or re.search(r"\b(?:phan tram|percent|ty le|ti le|tang truong|bien loi nhuan)\b", normalized)
        or any(re.search(r"(?:percent|pct|ty_le|ti_le|growth|margin)", _normalize(column).replace(" ", "_")) for column in columns)
    )
    if not percent_context:
        return False
    return any(0 < abs(number) < 1 for number in _extract_numeric_results(outcome))


def _extract_numeric_results(outcome: Any) -> List[float]:
    values: List[float] = []
    candidates = [
        _outcome_value(outcome, "rows", []),
        _outcome_value(outcome, "answer", None),
        _outcome_value(outcome, "value", None),
        _outcome_value(outcome, "result", None),
    ]

    def visit(value: Any) -> None:
        if isinstance(value, bool) or value is None:
            return
        if isinstance(value, (int, float)):
            number = float(value)
            if math.isfinite(number):
                values.append(number)
        elif isinstance(value, dict):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, (list, tuple, set)):
            for nested in value:
                visit(nested)

    for candidate in candidates:
        visit(candidate)
    return values


def _ranking_escalation_reasons(outcome: Any, question: str) -> List[str]:
    normalized = _normalize(question)
    if not _matches_any(normalized, _RANKING_PATTERNS):
        return []
    sql = str(_outcome_value(outcome, "sql", ""))
    normalized_sql = _normalize(sql).replace(" ", "_")
    reasons: List[str] = []
    order_match = re.search(r"\border\s+by\s+(.+?)(?:\blimit\b|\boffset\b|$)", sql, flags=re.IGNORECASE | re.DOTALL)
    if not order_match:
        reasons.append("ranking_missing_order_by")
    else:
        order_clause = _normalize(order_match.group(1)).replace(" ", "_")
        if not re.search(r"(?:^|_)(?:id|code|ma|sku)(?:_|$)", order_clause):
            reasons.append("ranking_missing_id_tiebreaker")
    if bool(
        _outcome_value(outcome, "boundary_tie", False)
        or _outcome_value(outcome, "tie_at_boundary", False)
        or _outcome_value(outcome, "suspected_boundary_tie", False)
    ):
        reasons.append("ranking_boundary_tie_suspected")
    else:
        top_match = re.search(r"\btop\s*-?\s*(\d+)\b", normalized)
        rows = _outcome_value(outcome, "rows", [])
        if top_match and isinstance(rows, list) and len(rows) == int(top_match.group(1)):
            columns = _as_string_list(_outcome_value(outcome, "columns", []))
            if _rows_share_boundary_metric(rows, columns):
                reasons.append("ranking_boundary_tie_suspected")
    if "order_by" not in normalized_sql and "ranking_missing_order_by" not in reasons:
        reasons.append("ranking_missing_order_by")
    return reasons


def _rows_share_boundary_metric(rows: List[Any], columns: List[str]) -> bool:
    if len(rows) < 2 or not isinstance(rows[-1], (list, tuple)) or not isinstance(rows[-2], (list, tuple)):
        return False
    metric_indexes = [
        index
        for index, column in enumerate(columns)
        if re.search(r"(?:amount|revenue|total|count|quantity|qty|score|value|sales|doanh_thu|so_luong)", _normalize(column).replace(" ", "_"))
    ]
    for index in metric_indexes:
        if index < len(rows[-1]) and index < len(rows[-2]):
            left = rows[-2][index]
            right = rows[-1][index]
            if isinstance(left, (int, float)) and not isinstance(left, bool) and left == right:
                return True
    return False


def _unique(values: Iterable[str]) -> List[str]:
    result: List[str] = []
    seen: Set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result
