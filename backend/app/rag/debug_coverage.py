"""Vector-store coverage and eligibility diagnostic utilities."""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from app.rag.vector_store import get_collection_name, get_vector_store

logger = logging.getLogger("rag.debug_coverage")


def _load_collection(audience: str) -> tuple[list[dict[str, Any]], list[str]]:
    store = get_vector_store(audience)
    raw = store._collection.get(include=["metadatas"])
    metadatas: list[dict[str, Any]] = raw.get("metadatas") or []
    ids: list[str] = raw.get("ids") or []
    return metadatas, ids


def _priority_bucket(value: Any) -> str:
    try:
        priority = int(value or 0)
    except Exception:
        priority = 0
    if priority >= 80:
        return "80-100"
    if priority >= 50:
        return "50-79"
    if priority >= 20:
        return "20-49"
    return "0-19"


def _match_filter(metadata: dict[str, Any], filter_dict: dict[str, Any] | None) -> bool:
    if not filter_dict:
        return True
    if "$and" in filter_dict:
        return all(_match_filter(metadata, item) for item in filter_dict.get("$and") or [])
    for key, expected in filter_dict.items():
        value = metadata.get(key)
        if isinstance(expected, str):
            if str(value or "").strip().lower() != expected.strip().lower():
                return False
        else:
            if value != expected:
                return False
    return True


def _summary_from_metadatas(
    audience: str,
    metadatas: list[dict[str, Any]],
    ids: list[str],
) -> dict[str, Any]:
    total = len(metadatas)
    active_count = 0
    inactive_count = 0
    by_category: Counter[str] = Counter()
    active_by_category: Counter[str] = Counter()
    by_document: Counter[str] = Counter()
    by_official: Counter[str] = Counter()
    active_by_official: Counter[str] = Counter()
    by_priority: Counter[str] = Counter()
    active_by_priority: Counter[str] = Counter()

    for md in metadatas:
        if not isinstance(md, dict):
            continue
        status = str(md.get("status") or "").strip().lower()
        category = str(md.get("category") or "unknown")
        title = str(md.get("document_title") or md.get("title") or "Untitled")
        official = "official" if bool(md.get("is_official")) else "non_official"
        priority_bucket = _priority_bucket(md.get("source_priority"))

        by_category[category] += 1
        by_document[title] += 1
        by_official[official] += 1
        by_priority[priority_bucket] += 1

        if status == "active":
            active_count += 1
            active_by_category[category] += 1
            active_by_official[official] += 1
            active_by_priority[priority_bucket] += 1
        else:
            inactive_count += 1

    return {
        "audience": audience,
        "collection_name": get_collection_name(audience),
        "total_chunks": total,
        "active_chunks": active_count,
        "inactive_chunks": inactive_count,
        "by_category": dict(by_category.most_common()),
        "active_by_category": dict(active_by_category.most_common()),
        "by_document": dict(by_document.most_common(50)),
        "by_official": dict(by_official.most_common()),
        "active_by_official": dict(active_by_official.most_common()),
        "by_priority_bucket": dict(by_priority.most_common()),
        "active_by_priority_bucket": dict(active_by_priority.most_common()),
        "sample_chunk_ids": ids[:10],
    }


def coverage_report(audience: str) -> dict[str, Any]:
    try:
        metadatas, ids = _load_collection(audience)
    except Exception as exc:
        logger.warning("coverage_report: failed to read collection for %s: %s", audience, exc)
        return {"error": str(exc), "audience": audience}

    report = _summary_from_metadatas(audience, metadatas, ids)
    logger.info(
        "rag.coverage audience=%s total=%s active=%s inactive=%s categories=%s",
        audience,
        report["total_chunks"],
        report["active_chunks"],
        report["inactive_chunks"],
        report["by_category"],
    )
    return report


def eligibility_report(audience: str, applied_filters: dict[str, Any] | None) -> dict[str, Any]:
    try:
        metadatas, _ids = _load_collection(audience)
    except Exception as exc:
        logger.warning("eligibility_report: failed to read collection for %s: %s", audience, exc)
        return {"error": str(exc), "audience": audience}

    eligible = [md for md in metadatas if isinstance(md, dict) and _match_filter(md, applied_filters)]
    by_category: Counter[str] = Counter(str(md.get("category") or "unknown") for md in eligible)
    by_document: Counter[str] = Counter(
        str(md.get("document_title") or md.get("title") or "Untitled")
        for md in eligible
    )
    by_official: Counter[str] = Counter("official" if bool(md.get("is_official")) else "non_official" for md in eligible)
    return {
        "audience": audience,
        "applied_filters": applied_filters or {},
        "eligible_chunks": len(eligible),
        "eligible_by_category": dict(by_category.most_common()),
        "eligible_by_document": dict(by_document.most_common(25)),
        "eligible_by_official": dict(by_official.most_common()),
    }
