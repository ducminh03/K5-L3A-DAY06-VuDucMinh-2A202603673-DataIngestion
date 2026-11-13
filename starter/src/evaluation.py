"""Evaluate retrieval evidence and summarize Hit@1 and Evidence Hit@3."""
import re
import unicodedata
from typing import Any


def _normalize_evidence(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = "".join(
        " " if unicodedata.category(character).startswith("P") else character
        for character in normalized
    )
    return " ".join(normalized.split())


def _contains_evidence(quote: str, text: str) -> bool:
    quote = quote.strip(" \t\r\n\"'“”")
    fragments = [
        _normalize_evidence(fragment)
        for fragment in re.split(r"(?:…|\.{3})", quote)
        if _normalize_evidence(fragment)
    ]
    normalized_text = _normalize_evidence(text)
    position = 0
    for fragment in fragments:
        found_at = normalized_text.find(fragment, position)
        if found_at < 0:
            return False
        position = found_at + len(fragment)
    return bool(fragments)


def evaluate_query(
    question: dict[str, Any], hits: list[dict[str, Any]]
) -> dict[str, Any]:
    """Keep the top-three trace and mark hits matching source, page, and evidence."""
    required_question_fields = {
        "id",
        "split",
        "question",
        "gold_source_id",
        "gold_page",
        "evidence_quote",
    }
    missing_fields = required_question_fields - question.keys()
    if missing_fields:
        raise ValueError(f"question is missing fields: {', '.join(sorted(missing_fields))}")
    if question["split"] not in {"dev", "test"}:
        raise ValueError(f"invalid question split: {question['split']!r}")

    try:
        gold_page = int(question["gold_page"])
    except (TypeError, ValueError) as error:
        raise ValueError("gold_page must be an integer") from error

    trace_hits: list[dict[str, Any]] = []
    for rank, hit in enumerate(hits[:3], start=1):
        required_hit_fields = {"document_id", "page", "text"}
        missing_hit_fields = required_hit_fields - hit.keys()
        if missing_hit_fields:
            raise ValueError(
                f"hit {rank} is missing fields: {', '.join(sorted(missing_hit_fields))}"
            )
        try:
            hit_page = int(hit["page"])
        except (TypeError, ValueError) as error:
            raise ValueError(f"hit {rank} page must be an integer") from error
        if not isinstance(hit["text"], str):
            raise ValueError(f"hit {rank} text must be a string")

        evidence_hit = (
            hit["document_id"] == question["gold_source_id"]
            and hit_page == gold_page
            and _contains_evidence(question["evidence_quote"], hit["text"])
        )
        trace_hit = dict(hit)
        trace_hit["evidence_hit"] = evidence_hit
        trace_hits.append(trace_hit)

    return {
        "id": question["id"],
        "split": question["split"],
        "question": question["question"],
        "gold_source_id": question["gold_source_id"],
        "gold_page": gold_page,
        "evidence_quote": question["evidence_quote"],
        "hits": trace_hits,
        "hit1": bool(trace_hits and trace_hits[0]["evidence_hit"]),
        "hit3": any(hit["evidence_hit"] for hit in trace_hits),
    }


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate Hit@1 and Evidence Hit@3 by strategy and dev/test split."""
    summary: dict[str, Any] = {}
    for trace in traces:
        strategy = trace.get("strategy")
        split = trace.get("split")
        if not isinstance(strategy, str) or not strategy.strip():
            raise ValueError("each trace must have a non-empty strategy")
        if split not in {"dev", "test"}:
            raise ValueError(f"invalid trace split: {split!r}")

        strategy_summary = summary.setdefault(strategy, {})
        split_summary = strategy_summary.setdefault(
            split, {"total": 0, "hit1": 0, "hit3": 0}
        )
        split_summary["total"] += 1
        split_summary["hit1"] += int(bool(trace.get("hit1")))
        split_summary["hit3"] += int(bool(trace.get("hit3")))
    return summary
