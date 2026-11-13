"""Facts Pack builder — the ONLY thing the LLM ever receives (BLUEPRINT §4).

It contains computed numbers from steps A-F, never raw rows in bulk.
Emails, phone-like strings and long IDs in example values are masked here,
so a summary call can never leak them.
"""
from __future__ import annotations

import re

_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s]+")
_PHONE = re.compile(r"\b\d{5}[\s-]?\d{5}\b")


def mask(text: str) -> str:
    text = _EMAIL.sub("[email]", str(text))
    return _PHONE.sub("[phone]", text)


def summarise_relations(relations: dict) -> dict:
    """Counts-only view of the dedupe result for the Facts Pack.

    Group members hold raw cell values (names, addresses), which must never
    reach the LLM, so only counts and confidence figures pass through.
    """
    groups = relations.get("groups", [])
    return {
        "skipped": relations.get("skipped", True),
        "reason": relations.get("reason"),
        "duplicate_groups": relations.get("duplicate_groups", 0),
        "duplicate_rows": relations.get("duplicate_rows", 0),
        "exact_groups": sum(g["kind"] == "exact" for g in groups),
        "near_groups": sum(g["kind"] == "near" for g in groups),
        "low_confidence_groups": sum(g["confidence"] < 0.9 for g in groups),
        "candidate_pairs": relations.get("stats", {}).get("candidate_pairs", 0),
    }


def build_facts_pack(profile: dict, score: int, reasons: list[str],
                     anomalies: dict, drift: dict, relations: dict) -> dict:
    return {
        "rows": profile["rows"],
        "cols": profile["cols"],
        "health_score": score,
        "score_reasons": reasons,
        "missing_pct": profile["missing_pct"],
        "duplicate_rows": profile["duplicate_rows"],
        "columns": [
            {k: (mask(v) if isinstance(v, str) else v) for k, v in c.items()
             if k in ("name", "type", "missing_pct", "unique", "min", "max", "mean")}
            for c in profile["columns"]
        ],
        "anomalies": anomalies,
        "drift": drift,
        "relations": summarise_relations(relations),
    }
