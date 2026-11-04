"""A conservative, explainable heuristic for possible passage conflicts."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from schemas import RetrievedChunk

NEGATION_MARKERS = {
    "not",
    "never",
    "no",
    "cannot",
    "cant",
    "isnt",
    "arent",
    "doesnt",
    "wont",
    "prohibited",
    "without",
}
WORD_PATTERN = re.compile(r"[a-z0-9]+")
TIME_PATTERN = re.compile(
    r"\b\d{1,2}(?::\d{2})?\s*(?:a\.?m\.?|p\.?m\.?)\b", re.IGNORECASE
)
QUANTITY_PATTERN = re.compile(
    r"\b(\d+)\s*(minutes?|hours?|days?|weeks?|months?|years?)\b",
    re.IGNORECASE,
)
SUBJECT_STOP_WORDS = {
    "a",
    "am",
    "an",
    "and",
    "are",
    "at",
    "day",
    "days",
    "every",
    "for",
    "from",
    "in",
    "is",
    "it",
    "monday",
    "on",
    "open",
    "opens",
    "opening",
    "or",
    "pm",
    "saturday",
    "sunday",
    "the",
    "through",
    "to",
    "tuesday",
    "wednesday",
    "weekday",
    "weekdays",
    "when",
    "where",
    "which",
    "with",
    "thursday",
    "friday",
}
TOPIC_TERMS = {
    "opening_hours": {"hour", "hours", "open", "opens", "opening", "close", "closes"},
    "service_frequency": {
        "every",
        "minute",
        "minutes",
        "hourly",
        "frequency",
        "interval",
        "departs",
        "runs",
        "operates",
    },
    "loan_duration": {"loan", "loans", "borrow", "borrowing", "period", "duration"},
    "safety_requirement": {"must", "required", "require", "wear", "prohibited"},
}


@dataclass(frozen=True)
class Conflict:
    first: RetrievedChunk
    second: RetrievedChunk
    shared_terms: tuple[str, ...]
    reason: str


def detect_conflicts(chunks: Sequence[RetrievedChunk]) -> list[Conflict]:
    conflicts: list[Conflict] = []
    for index, first in enumerate(chunks):
        for second in chunks[index + 1 :]:
            if first.document_id == second.document_id:
                continue
            first_statements = re.split(r"(?<=[.!?])\s+|\n+", first.text)
            second_statements = re.split(r"(?<=[.!?])\s+|\n+", second.text)
            found_conflict = False
            for first_statement in first_statements:
                first_words = set(WORD_PATTERN.findall(first_statement.lower()))
                first_topics = {
                    topic
                    for topic, terms in TOPIC_TERMS.items()
                    if first_words & terms
                }
                if not first_topics:
                    continue
                first_times = {
                    re.sub(r"\s+", "", value.lower().replace(".", ""))
                    for value in TIME_PATTERN.findall(first_statement)
                }
                first_quantities = {
                    (int(value), unit.lower().rstrip("s"))
                    for value, unit in QUANTITY_PATTERN.findall(first_statement)
                }
                for second_statement in second_statements:
                    second_words = set(WORD_PATTERN.findall(second_statement.lower()))
                    subject_terms = (
                        first_words & second_words - NEGATION_MARKERS - SUBJECT_STOP_WORDS
                    )
                    if not subject_terms or not first_topics:
                        continue
                    second_topics = {
                        topic
                        for topic, terms in TOPIC_TERMS.items()
                        if second_words & terms
                    }
                    shared_topics = first_topics & second_topics
                    if not shared_topics:
                        continue
                    second_times = {
                        re.sub(r"\s+", "", value.lower().replace(".", ""))
                        for value in TIME_PATTERN.findall(second_statement)
                    }
                    second_quantities = {
                        (int(value), unit.lower().rstrip("s"))
                        for value, unit in QUANTITY_PATTERN.findall(second_statement)
                    }
                    contradiction_markers = bool(
                        (first_words & NEGATION_MARKERS)
                        != (second_words & NEGATION_MARKERS)
                    )
                    time_disagreement = bool(
                        first_times and second_times and first_times.isdisjoint(second_times)
                    )
                    quantity_disagreement = any(
                        first_value != second_value
                        for first_value, first_unit in first_quantities
                        for second_value, second_unit in second_quantities
                        if first_unit == second_unit
                    )
                    if contradiction_markers or time_disagreement or quantity_disagreement:
                        reason = (
                            "Passages about the same subject differ in negation."
                            if contradiction_markers
                            else "Passages about the same subject contain different times."
                            if time_disagreement
                            else "Passages about the same subject report different service frequencies."
                            if "service_frequency" in shared_topics
                            else "Passages about the same subject contain different quantities."
                        )
                        conflicts.append(
                            Conflict(
                                first=first,
                                second=second,
                                shared_terms=tuple(sorted(subject_terms)),
                                reason=reason,
                            )
                        )
                        found_conflict = True
                        break
                if found_conflict:
                    break
    return conflicts
