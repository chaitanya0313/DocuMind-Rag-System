"""Citation reference validation and transparent human support scoring."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

CITATION_PATTERN = re.compile(r"\[(S\d+|Doc\s+\d+)\]")
DOCUMENT_CITATION_PATTERN = re.compile(r"\[Doc\s+(\d+)\]")
REFUSAL_PATTERN = re.compile(
    r"(could not find|not found|do(es)? not "
    r"(say|give|indicate|state|explain|mention|contain|provide|specify)|"
    r"not (stated|available|specified|given|identified|mentioned|provided)|"
    r"cannot be (determined|established)|no information|neither mention)",
    re.IGNORECASE,
)


def is_refusal_answer(answer: str) -> bool:
    return REFUSAL_PATTERN.search(answer) is not None


def _citation_id(reference: str) -> str:
    document_number = re.fullmatch(r"Doc\s+(\d+)", reference)
    return f"S{document_number.group(1)}" if document_number else reference


def validate_citations(
    answer: str, valid_citation_ids: set[str]
) -> tuple[str, list[str], list[str]]:
    references = CITATION_PATTERN.findall(answer)
    valid = [
        citation_id
        for citation_id in map(_citation_id, references)
        if citation_id in valid_citation_ids
    ]
    invalid = [
        reference
        for reference in references
        if _citation_id(reference) not in valid_citation_ids
    ]

    def preserve_valid_or_document_reference(match: re.Match[str]) -> str:
        reference = match.group(1)
        if reference.startswith("Doc ") or _citation_id(reference) in valid_citation_ids:
            return match.group(0)
        return ""

    cleaned = CITATION_PATTERN.sub(preserve_valid_or_document_reference, answer)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
    cleaned = re.sub(r" +([,.!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned, valid, invalid


def format_document_citations(
    answer: str, sources: Sequence[Mapping[str, object]]
) -> tuple[str, list[int]]:
    missing_sources: list[int] = []

    def replace_reference(match: re.Match[str]) -> str:
        source_number = int(match.group(1))
        if source_number < 1 or source_number > len(sources):
            missing_sources.append(source_number)
            return f"[Doc {source_number}: source not found]"

        source = sources[source_number - 1]
        citation_id = str(source["citation_id"])
        filename = str(source["filename"])
        page_number = source.get("page_number")
        page = f" p.{page_number}" if page_number is not None else ""
        label = f"{citation_id}: {filename}{page}"
        for character in r"\`*_{}[]()#+-.!|>~":
            label = label.replace(character, f"\\{character}")
        return f"[{label}](#source-{citation_id})"

    return DOCUMENT_CITATION_PATTERN.sub(replace_reference, answer), missing_sources


def extract_answer_claims(answer: str) -> list[dict[str, object]]:
    """Split answer text into simple, human-reviewable sentence units."""
    claims = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        text = re.sub(r"^\s*(?:[-*#]+\s*)", "", sentence).strip()
        if text:
            claims.append(
                {
                    "text": text,
                    "citation_ids": [
                        _citation_id(reference) for reference in CITATION_PATTERN.findall(text)
                    ],
                }
            )
    return claims


def evaluate_citation_support(
    claims: Sequence[dict[str, object]],
    human_assessments: Mapping[str, bool | None],
) -> dict[str, float | int | None]:
    evaluated = [
        human_assessments[str(index)]
        for index in range(len(claims))
        if str(index) in human_assessments
        and human_assessments[str(index)] is not None
    ]
    supported = sum(assessment is True for assessment in evaluated)
    unsupported = sum(assessment is False for assessment in evaluated)
    return {
        "evaluated_claims": len(evaluated),
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "score": supported / len(evaluated) if evaluated else None,
    }
