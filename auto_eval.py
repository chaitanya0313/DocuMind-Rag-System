"""Automatic benchmark evaluation for retrieved, generated answers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import config
from citations import validate_citations
from generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    GenerationError,
    generate_answer,
)
from retrieval import has_sufficient_relevance, retrieve

REFUSAL_PATTERN = re.compile(
    r"(could not find|not found|do(es)? not "
    r"(say|give|indicate|state|explain|mention|contain|provide|specify)|"
    r"not (stated|available|specified|given|identified|mentioned|provided)|"
    r"cannot be (determined|established)|no information|neither mention)",
    re.IGNORECASE,
)
_TIME_PATTERN = re.compile(
    r"\b0?(\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b",
    re.IGNORECASE,
)
_NUMBER_WORDS = {
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an",
    "and", "any", "are", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "could", "did",
    "do", "does", "doing", "down", "during", "each", "few", "for", "from",
    "further", "had", "has", "have", "having", "he", "her", "here", "hers",
    "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is",
    "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no",
    "nor", "not", "of", "off", "on", "once", "only", "or", "other", "our",
    "ours", "ourselves", "out", "over", "own", "same", "she", "should", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them",
    "themselves", "then", "there", "these", "they", "this", "those", "through",
    "to", "too", "under", "until", "up", "very", "was", "we", "were", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "would", "you", "your", "yours", "yourself", "yourselves",
}
_CITATION_PATTERN = re.compile(r"\[Doc\s+(\d+)\]", re.IGNORECASE)
_NUMBER_PATTERN = re.compile(
    r"[$€£]\s*\d[\d,]*(?:\.\d+)?"
    r"|\b0?\d{1,2}:\d{2}\s*(?:a\.?m\.?|p\.?m\.?)\b"
    r"|\b0?\d{1,2}\s*(?:a\.?m\.?|p\.?m\.?)\b"
    r"|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\b\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
_BENCHMARK_TYPES = {"single", "multi_doc", "conflict", "unanswerable"}
_BENCHMARK_FIELDS = {
    "id",
    "question",
    "type",
    "expected_facts",
    "expected_sources",
    "min_sources_cited",
}


def normalize_text(text: str) -> str:
    """Normalize times, number words, punctuation, case, and whitespace."""
    normalized = text.lower()

    def normalize_time(match: re.Match[str]) -> str:
        hour = str(int(match.group(1)))
        minutes = match.group(2)
        meridiem = re.sub(r"[^a-z]", "", match.group(3)).lower()
        return f"{hour}{minutes if minutes and minutes != '00' else ''}{meridiem}"

    normalized = _TIME_PATTERN.sub(normalize_time, normalized)
    words = re.findall(r"[a-z]+|\d+(?:\.\d+)?", normalized)
    normalized = " ".join(_NUMBER_WORDS.get(word, word) for word in words)
    return " ".join(normalized.split())


def _contains_normalized(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize_text(text)} "
    normalized_phrase = f" {normalize_text(phrase)} "
    return normalized_phrase.strip() in normalized_text


def expected_facts_satisfied(answer: str, expected_facts: Sequence[str]) -> bool:
    """Each fact group is satisfied when any pipe-delimited alternative matches."""
    return all(
        any(_contains_normalized(answer, alternative) for alternative in group.split("|"))
        for group in expected_facts
    )


def is_refusal(answer: str) -> bool:
    return REFUSAL_PATTERN.search(answer) is not None


def _contains_amount_or_number(value: str, source_text: str) -> bool:
    normalized_value = normalize_text(value)
    normalized_source = normalize_text(source_text)
    match = re.match(r"([$€£])\s*(.*)", value)
    if match:
        currency_names = {"$": "dollar", "€": "euro", "£": "pound"}
        amount = normalize_text(match.group(2))
        normalized_source = re.sub(
            r"\b(?:dollars?|euros?|pounds?)\b",
            "",
            normalized_source,
        )
        if match.group(1) not in source_text and currency_names[match.group(1)] not in source_text.lower():
            return False
        return _contains_normalized(normalized_source, amount)
    return _contains_normalized(normalized_source, normalized_value)


def _sentence_is_supported(sentence: str, cited_passages: Sequence[object]) -> bool:
    cited_text = "\n".join(str(getattr(passage, "text", "")) for passage in cited_passages)
    if not cited_passages:
        return False
    sentence_text = _CITATION_PATTERN.sub("", sentence)
    if any(
        not _contains_amount_or_number(number, cited_text)
        for number in _NUMBER_PATTERN.findall(sentence_text)
    ):
        return False
    answer_words = {
        word
        for word in normalize_text(sentence_text).split()
        if len(word) > 2 and word not in _STOP_WORDS
    }
    source_words = {
        word
        for word in normalize_text(cited_text).split()
        if len(word) > 2 and word not in _STOP_WORDS
    }
    return not answer_words or len(answer_words & source_words) / len(answer_words) >= 0.6


def citation_support(
    answer: str, passages: Sequence[object]
) -> tuple[float | None, bool, list[dict[str, object]]]:
    """Return cited-sentence support, validity, and per-sentence support details."""
    citation_matches = list(_CITATION_PATTERN.finditer(answer))
    citation_valid = True
    citations_by_sentence: dict[int, list[object]] = {}
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer)

    for match in citation_matches:
        index = int(match.group(1)) - 1
        if index < 0 or index >= len(passages):
            citation_valid = False
            continue
        prefix = answer[: match.start()]
        sentence_index = len(re.split(r"(?<=[.!?])\s+|\n+", prefix)) - 1
        citations_by_sentence.setdefault(sentence_index, []).append(passages[index])

    details = []
    for sentence_index, cited_passages in citations_by_sentence.items():
        sentence = sentences[sentence_index]
        supported = _sentence_is_supported(sentence, cited_passages)
        details.append(
            {
                "sentence": sentence.strip(),
                "supported": supported,
                "source_files": sorted(
                    {str(getattr(passage, "filename", "")) for passage in cited_passages}
                ),
            }
        )
    score = (
        sum(item["supported"] is True for item in details) / len(details)
        if details
        else None
    )
    return score, citation_valid, details


def conflict_handled(
    answer: str,
    passages: Sequence[object],
    expected_facts: Sequence[str],
    min_sources_cited: int,
) -> bool:
    cited_files: set[str] = set()
    for match in _CITATION_PATTERN.finditer(answer):
        index = int(match.group(1)) - 1
        if 0 <= index < len(passages):
            cited_files.add(str(getattr(passages[index], "filename", "")))
    return (
        len(cited_files) >= min_sources_cited
        and expected_facts_satisfied(answer, expected_facts)
    )


def validate_benchmark(questions: object) -> list[dict[str, object]]:
    if not isinstance(questions, list):
        raise ValueError("Benchmark must be a JSON list.")
    validated = []
    seen_ids: set[str] = set()
    for index, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(f"Question {index + 1} must be an object.")
        if set(question) != _BENCHMARK_FIELDS:
            raise ValueError(f"Question {index + 1} has invalid fields.")
        if not all(
            isinstance(question[key], str) and question[key].strip()
            for key in ("id", "question")
        ):
            raise ValueError(f"Question {index + 1} needs a non-empty id and question.")
        if question["id"] in seen_ids:
            raise ValueError(f"Duplicate question id: {question['id']}.")
        seen_ids.add(question["id"])
        if question["type"] not in _BENCHMARK_TYPES:
            raise ValueError(f"Question {question['id']} has an invalid type.")
        if not isinstance(question["expected_facts"], list) or not all(
            isinstance(fact, str) and fact.strip()
            for fact in question["expected_facts"]
        ):
            raise ValueError(f"Question {question['id']} has invalid expected facts.")
        if not isinstance(question["expected_sources"], list) or not all(
            isinstance(source, str) and source.strip()
            for source in question["expected_sources"]
        ):
            raise ValueError(f"Question {question['id']} has invalid expected sources.")
        minimum_sources = question["min_sources_cited"]
        if (
            not isinstance(minimum_sources, int)
            or isinstance(minimum_sources, bool)
            or minimum_sources < 0
        ):
            raise ValueError(f"Question {question['id']} has invalid minimum source count.")
        if question["type"] == "unanswerable" and (
            question["expected_facts"] or question["expected_sources"]
        ):
            raise ValueError(
                f"Unanswerable question {question['id']} cannot require facts or sources."
            )
        validated.append(dict(question))
    return validated


def load_benchmark(path: str | Path) -> list[dict[str, object]]:
    with Path(path).open(encoding="utf-8") as benchmark_file:
        return validate_benchmark(json.load(benchmark_file))


def save_benchmark(path: str | Path, questions: object) -> list[dict[str, object]]:
    validated = validate_benchmark(questions)
    benchmark_path = Path(path)
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_path.write_text(
        json.dumps(validated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return validated


def evaluate_question(
    question: Mapping[str, object],
    passages: Sequence[object],
    answer: str,
    *,
    retrieval_rank: int | None,
) -> dict[str, object]:
    expected_sources = question["expected_sources"]
    expected_facts = question["expected_facts"]
    unanswerable = question["type"] == "unanswerable"
    refusal = is_refusal(answer)
    cited_sentences = _CITATION_PATTERN.findall(answer)
    support_score, citation_ids_valid, sentence_details = citation_support(
        answer, passages
    )
    citation_valid = citation_ids_valid and (
        unanswerable or refusal or bool(cited_sentences)
    )
    correct_refusal = unanswerable and refusal and not cited_sentences
    facts_correct = (
        None
        if unanswerable
        else expected_facts_satisfied(answer, expected_facts)
    )
    citations_supported = (
        all(item["supported"] is True for item in sentence_details)
        if sentence_details
        else None
    )
    grounded = bool(
        correct_refusal
        or (
            citation_valid
            and bool(sentence_details)
            and citations_supported
        )
    )
    expected_source_set = set(expected_sources)
    retrieved_source_set = {
        str(getattr(passage, "filename", "")) for passage in passages
    }
    cited_source_set = {
        str(getattr(passages[int(citation) - 1], "filename", ""))
        for citation in cited_sentences
        if 1 <= int(citation) <= len(passages)
    }
    return {
        "id": question["id"],
        "question": question["question"],
        "type": question["type"],
        "answer": answer,
        "retrieved_files": [
            str(getattr(passage, "filename", "")) for passage in passages
        ],
        "retrieval_hit": (
            None if unanswerable else bool(expected_source_set & retrieved_source_set)
        ),
        "reciprocal_rank": (
            None
            if unanswerable or retrieval_rank is None
            else 1 / retrieval_rank
        ),
        "answer_correct": facts_correct,
        "citation_valid": citation_valid,
        "citation_supported": support_score,
        "supported_sentences": sentence_details,
        "grounded": grounded,
        "refusal": correct_refusal if unanswerable else not refusal,
        "is_refusal": refusal,
        "conflict_handled": (
            conflict_handled(
                answer,
                passages,
                expected_facts,
                int(question["min_sources_cited"]),
            )
            if question["type"] == "conflict"
            else None
        ),
    }


def run_evaluation(
    questions: Sequence[Mapping[str, object]],
    *,
    store: object,
    embedder: object,
    top_k: int,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> dict[str, object]:
    benchmark = validate_benchmark([dict(question) for question in questions])
    indexed_files = {
        str(chunk["filename"]) for chunk in store.all_chunks()
    }
    results = []
    skipped = []
    for index, question in enumerate(benchmark, start=1):
        if progress_callback is not None:
            progress_callback(index, len(benchmark), str(question["question"]))
        missing_sources = sorted(set(question["expected_sources"]) - indexed_files)
        if missing_sources:
            skipped.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "missing_sources": missing_sources,
                }
            )
            continue
        try:
            passages, _ = retrieve(
                str(question["question"]),
                store=store,
                embedder=embedder,
                top_k=top_k,
            )
            answer = INSUFFICIENT_CONTEXT_ANSWER
            if passages and has_sufficient_relevance(
                passages, max_distance=config.MIN_RELEVANCE_DISTANCE
            ):
                generated = generate_answer(str(question["question"]), passages)
                valid_ids = {str(getattr(passage, "citation_id")) for passage in passages}
                answer, _, _ = validate_citations(generated.answer, valid_ids)
        except (GenerationError, OSError, RuntimeError, ValueError) as exc:
            passages = []
            answer = ""
            error = str(exc)
        else:
            error = None
        expected_sources = set(question["expected_sources"])
        retrieval_rank = next(
            (
                rank
                for rank, passage in enumerate(passages, start=1)
                if str(getattr(passage, "filename", "")) in expected_sources
            ),
            None,
        )
        result = evaluate_question(
            question,
            passages,
            answer,
            retrieval_rank=retrieval_rank,
        )
        result["error"] = error
        results.append(result)
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
        "skipped": skipped,
    }


def save_results(path: str | Path, results: Mapping[str, object]) -> None:
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _rate(values: Sequence[bool | None]) -> str:
    applicable = [value for value in values if value is not None]
    if not applicable:
        return "n/a"
    return f"{sum(applicable) / len(applicable):.0%}"


def render_evaluation(
    *,
    store: object,
    embedder: object,
    benchmark_path: str | Path,
    results_path: str | Path,
    advanced_renderer: Callable[[], None],
) -> None:
    import pandas as pd
    import streamlit as st

    st.title("Evaluation")
    st.write(
        "Runs benchmark questions through retrieval, answer generation, and citation "
        "validation. Citation support is an automatic estimate."
    )
    try:
        questions = load_benchmark(benchmark_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load benchmark questions: {exc}")
        return

    result = None
    try:
        with Path(results_path).open(encoding="utf-8") as result_file:
            result = json.load(result_file)
    except FileNotFoundError:
        pass
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not load saved evaluation results: {exc}")

    if st.button("Run evaluation", type="primary"):
        progress_bar = st.progress(0.0)
        current_question = st.empty()

        def update_progress(index: int, total: int, question_text: str) -> None:
            current_question.caption(f"Question {index} of {total}: {question_text}")
            progress_bar.progress(index / total if total else 1.0)

        try:
            result = run_evaluation(
                questions,
                store=store,
                embedder=embedder,
                top_k=config.DEFAULT_TOP_K,
                progress_callback=update_progress,
            )
            save_results(results_path, result)
        except (OSError, RuntimeError, ValueError) as exc:
            st.error(f"Evaluation could not be completed: {exc}")
            return

    with st.expander("Edit questions"):
        editor_rows = [
            {
                "id": question["id"],
                "question": question["question"],
                "type": question["type"],
                "expected_facts": ", ".join(question["expected_facts"]),
                "expected_sources": ", ".join(question["expected_sources"]),
                "min_sources_cited": question["min_sources_cited"],
            }
            for question in questions
        ]
        edited_rows = st.data_editor(
            editor_rows,
            use_container_width=True,
            hide_index=True,
            key="automatic_benchmark_editor",
            column_config={
            "id": None,
            "min_sources_cited": None,
            },
        )
        if st.button("Save questions"):
            try:
                edited_questions = []
                for row in edited_rows.to_dict(orient="records"):
                    edited_questions.append(
                        {
                            **row,
                            "expected_facts": [
                                item.strip()
                                for item in str(row["expected_facts"]).split(",")
                                if item.strip()
                            ],
                            "expected_sources": [
                                item.strip()
                                for item in str(row["expected_sources"]).split(",")
                                if item.strip()
                            ],
                        }
                    )
                questions = save_benchmark(benchmark_path, edited_questions)
            except (OSError, TypeError, ValueError, KeyError) as exc:
                st.error(f"Could not save benchmark questions: {exc}")
            else:
                st.success("Benchmark questions saved.")
                st.rerun()

    with st.expander("Advanced"):
        advanced_renderer()

    if result is None:
        st.info("Run evaluation to create a saved results report.")
        return
    for skipped_question in result.get("skipped", []):
        missing = ", ".join(skipped_question["missing_sources"])
        st.warning(
            f"Skipped {skipped_question['id']}: expected source files not indexed: {missing}"
        )
    results = result.get("results", [])
    if not results:
        st.info("No benchmark questions have results yet.")
        return

    cards = (
        ("Answer accuracy", "answer_correct", lambda row: row["type"] != "unanswerable"),
        ("Retrieval hit rate", "retrieval_hit", lambda row: row["type"] != "unanswerable"),
        (
            "Citation support",
            "citation_supported",
            lambda row: row.get("citation_supported") is not None,
        ),
        ("Grounded answers", "grounded", lambda row: True),
        ("Correct refusals", "refusal", lambda row: row["type"] == "unanswerable"),
        ("Conflicts handled", "conflict_handled", lambda row: row["type"] == "conflict"),
    )
    columns = st.columns(3)
    for index, (label, key, applies) in enumerate(cards):
        values = [
            row.get(key)
            for row in results
            if applies(row) and row.get(key) is not None
        ]
        columns[index % len(columns)].metric(label, _rate(values))
        if key == "grounded":
            grounded_values = [row["grounded"] for row in results]
            hallucination_rate = (
                f"{1 - sum(grounded_values) / len(grounded_values):.0%}"
                if grounded_values
                else "n/a"
            )
            columns[index % len(columns)].caption(
                f"Hallucination rate: {hallucination_rate}"
            )

    check_keys = (
        ("Retrieval", "retrieval_hit"),
        ("Answer", "answer_correct"),
        ("Citation valid", "citation_valid"),
        ("Citation supported", "citation_supported"),
        ("Grounded", "grounded"),
        ("Refusal", "refusal"),
        ("Conflict handled", "conflict_handled"),
    )
    table_rows = []
    for row in results:
        display_row: dict[str, object] = {
            "Question": row["question"],
            "Type": row["type"],
        }
        for label, key in check_keys:
            value = row.get(key)
            if key == "citation_supported" and isinstance(value, (int, float)):
                display_row[label] = "Pass" if value == 1 else "Fail"
            else:
                display_row[label] = (
                    "n/a"
                    if value is None
                    else "Pass"
                    if value is True
                    else "Fail"
                )
        table_rows.append(display_row)
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    csv_rows = []
    for row in results:
        csv_row = {
            "Question": row["question"],
            "Type": row["type"],
        }
        for label, key in check_keys:
            csv_row[label] = row.get(key)
        csv_rows.append(csv_row)
    csv_data = pd.DataFrame(csv_rows).to_csv(index=False)
    st.download_button(
        "Download CSV",
        csv_data,
        file_name="documind-automatic-evaluation.csv",
        mime="text/csv",
    )
    for row in results:
        with st.expander(str(row["question"])):
            st.write(row.get("answer") or "No answer was produced.")
            if row.get("error"):
                st.error(row["error"])
            failed_checks = [
                label
                for label, key in check_keys
                if row.get(key) is False
                or (
                    key == "citation_supported"
                    and isinstance(row.get(key), (int, float))
                    and row[key] < 1
                )
            ]
            if failed_checks:
                st.caption("Failed checks: " + ", ".join(failed_checks))
            else:
                st.caption("No failed checks.")
            st.caption(
                "Retrieved files: "
                + (", ".join(row.get("retrieved_files", [])) or "none")
            )
