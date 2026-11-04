from types import SimpleNamespace

from auto_eval import (
    citation_support,
    conflict_handled,
    expected_facts_satisfied,
    is_refusal,
    normalize_text,
)


def passage(filename, text, citation_id="S1"):
    return SimpleNamespace(filename=filename, text=text, citation_id=citation_id)


def test_normalization_equates_time_and_number_words():
    assert normalize_text("9:00 AM") == normalize_text("9am")
    assert normalize_text("five books") == normalize_text("5 books")


def test_expected_fact_alternatives_accept_any_normalized_match():
    assert expected_facts_satisfied(
        "The library opens at 9am and students may borrow five books.",
        ["9:00 AM|9 AM", "five|5"],
    )
    assert not expected_facts_satisfied("The library opens at 10 AM.", ["9:00 AM|9 AM"])


def test_refusal_matcher_handles_negative_phrases():
    assert is_refusal("The documents do not mention the late-return fine.")
    assert is_refusal("The fine cannot be determined.")
    assert not is_refusal("The fine is ten dollars.")


def test_citation_support_requires_numbers_in_cited_passages():
    source = passage("handbook.txt", "The library opens at 9 AM on weekdays.")
    score, valid, details = citation_support(
        "The library opens at 9:00 AM [Doc 1].",
        [source],
    )
    assert score == 1.0
    assert valid
    assert details[0]["supported"]

    score, valid, details = citation_support(
        "The library opens at 10 AM [Doc 1].",
        [source],
    )
    assert score == 0.0
    assert valid
    assert not details[0]["supported"]


def test_conflict_handling_requires_all_facts_and_distinct_sources():
    passages = [
        passage("handbook.txt", "Library opens at 9 AM and closes at 7 PM."),
        passage("services.txt", "Library opens at 8 AM and closes at 8 PM.", "S2"),
    ]
    answer = (
        "The handbook lists 9 AM to 7 PM [Doc 1]. "
        "Services list 8 AM to 8 PM [Doc 2]."
    )
    facts = ["9:00 AM|9 AM", "7:00 PM|7 PM", "8:00 AM|8 AM", "8:00 PM|8 PM"]
    assert conflict_handled(answer, passages, facts, min_sources_cited=2)
    assert not conflict_handled(answer, passages, facts, min_sources_cited=3)
    assert not conflict_handled(
        answer.replace("8 PM", "6 PM", 1),
        passages,
        facts,
        min_sources_cited=2,
    )
