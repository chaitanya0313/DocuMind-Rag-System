import pytest
import requests

from generation import (
    GenerationError,
    INSUFFICIENT_CONTEXT_ANSWER,
    build_prompt,
    generate_answer,
)
from schemas import RetrievedChunk


def source():
    return RetrievedChunk(
        "document-id",
        "guide.txt",
        None,
        "chunk-id",
        "content-hash",
        "The library opens at eight.",
        0.2,
        "S1",
    )


def test_prompt_includes_ordered_document_context_and_question():
    prompt = build_prompt("When does it open?", [source()])
    assert prompt == "Documents:\n[Doc 1: guide.txt]\nThe library opens at eight.\n\nQuestion: When does it open?"


def test_insufficient_context_answer_is_explicit_and_stable():
    assert INSUFFICIENT_CONTEXT_ANSWER == (
        "I could not find sufficient supporting information in the uploaded documents."
    )


def test_generation_uses_mocked_local_request_without_network(monkeypatch):
    calls = {}

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "message": {"content": "The library opens at eight [Doc 1]."},
                "prompt_eval_count": 22,
                "eval_count": 8,
            }

    def fake_post(url, **kwargs):
        calls["url"] = url
        calls.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr("generation.requests.post", fake_post)
    result = generate_answer("When does it open?", [source()])
    assert result.answer.endswith("[Doc 1].")
    assert result.actual_usage == {
        "prompt_tokens": 22,
        "answer_tokens": 8,
        "total_tokens": 30,
    }
    assert calls["url"] == "http://localhost:11434/api/chat"
    assert calls["json"]["model"] == "policy-bot"
    assert calls["json"]["stream"] is False
    assert calls["json"]["options"] == {"temperature": 0, "num_ctx": 4096}
    assert calls["json"]["messages"][1]["content"] == result.prompt
    assert calls["timeout"] == 180


def test_local_model_connection_error_is_reported(monkeypatch):
    def fail_post(*args, **kwargs):
        raise requests.ConnectionError()

    monkeypatch.setattr("generation.requests.post", fail_post)
    with pytest.raises(GenerationError, match="Local model is not running"):
        generate_answer("Question?", [source()])
