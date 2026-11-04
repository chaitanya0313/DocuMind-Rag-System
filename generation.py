"""Grounded answer generation through a local Ollama model."""

from __future__ import annotations

from dataclasses import dataclass

import requests

from config import MODEL_NAME, OLLAMA_URL
from schemas import RetrievedChunk

INSUFFICIENT_CONTEXT_ANSWER = (
    "I could not find sufficient supporting information in the uploaded documents."
)


class GenerationError(RuntimeError):
    pass


@dataclass(frozen=True)
class GenerationResult:
    answer: str
    model: str
    prompt: str
    actual_usage: dict[str, int] | None


SYSTEM_INSTRUCTION = (
    "You answer only from the provided documents. Cite every claim inline as [Doc N]. "
    "If documents disagree, state both values with their citations and do not pick one "
    "unless a document states which takes precedence. If the answer is not in the "
    "documents, say so and do not guess."
)


def build_prompt(question: str, passages: list[RetrievedChunk]) -> str:
    chunks = [
        {
            "title": (
                f"{source.filename} p.{source.page_number}"
                if source.page_number is not None
                else source.filename
            ).replace("]", ""),
            "text": source.text,
        }
        for source in passages
    ]
    context = "\n\n".join(
        f"[Doc {index}: {chunk['title']}]\n{chunk['text']}"
        for index, chunk in enumerate(chunks, 1)
    )
    return f"Documents:\n{context}\n\nQuestion: {question}"


def generate_answer(
    question: str,
    passages: list[RetrievedChunk],
    *,
    model_name: str | None = None,
) -> GenerationResult:
    selected_model = model_name or MODEL_NAME
    prompt = build_prompt(question, passages)
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": selected_model,
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 4096},
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=180,
        )
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise GenerationError(
            "Local model is not running. Start Ollama and try again."
        ) from exc
    except requests.Timeout as exc:
        raise GenerationError("Local model request timed out. Please try again.") from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else "unknown"
        raise GenerationError(
            f"Local model request failed (HTTP status {status_code})."
        ) from exc
    except requests.RequestException as exc:
        raise GenerationError("Could not connect to the local model. Try again.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise GenerationError("Local model returned an invalid response.") from exc

    message = payload.get("message") if isinstance(payload, dict) else None
    answer = message.get("content") if isinstance(message, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise GenerationError("Local model returned no text. Try again.")

    prompt_tokens = payload.get("prompt_eval_count")
    answer_tokens = payload.get("eval_count")
    actual_usage = None
    if isinstance(prompt_tokens, int) or isinstance(answer_tokens, int):
        actual_usage = {}
        if isinstance(prompt_tokens, int):
            actual_usage["prompt_tokens"] = prompt_tokens
        if isinstance(answer_tokens, int):
            actual_usage["answer_tokens"] = answer_tokens
        if isinstance(prompt_tokens, int) and isinstance(answer_tokens, int):
            actual_usage["total_tokens"] = prompt_tokens + answer_tokens
    return GenerationResult(
        answer=answer.strip(),
        model=selected_model,
        prompt=prompt,
        actual_usage=actual_usage,
    )
