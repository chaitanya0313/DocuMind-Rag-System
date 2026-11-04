"""DocuMind AI: multi-document retrieval with grounded, inspectable answers."""

from __future__ import annotations

import csv
import io
import json
import statistics
import time
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

import streamlit as st

import auto_eval
import config
from config import GENERATION_TEMPERATURE
from citations import (
    evaluate_citation_support,
    extract_answer_claims,
    format_document_citations,
    validate_citations,
)
from conflict_detection import detect_conflicts
from embeddings import SentenceTransformerEmbedder
from evaluation import evaluate_retrieval_dataset, load_dataset, save_dataset
from generation import (
    INSUFFICIENT_CONTEXT_ANSWER,
    GenerationError,
    SYSTEM_INSTRUCTION,
    build_prompt,
    generate_answer,
)
from ingestion import ingest_documents
from metrics import (
    MetricsStore,
    average_measurement,
    estimate_tokens,
    new_query_record,
    query_summary,
)
from retrieval import has_sufficient_relevance, retrieve
from vector_store import ChromaVectorStore

st.set_page_config(
    page_title="DocuMind AI",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
      :root { --primary-color: #1769aa; }
      .block-container { max-width: 1200px; padding-top: 2rem; }
      [data-testid="stMetric"] { border: 1px solid #d9e5f2; border-radius: 12px; padding: 12px; }
      .documind-muted { color: #52657a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_store() -> ChromaVectorStore:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    return ChromaVectorStore(config.CHROMA_DIR)


@st.cache_resource
def get_embedder() -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder()


@st.cache_resource
def get_metrics_store() -> MetricsStore:
    return MetricsStore(config.METRICS_DB)


def show_sources(sources: list[dict]) -> None:
    if not sources:
        st.info("No source passages are available for this answer.")
        return
    st.subheader("Sources")
    for source in sources:
        st.markdown(
            f'<div id="source-{source["citation_id"]}"></div>',
            unsafe_allow_html=True,
        )
        page = (
            f" · page {source['page_number']}"
            if source.get("page_number") is not None
            else ""
        )
        with st.expander(
            f"[{source['citation_id']}] {source['filename']}{page} · "
            f"chunk {source['chunk_id']} · distance {source['distance']:.4f}"
        ):
            st.write(source["text"])


def render_home(store: ChromaVectorStore) -> None:
    st.title("DocuMind AI")
    st.subheader("Multi-document RAG with citations and measurable evaluation")
    st.write(
        "Upload your own PDF, Word, or text documents, retrieve relevant passages, "
        "and ask the local model for a concise answer grounded in those passages."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("Active documents", store.document_count())
    col2.metric("Indexed chunks", store.chunk_count())
    col3.metric("Answer model", config.MODEL_NAME)
    st.info(
        "Suggested question (not a generated result): “Which details differ between "
        "the uploaded documents?”"
    )
    st.markdown(
        """
        **Quick start:** Open **Documents**, upload files, then open **Ask questions**.
        Retrieval, context inspection, and answer generation use local services. Start
        Ollama and make sure the configured model is available before asking a question.
        """
    )


def render_documents(store: ChromaVectorStore, embedder: SentenceTransformerEmbedder) -> None:
    st.title("Documents")
    st.write("Add PDF, DOCX, or TXT files to your knowledge base.")
    document_notice = st.session_state.pop("document_notice", None)
    if document_notice:
        st.success(document_notice)
    chunk_size = st.session_state.get("chunk_size", config.DEFAULT_CHUNK_SIZE)
    overlap = st.session_state.get("chunk_overlap", config.DEFAULT_CHUNK_OVERLAP)
    uploads = st.file_uploader(
        "Choose one or more documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if st.button("Add to knowledge base", type="primary", disabled=not uploads):
        with st.spinner("Extracting text, embedding chunks locally, and indexing..."):
            report = ingest_documents(
                ((upload.name, upload.getvalue()) for upload in uploads or []),
                store=store,
                embedder=embedder,
                chunk_size=chunk_size,
                overlap=overlap,
            )
        st.session_state.pop("last_query", None)
        st.session_state.pop("last_benchmark", None)
        st.session_state["last_ingestion_duration_ms"] = report.duration_ms
        st.success(
            f"Added {report.uploaded_documents} documents and "
            f"{report.chunks_created} chunks."
        )
        for error in report.errors:
            st.error(error)

    st.subheader("Indexed documents")
    chunks = store.all_chunks()
    indexed_files = store.indexed_files()
    if indexed_files:
        for file in indexed_files:
            file_column, action_column = st.columns([5, 1])
            page_label = (
                str(file["page_count"]) if file["page_count"] else "—"
            )
            file_column.markdown(
                f"**{file['filename']}**  \n"
                f"Pages: {page_label} · Chunks: {file['chunk_count']}"
            )
            if action_column.button(
                "Delete",
                key=f"delete_document_{file['filename']}",
                help=f"Delete all indexed chunks for {file['filename']}",
            ):
                deleted_chunks = store.delete_filename(file["filename"])
                st.session_state.pop("last_query", None)
                st.session_state.pop("last_benchmark", None)
                st.session_state["document_notice"] = (
                    f"Deleted {file['filename']} and {deleted_chunks} indexed chunks."
                )
                st.rerun()
    else:
        st.info("No documents have been indexed yet.")

    with st.expander("Advanced"):
        chunk_size = st.slider(
            "Chunk size (characters)",
            200,
            config.MAX_CHUNK_SIZE,
            config.DEFAULT_CHUNK_SIZE,
            50,
            key="chunk_size",
        )
        overlap = st.slider(
            "Chunk overlap (characters)",
            0,
            min(400, chunk_size - 1),
            config.DEFAULT_CHUNK_OVERLAP,
            10,
            key="chunk_overlap",
        )
        st.caption(
            f"Chunking uses {chunk_size} characters with {overlap} characters of overlap. "
            "This is character-based; token counts are measured separately."
        )
        st.caption(
            "Chunk settings apply to new documents. To re-chunk existing documents, reset "
            "the collection and upload them again."
        )
        if chunks:
            st.dataframe(
                [
                    {
                        "filename": row["filename"],
                        "page": (
                            str(row["page_number"])
                            if row["page_number"] > 0
                            else "Not provided"
                        ),
                        "chunk_id": row["chunk_id"],
                        "chunk_size": row["chunk_size"],
                        "chunk_overlap": row["chunk_overlap"],
                        "content_hash": row["content_hash"],
                        "text": row["text"],
                    }
                    for row in chunks
                ],
                use_container_width=True,
                hide_index=True,
            )
        st.subheader("Reset collection")
        st.warning("Reset permanently deletes all locally indexed chunks in this project.")
        confirmed = st.checkbox(
            "I understand and want to delete the active document collection."
        )
        if st.button("Reset document collection", disabled=not confirmed):
            store.reset()
            st.session_state.pop("last_query", None)
            st.session_state.pop("last_benchmark", None)
            st.success(
                "The document collection was reset. Previous source citations are no longer active."
            )
            st.rerun()


def _source_dict(source, include_distance: bool = True) -> dict:
    result = {
        "citation_id": source.citation_id,
        "document_id": source.document_id,
        "filename": source.filename,
        "page_number": source.page_number,
        "chunk_id": source.chunk_id,
        "content_hash": source.content_hash,
        "text": source.text,
        "chunk_size": source.chunk_size,
        "chunk_overlap": source.chunk_overlap,
    }
    if include_distance:
        result["distance"] = source.distance
    return result


def _escape_markdown_label(label: str) -> str:
    for character in r"\`*_{}[]()#+-.!|>~":
        label = label.replace(character, f"\\{character}")
    return label


def _source_label(source: dict) -> str:
    page = (
        f" · p.{source['page_number']}"
        if source.get("page_number") is not None
        else ""
    )
    return f"{source['filename']}{page}"


def render_citation_support_review(
    result: dict, metrics_store: MetricsStore
) -> None:
    claims = extract_answer_claims(result.get("answer") or "")
    if not claims or result["record"]["answer_status"] != "success":
        return

    st.subheader("Manual citation-support review")
    st.caption(
        "The answer is split into sentence-sized claims for review. Decide whether the "
        "cited passage(s) support each claim. Uncited claims cannot be marked supported. "
        "This human review is separate from citation-reference validity."
    )
    with st.form("citation_support_review"):
        assessments = {}
        for index, claim in enumerate(claims):
            citation_ids = claim["citation_ids"]
            st.markdown(
                f"**Claim {index + 1}:** {claim['text']} "
                f"({' '.join(f'[{item}]' for item in citation_ids) or 'no citation'})"
            )
            options = (
                ["Not reviewed", "Supported", "Not supported"]
                if citation_ids
                else ["Not reviewed", "Not supported"]
            )
            assessments[str(index)] = st.selectbox(
                f"Is claim {index + 1} supported by its cited passage(s)?",
                options,
                key=f"support_{result['record_id']}_{index}",
            )
        submitted = st.form_submit_button("Save citation-support review")
    if submitted:
        booleans = {
            claim_index: (
                True if value == "Supported" else False if value == "Not supported" else None
            )
            for claim_index, value in assessments.items()
        }
        support = evaluate_citation_support(claims, booleans)
        result["record"]["citation_support_score"] = support["score"]
        result["record"]["citation_support_evaluated_claims"] = support[
            "evaluated_claims"
        ]
        result["record"]["supported_claim_count"] = support["supported_claims"]
        result["record"]["unsupported_claim_count"] = support[
            "unsupported_claims"
        ]
        metrics_store.update(result["record_id"], result["record"])
        st.success(
            "Manual review saved."
            if support["score"] is not None
            else "No answer claims were marked as reviewed."
        )


def render_ask(
    store: ChromaVectorStore,
    embedder: SentenceTransformerEmbedder,
    metrics_store: MetricsStore,
) -> None:
    st.title("Ask")
    top_k = config.DEFAULT_TOP_K
    with st.form("ask_question"):
        question = st.text_area(
            "Question",
            placeholder="Ask something about your documents.",
            height=100,
        )
        submitted = st.form_submit_button("Ask", type="primary")
    if submitted:
        started = time.perf_counter()
        record = new_query_record()
        record["question"] = question.strip()
        record["active_documents"] = store.document_count()
        record["top_k"] = top_k
        record["model_name"] = config.MODEL_NAME
        record["model_configuration"] = {
            "temperature": GENERATION_TEMPERATURE,
            "top_k": top_k,
        }
        record["embedding_indexing_duration_ms"] = st.session_state.get(
            "last_ingestion_duration_ms"
        )
        try:
            with st.spinner("Searching your documents..."):
                sources, retrieval_latency = retrieve(
                    question.strip(), store=store, embedder=embedder, top_k=top_k
                )
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(f"Retrieval could not complete: {exc}")
            return

        record["retrieval_latency_ms"] = retrieval_latency
        record["retrieved_chunks"] = len(sources)
        record["retrieved_chunk_ids"] = [source.chunk_id for source in sources]
        record["distances"] = [source.distance for source in sources]
        record["retrieved_chunk_sizes"] = [source.chunk_size for source in sources]
        record["retrieved_chunk_overlaps"] = [source.chunk_overlap for source in sources]
        record["context_characters"] = sum(len(source.text) for source in sources)
        record["estimated_context_tokens"] = estimate_tokens(
            "\n\n".join(source.text for source in sources)
        )
        record["estimated_prompt_tokens"] = estimate_tokens(
            SYSTEM_INSTRUCTION
            + "\n\n"
            + build_prompt(question.strip(), sources)
        )

        conflicts = detect_conflicts(sources)
        conflict_details = [
            {
                "first": _source_dict(conflict.first),
                "second": _source_dict(conflict.second),
                "reason": conflict.reason,
                "shared_terms": list(conflict.shared_terms),
            }
            for conflict in conflicts
        ]
        if not sources or not has_sufficient_relevance(
            sources, max_distance=config.MIN_RELEVANCE_DISTANCE
        ):
            answer = INSUFFICIENT_CONTEXT_ANSWER
            record["answer_status"] = "insufficient_context"
            record["estimated_answer_tokens"] = estimate_tokens(answer)
            record["estimated_total_tokens"] = (
                record["estimated_prompt_tokens"] + record["estimated_answer_tokens"]
            )
            record["generation_latency_ms"] = None
            record["citation_reference_validity_rate"] = None
            record["end_to_end_latency_ms"] = (time.perf_counter() - started) * 1000
            record_id = metrics_store.record(record)
            st.session_state["last_query"] = {
                "record": record,
                "record_id": record_id,
                "answer": answer,
                "sources": [_source_dict(source) for source in sources],
                "conflicts": conflict_details,
                "valid_references": [],
                "invalid_references": [],
            }
        else:
            generation_started = time.perf_counter()
            try:
                with st.spinner(f"Generating an answer with {config.MODEL_NAME}..."):
                    generated = generate_answer(question.strip(), sources)
            except GenerationError as exc:
                record["answer_status"] = "api_failure"
                record["error"] = str(exc)
                record["generation_latency_ms"] = (time.perf_counter() - generation_started) * 1000
                record["end_to_end_latency_ms"] = (time.perf_counter() - started) * 1000
                record_id = metrics_store.record(record)
                st.session_state["last_query"] = {
                    "record": record,
                    "record_id": record_id,
                    "answer": None,
                    "error": str(exc),
                    "sources": [_source_dict(source) for source in sources],
                    "conflicts": conflict_details,
                    "valid_references": [],
                    "invalid_references": [],
                }
            else:
                valid_ids = {source.citation_id for source in sources}
                answer, valid, invalid = validate_citations(generated.answer, valid_ids)
                record["answer_status"] = "success"
                record["generation_latency_ms"] = (
                    time.perf_counter() - generation_started
                ) * 1000
                record["model_name"] = generated.model
                record["model_configuration"] = {
                    "temperature": GENERATION_TEMPERATURE
                }
                record["actual_provider_usage"] = generated.actual_usage
                record["citation_count"] = len(valid) + len(invalid)
                record["valid_citation_count"] = len(valid)
                record["invalid_citation_count"] = len(invalid)
                record["citation_reference_validity_rate"] = (
                    len(valid) / (len(valid) + len(invalid))
                    if valid or invalid
                    else None
                )
                record["estimated_prompt_tokens"] = estimate_tokens(
                    SYSTEM_INSTRUCTION + "\n\n" + generated.prompt
                )
                record["estimated_answer_tokens"] = estimate_tokens(answer)
                record["estimated_total_tokens"] = (
                    record["estimated_prompt_tokens"] + record["estimated_answer_tokens"]
                )
                record["end_to_end_latency_ms"] = (time.perf_counter() - started) * 1000
                record_id = metrics_store.record(record)
                st.session_state["last_query"] = {
                    "record": record,
                    "record_id": record_id,
                    "answer": answer,
                    "sources": [_source_dict(source) for source in sources],
                    "conflicts": conflict_details,
                    "valid_references": valid,
                    "invalid_references": invalid,
                }
        st.rerun()

    result = st.session_state.get("last_query")
    if not result:
        return

    if result.get("answer") is None:
        st.error(result.get("error", "Answer generation failed."))
    else:
        display_answer, _ = format_document_citations(
            result["answer"], result["sources"]
        )
        for source in result["sources"]:
            citation_id = source["citation_id"]
            page = (
                f" p.{source['page_number']}"
                if source["page_number"] is not None
                else ""
            )
            old_label = _escape_markdown_label(
                f"{citation_id}: {source['filename']}{page}"
            )
            new_label = _escape_markdown_label(_source_label(source))
            display_answer = display_answer.replace(
                f"[{old_label}](#source-{citation_id})",
                f"[{new_label}](#source-{citation_id})",
            )
        with st.container(border=True):
            if result["record"]["answer_status"] == "insufficient_context":
                st.warning(result["answer"])
            else:
                st.markdown(display_answer)

    if result["conflicts"]:
        st.info("The sources disagree on this.")
        with st.expander("Conflict details"):
            for conflict in result["conflicts"]:
                st.write(conflict["reason"])
                first, second = st.columns(2)
                first.markdown(f"**{_source_label(conflict['first'])}**")
                first.write(conflict["first"]["text"])
                second.markdown(f"**{_source_label(conflict['second'])}**")
                second.write(conflict["second"]["text"])

    record = result["record"]
    elapsed_ms = record.get("end_to_end_latency_ms")
    elapsed = f"{elapsed_ms / 1000:.1f}" if elapsed_ms is not None else "—"
    passages = record.get("retrieved_chunks", len(result["sources"]))
    usage = record.get("actual_provider_usage") or {}
    tokens = usage.get("total_tokens")
    if tokens is None:
        tokens = record.get("estimated_total_tokens")
    token_label = str(tokens) if tokens is not None else "—"
    if result.get("answer") is not None:
        st.caption(f"Answered in {elapsed} s · {passages} passages · {token_label} tokens")

    cited_ids = set(result.get("valid_references") or [])
    cited_sources = []
    seen_chunks = set()
    for source in result["sources"]:
        if cited_ids and source["citation_id"] not in cited_ids:
            continue
        if source["chunk_id"] not in seen_chunks:
            cited_sources.append(source)
            seen_chunks.add(source["chunk_id"])
    with st.expander("Sources used"):
        if not cited_sources:
            st.info("No retrieved passages are available.")
        for source in cited_sources:
            st.markdown(
                f'<div id="source-{source["citation_id"]}"></div>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**{_source_label(source)}**")
            st.write(source["text"][:400])


def render_performance(store: ChromaVectorStore, metrics_store: MetricsStore) -> None:
    st.title("Model performance")
    report_path = config.BASE_DIR / "data" / "model_report" / "metrics.json"
    try:
        with report_path.open(encoding="utf-8") as report_file:
            report = json.load(report_file)
    except FileNotFoundError:
        st.error(f"Model report not found: {report_path}")
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read model report: {exc}")
    else:
        _render_model_comparison(report)
        _render_training_and_parameters(report)
        _render_token_sizes(report)

    _render_live_speed(metrics_store)
    with st.expander("Advanced measurements"):
        _render_advanced_measurements(store, metrics_store)
    with st.expander("How it works"):
        render_about()


def _render_model_comparison(report: dict) -> None:
    st.subheader("Fine-tuned model vs base model")
    comparison = report["comparison"]
    ratio_metrics = {
        "Citation precision",
        "Citation recall",
        "Valid citation IDs",
        "Conflict handled (both sources cited)",
        "Abstain accuracy",
        "False-abstain rate",
        "ROUGE-L",
        "Claims supported (word-overlap estimate)",
    }
    table_rows = []
    for item in comparison:
        base = item["base"]
        fine_tuned = item["fine_tuned"]
        change = fine_tuned - base
        is_ratio = item["metric"] in ratio_metrics
        table_rows.append(
            {
                "Metric": item["metric"],
                "Base": f"{base:.1%}" if is_ratio else f"{base:g}",
                "Fine-tuned": (
                    f"{fine_tuned:.1%}" if is_ratio else f"{fine_tuned:g}"
                ),
                "Change": (
                    f"{change * 100:+.1f} pp" if is_ratio else f"{change:+.2f}"
                ),
            }
        )
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    chart_metrics = {
        "Citation precision": "Citation precision",
        "Citation recall": "Citation recall",
        "Conflict handled (both sources cited)": "Conflict handled",
        "Abstain accuracy": "Abstain accuracy",
        "ROUGE-L": "ROUGE-L",
    }
    comparison_by_name = {item["metric"]: item for item in comparison}
    st.bar_chart(
        [
            {
                "Metric": label,
                "Base": comparison_by_name[name]["base"],
                "Fine-tuned": comparison_by_name[name]["fine_tuned"],
            }
            for name, label in chart_metrics.items()
        ],
        x="Metric",
        y=["Base", "Fine-tuned"],
    )
    st.caption(report["test_set"])
    st.caption(f"Source: {report['source']}")
    for note in report["notes"]:
        st.caption(note)


def _render_training_and_parameters(report: dict) -> None:
    st.subheader("Training and parameters")
    training = report["training"]
    trainable_percent = (
        training["trainable_parameters"] / training["total_parameters"] * 100
    )
    cards = [
        ("Base model", training["base_model"]),
        (
            "Trainable parameters",
            f"{training['trainable_parameters'] / 1_000_000:.2f} M of "
            f"{training['total_parameters'] / 1_000_000_000:.2f} B "
            f"({trainable_percent:.2f}%)",
        ),
        ("LoRA rank / alpha", f"{training['lora_rank']} / {training['lora_alpha']}"),
        ("Epochs / steps", f"{training['epochs']} / {training['steps']}"),
        ("Learning rate", f"{training['learning_rate']:g}"),
        ("Effective batch size", str(training["effective_batch_size"])),
        ("Training time", f"{training['train_minutes']:g} min"),
        (
            "Validation loss / perplexity",
            f"{training['final_validation_loss']:g} / {training['perplexity']:g}",
        ),
        ("Exported model", training["exported_model"]),
    ]
    for start in range(0, len(cards), 3):
        columns = st.columns(3)
        for column, (label, value) in zip(columns, cards[start : start + 3]):
            column.metric(label, value)

    epoch_losses = training["epoch_losses"]
    st.dataframe(epoch_losses, use_container_width=True, hide_index=True)
    loss_curve = config.BASE_DIR / "data" / "model_report" / "loss_curve.png"
    if loss_curve.is_file():
        st.image(str(loss_curve))
    else:
        st.line_chart(
            [
                {
                    "Epoch": item["epoch"],
                    "Train": item["train"],
                    "Validation": item["validation"],
                }
                for item in epoch_losses
            ],
            x="Epoch",
            y=["Train", "Validation"],
        )
    if (
        len(epoch_losses) >= 2
        and epoch_losses[-1]["validation"] > epoch_losses[-2]["validation"]
    ):
        st.caption("Validation loss rose in epoch 3 (slight overfitting).")
    else:
        st.caption("Validation loss did not rise in the final epoch.")

    with st.expander("Advanced"):
        st.write(f"Training method: {training['method']}")
        st.write(
            f"Trainable parameters: {training['trainable_parameters']} of "
            f"{training['total_parameters']} ({trainable_percent:.2f}%)"
        )
        st.write(f"LoRA dropout: {training['lora_dropout']}")
        st.write(f"Target modules: {', '.join(training['target_modules'])}")
        st.write(
            f"Per-device batch size: {training['per_device_batch_size']} · "
            f"Gradient accumulation steps: {training['gradient_accumulation_steps']}"
        )
        st.write(f"Optimizer: {training['optimizer']}")
        st.write(f"Maximum sequence length: {training['max_seq_length']}")
        st.write(
            f"Training examples: {training['train_examples']} · "
            f"Validation examples: {training['validation_examples']}"
        )
        st.write(f"GPU: {training['gpu']}")
        st.write(
            f"GPU memory after model load: "
            f"{training['gpu_memory_after_model_load_gb']} GB"
        )
        st.write(f"Served with: {training['served_with']}")


def _render_token_sizes(report: dict) -> None:
    st.subheader("Token and context sizes")
    tokens = report["tokens"]
    cards = [
        ("Training average", tokens["train_avg"]),
        ("Training maximum", tokens["train_max"]),
        ("Test average input", tokens["test_avg_input"]),
        ("Model context window", tokens["model_context_window"]),
        ("Ollama context", tokens["ollama_num_ctx"]),
        ("Retrieval top-k", tokens["retrieval_top_k"]),
        ("Chunk size (characters)", tokens["chunk_size_characters"]),
        ("Chunk overlap (characters)", tokens["chunk_overlap_characters"]),
    ]
    for start in range(0, len(cards), 4):
        columns = st.columns(4)
        for column, (label, value) in zip(columns, cards[start : start + 4]):
            column.metric(label, value)


def _render_live_speed(metrics_store: MetricsStore) -> None:
    st.subheader("Live speed on this computer")
    started_at = datetime.fromisoformat(
        st.session_state["model_performance_started_at"]
    )
    model_records = []
    for record in metrics_store.recent():
        if record.get("model_name") != config.MODEL_NAME:
            continue
        if record.get("answer_status") not in {
            "success",
            "insufficient_context",
            "api_failure",
        }:
            continue
        timestamp = datetime.fromisoformat(record["timestamp"])
        if timestamp >= started_at:
            model_records.append((timestamp, record))
    model_records.sort(key=lambda item: item[0])
    warmup_records = model_records[1:]

    qualifying_rows = []
    for _, record in warmup_records:
        if record.get("answer_status") != "success":
            continue
        usage = record.get("actual_provider_usage") or {}
        answer_tokens = usage.get("answer_tokens")
        if answer_tokens is None:
            answer_tokens = record.get("estimated_answer_tokens")
        measurements = (
            record.get("retrieval_latency_ms"),
            record.get("generation_latency_ms"),
            record.get("end_to_end_latency_ms"),
            answer_tokens,
        )
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in measurements
        ):
            qualifying_rows.append(measurements)

    if len(qualifying_rows) < 5:
        st.info("Ask at least 5 questions first.")
        return

    labels = (
        "Median retrieval (ms)",
        "Median generation (ms)",
        "Median end-to-end (ms)",
        "Average tokens per answer",
    )
    columns = st.columns(4)
    for index, label in enumerate(labels):
        values = [row[index] for row in qualifying_rows]
        value = (
            sum(values) / len(values)
            if index == 3
            else statistics.median(values)
        )
        columns[index].metric(label, f"{value:.1f}")


def _render_advanced_measurements(
    store: ChromaVectorStore, metrics_store: MetricsStore
) -> None:
    records = metrics_store.recent()
    if not records:
        st.info("No question measurements yet. Run a query to collect real measurements.")
        return
    summary = query_summary(records)
    columns = st.columns(5)
    columns[0].metric("Queries tested", summary["queries_tested"])
    columns[1].metric("Successful answers", summary["successful_answers"])
    columns[2].metric("Insufficient context", summary["insufficient_context_responses"])
    columns[3].metric("API failures", summary["api_failures"])
    columns[4].metric("Benchmark questions", summary["benchmark_questions"])
    columns = st.columns(4)
    columns[0].metric("Active documents now", store.document_count())
    measured_queries = [
        record
        for record in records
        if record.get("answer_status")
        in {"success", "insufficient_context", "api_failure"}
    ]
    avg_retrieval = average_measurement(measured_queries, "retrieval_latency_ms")
    avg_generation = average_measurement(measured_queries, "generation_latency_ms")
    avg_end_to_end = average_measurement(measured_queries, "end_to_end_latency_ms")
    columns[1].metric(
        "Avg retrieval (ms)",
        f"{avg_retrieval:.1f}" if avg_retrieval is not None else "—",
    )
    columns[2].metric(
        "Avg generation (ms)",
        f"{avg_generation:.1f}" if avg_generation is not None else "—",
    )
    columns[3].metric(
        "Avg end-to-end (ms)",
        f"{avg_end_to_end:.1f}" if avg_end_to_end is not None else "—",
    )

    st.subheader("Answer quality")
    quality_columns = st.columns(4)
    reference_validity = average_measurement(
        records, "citation_reference_validity_rate"
    )
    citation_support = average_measurement(records, "citation_support_score")
    groundedness = average_measurement(records, "groundedness_score")
    quality_columns[0].metric(
        "Citation ID validity",
        f"{reference_validity:.1%}" if reference_validity is not None else "Not evaluated",
    )
    quality_columns[1].metric(
        "Human-checked citation support",
        f"{citation_support:.1%}" if citation_support is not None else "Not evaluated",
    )
    quality_columns[2].metric(
        "Human-labeled groundedness",
        f"{groundedness:.1%}" if groundedness is not None else "Not evaluated",
    )
    quality_columns[3].metric(
        "Unsupported claims recorded",
        sum(int(record.get("unsupported_claim_count") or 0) for record in records),
    )

    st.subheader("Latency by query")
    latency_records = [
        record
        for record in records
        if record.get("answer_status")
        in {"success", "insufficient_context", "api_failure"}
    ]
    if latency_records:
        st.line_chart(
            [
                {
                    "query": record["timestamp"],
                    "retrieval_ms": record["retrieval_latency_ms"],
                    "generation_ms": record["generation_latency_ms"],
                    "end_to_end_ms": record["end_to_end_latency_ms"],
                }
                for record in reversed(latency_records)
            ],
            x="query",
        )
    else:
        st.info(
            "No user-query latency measurements yet. Run a question to populate this chart."
        )
    st.subheader("Effect of retrieval and chunk settings")
    st.caption(
        "Compare repeated runs with different top-k values. Points are colored by the "
        "chunk size recorded in the retrieved chunks; reset and re-index to compare a "
        "different chunk size fairly."
    )
    comparison_rows = [
        {
            "top_k": record.get("top_k"),
            "context_characters": record.get("context_characters"),
            "retrieval_latency_ms": record.get("retrieval_latency_ms"),
            "chunk_size": (
                record["retrieved_chunk_sizes"][0]
                if record.get("retrieved_chunk_sizes")
                else None
            ),
        }
        for record in records
        if record.get("top_k") is not None
    ]
    settings = {
        (row["top_k"], row["chunk_size"])
        for row in comparison_rows
        if row["chunk_size"] is not None
    }
    if len(settings) > 1:
        context_chart, latency_chart = st.columns(2)
        context_chart.scatter_chart(
            comparison_rows,
            x="top_k",
            y="context_characters",
            color="chunk_size",
        )
        latency_chart.scatter_chart(
            comparison_rows,
            x="top_k",
            y="retrieval_latency_ms",
            color="chunk_size",
        )
    else:
        st.info(
            "Run queries with at least two different top-k or chunk-size settings "
            "to compare their effects."
        )
    st.subheader("Estimated and provider-reported tokens")
    token_records = [
        record
        for record in records
        if any(
            record.get(field) is not None
            for field in (
                "estimated_prompt_tokens",
                "estimated_context_tokens",
                "estimated_answer_tokens",
            )
        )
        or (record.get("actual_provider_usage") or {}).get("total_tokens")
        is not None
    ]
    if token_records:
        st.bar_chart(
            [
                {
                    "query": record["timestamp"],
                    "estimated_prompt": record["estimated_prompt_tokens"],
                    "estimated_context": record["estimated_context_tokens"],
                    "estimated_answer": record["estimated_answer_tokens"],
                    "provider_total": (record.get("actual_provider_usage") or {}).get(
                        "total_tokens"
                    ),
                }
                for record in reversed(token_records)
            ],
            x="query",
        )
    else:
        st.info(
            "No token measurements yet. Run a question with generation enabled to "
            "populate this chart."
        )
    st.subheader("Retrieval and citation quality")
    retrieval_records = [
        record for record in records if record.get("precision_at_k") is not None
    ]
    if retrieval_records:
        st.caption(
            "Retrieval scores below come from labeled evaluation questions. "
            "Unlabeled questions are not included."
        )
        st.bar_chart(
            [
                {
                    "question": record["question"],
                    "precision_at_k": record["precision_at_k"],
                    "recall_at_k": record["recall_at_k"],
                    "mrr": record["mrr"],
                }
                for record in retrieval_records
            ],
            x="question",
        )
    else:
        st.info("Retrieval scores: Not evaluated. Add labels and run the Evaluation benchmark.")
    citation_records = [
        record
        for record in records
        if record.get("citation_reference_validity_rate") is not None
        or record.get("citation_support_score") is not None
    ]
    if citation_records:
        st.bar_chart(
            [
                {
                    "question": record["question"],
                    "reference_validity": record["citation_reference_validity_rate"],
                    "manual_claim_support": record["citation_support_score"],
                }
                for record in citation_records
            ],
            x="question",
        )
    else:
        st.info("Citation quality: Not evaluated. Run a generated answer and review its claims.")
    table_columns = [
        "timestamp",
        "question",
        "active_documents",
        "retrieved_chunks",
        "retrieval_latency_ms",
        "generation_latency_ms",
        "end_to_end_latency_ms",
        "embedding_indexing_duration_ms",
        "retrieved_chunk_ids",
        "distances",
        "top_k",
        "retrieved_chunk_sizes",
        "retrieved_chunk_overlaps",
        "context_characters",
        "estimated_context_tokens",
        "estimated_prompt_tokens",
        "estimated_answer_tokens",
        "estimated_total_tokens",
        "actual_provider_usage",
        "model_name",
        "model_configuration",
        "citation_count",
        "valid_citation_count",
        "invalid_citation_count",
        "citation_reference_validity_rate",
        "citation_support_score",
        "citation_support_evaluated_claims",
        "supported_claim_count",
        "unsupported_claim_count",
        "precision_at_k",
        "recall_at_k",
        "mrr",
        "groundedness_score",
        "answer_status",
        "error",
    ]
    st.subheader("Recent query measurements")
    st.dataframe(
        [{key: record.get(key) for key in table_columns} for record in records],
        use_container_width=True,
        hide_index=True,
    )
    json_data = json.dumps(records, indent=2, ensure_ascii=False)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=table_columns, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        writer.writerow(
            {
                **record,
                "retrieved_chunk_ids": json.dumps(record.get("retrieved_chunk_ids", [])),
                "distances": json.dumps(record.get("distances", [])),
                "retrieved_chunk_sizes": json.dumps(
                    record.get("retrieved_chunk_sizes", [])
                ),
                "retrieved_chunk_overlaps": json.dumps(
                    record.get("retrieved_chunk_overlaps", [])
                ),
                "actual_provider_usage": json.dumps(record.get("actual_provider_usage")),
                "model_configuration": json.dumps(record.get("model_configuration")),
            }
        )
    export_json, export_csv = st.columns(2)
    export_json.download_button(
        "Export measurements as JSON",
        json_data,
        file_name="documind-query-measurements.json",
        mime="application/json",
    )
    export_csv.download_button(
        "Export measurements as CSV",
        csv_buffer.getvalue(),
        file_name="documind-query-measurements.csv",
        mime="text/csv",
    )
    st.info("Cost estimate unavailable. No pricing configuration is included.")
    st.caption(
        f"Embedding/indexing time from the most recent ingestion: "
        f"{st.session_state.get('last_ingestion_duration_ms', 'not measured in this session')} ms."
    )


def _render_legacy_evaluation(
    store: ChromaVectorStore,
    embedder: SentenceTransformerEmbedder,
    metrics_store: MetricsStore,
) -> None:
    st.title("Evaluation and benchmark")
    st.write(
        "Edit the question set, expected answers, and comma-separated relevant chunk IDs. "
        "Chunk IDs are available in the Documents page and in retrieved-source details. "
        "Unlabeled rows are excluded instead of receiving guessed scores."
    )
    try:
        dataset = load_dataset(config.EVALUATION_DATASET_PATH)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load evaluation_dataset.json: {exc}")
        return
    editor_rows = [
        {
            "question": row.get("question", ""),
            "expected_answer": row.get("expected_answer", ""),
            "relevant_chunk_ids": ", ".join(row.get("relevant_chunk_ids", [])),
            "claims_total": row.get("claims_total", 0),
            "claims_supported": row.get("claims_supported", 0),
        }
        for row in dataset
    ]
    edited_rows = st.data_editor(
        editor_rows,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "question": st.column_config.TextColumn("Question", required=True),
            "expected_answer": st.column_config.TextColumn("Expected answer"),
            "relevant_chunk_ids": st.column_config.TextColumn(
                "Relevant chunk IDs (comma-separated)"
            ),
            "claims_total": st.column_config.NumberColumn("Factual claims", min_value=0, step=1),
            "claims_supported": st.column_config.NumberColumn(
                "Supported claims", min_value=0, step=1
            ),
        },
        key="evaluation_dataset_editor",
    )
    if st.button("Save evaluation labels"):
        try:
            save_dataset(config.EVALUATION_DATASET_PATH, edited_rows)
        except (OSError, TypeError, ValueError) as exc:
            st.error(f"Could not save the evaluation dataset: {exc}")
        else:
            st.session_state.pop("last_benchmark", None)
            st.success("Evaluation labels saved to evaluation_dataset.json.")
            st.rerun()

    if st.button("Run labeled retrieval benchmark", type="primary"):
        benchmark_details = {}

        def retrieve_ids(question: str, k: int) -> list[str]:
            results, latency = retrieve(
                question,
                store=store,
                embedder=embedder,
                top_k=k,
            )
            benchmark_details[question] = (results, latency)
            return [source.chunk_id for source in results]

        with st.spinner("Retrieving against labeled questions..."):
            try:
                result = evaluate_retrieval_dataset(
                    dataset,
                    retrieve_ids,
                    k=config.DEFAULT_TOP_K,
                )
            except (ValueError, RuntimeError, OSError) as exc:
                st.error(f"Benchmark could not complete: {exc}")
                return
        for row in result["rows"]:
            sources, retrieval_latency = benchmark_details[row["question"]]
            record = new_query_record()
            record.update(
                {
                    "question": row["question"],
                    "active_documents": store.document_count(),
                    "retrieved_chunks": len(sources),
                    "retrieval_latency_ms": retrieval_latency,
                    "generation_latency_ms": None,
                    "end_to_end_latency_ms": None,
                    "retrieved_chunk_ids": [source.chunk_id for source in sources],
                    "distances": [source.distance for source in sources],
                    "top_k": config.DEFAULT_TOP_K,
                    "model_configuration": {
                        "temperature": GENERATION_TEMPERATURE,
                        "top_k": config.DEFAULT_TOP_K,
                    },
                    "retrieved_chunk_sizes": [source.chunk_size for source in sources],
                    "retrieved_chunk_overlaps": [source.chunk_overlap for source in sources],
                    "context_characters": sum(len(source.text) for source in sources),
                    "estimated_context_tokens": estimate_tokens(
                        "\n\n".join(source.text for source in sources)
                    ),
                    "precision_at_k": row["precision_at_k"],
                    "recall_at_k": row["recall_at_k"],
                    "mrr": row["mrr"],
                    "groundedness_score": row["groundedness_score"],
                    "answer_status": "retrieval_evaluation",
                }
            )
            metrics_store.record(record)
        st.session_state["last_benchmark"] = result

    result = st.session_state.get("last_benchmark")
    if result is None:
        st.info("Run the benchmark to measure retrieval against the labels currently saved.")
        return
    if result["evaluated_questions"] == 0:
        st.warning("Not evaluated: add at least one relevant chunk ID to the dataset.")
    metrics = st.columns(4)
    for column, label, key in zip(
        metrics,
        ("Evaluated questions", "Precision@k", "Recall@k", "MRR"),
        ("evaluated_questions", "precision_at_k", "recall_at_k", "mrr"),
    ):
        value = result[key]
        column.metric(label, f"{value:.3f}" if isinstance(value, float) else value if value is not None else "Not evaluated")
    groundedness = result["groundedness_score"]
    st.metric(
        "Groundedness (manual claim labels)",
        f"{groundedness:.3f}" if groundedness is not None else "Not evaluated",
    )
    st.caption(result["groundedness_method"])
    st.dataframe(result["rows"], use_container_width=True, hide_index=True)
    st.download_button(
        "Export benchmark results as JSON",
        json.dumps(result, indent=2, ensure_ascii=False),
        file_name="documind-evaluation-results.json",
        mime="application/json",
    )


def _render_evaluation_advanced(
    store: ChromaVectorStore,
    embedder: SentenceTransformerEmbedder,
    metrics_store: MetricsStore,
) -> None:
    _render_legacy_evaluation(store, embedder, metrics_store)
    st.subheader("Manual citation-support review")
    support_score = average_measurement(
        metrics_store.recent(), "citation_support_score"
    )
    st.metric(
        "Saved manual citation-support average",
        f"{support_score:.1%}" if support_score is not None else "n/a",
    )
    last_query = st.session_state.get("last_query")
    if last_query and last_query.get("answer") and (
        last_query.get("record", {}).get("answer_status") == "success"
    ):
        render_citation_support_review(last_query, metrics_store)
    else:
        st.info("No generated answer is available for manual citation review.")


def render_evaluation(
    store: ChromaVectorStore,
    embedder: SentenceTransformerEmbedder,
    metrics_store: MetricsStore,
) -> None:
    auto_eval.render_evaluation(
        store=store,
        embedder=embedder,
        benchmark_path=config.BASE_DIR / "eval" / "benchmark.json",
        results_path=config.BASE_DIR / "data" / "eval_results.json",
        advanced_renderer=lambda: _render_evaluation_advanced(
            store, embedder, metrics_store
        ),
    )


def render_about() -> None:
    st.markdown(
        """
        ```mermaid
        flowchart LR
          U[PDF / DOCX / TXT] --> X[Text extraction with source metadata]
          X --> C[Character chunking with overlap]
          C --> E[Local all-MiniLM-L6-v2 embeddings]
          E --> V[(Persistent ChromaDB)]
          Q[Question] --> QE[Same local embedding model]
          QE --> R[Top-k semantic retrieval]
          V --> R
          R -->|Retrieved passages| G[Local Ollama grounded generation]
          G --> A[Answer, citation checks, conflict review]
          R --> M[Latency, token, retrieval, citation metrics]
          G --> M
        ```

        **Design choices:** embeddings, vector search, and answer generation run locally
        through sentence-transformers, ChromaDB, and Ollama. Character chunking is easy
        to explain and configure; `tiktoken` provides estimates separately.

        **Limitations:** OCR is not applied to scanned PDFs; DOCX/TXT page numbers are
        unavailable; cosine distance is a ranking signal, not confidence; heuristic
        conflict detection can miss or over-report disagreements; generated citations
        need human semantic review; manual groundedness labels and relevant-chunk labels
        are required for meaningful evaluation. Actual token usage appears only when
        Ollama returns it.
        """
    )


def ollama_is_reachable() -> bool:
    parsed_url = urlsplit(config.OLLAMA_URL)
    tags_url = urlunsplit(
        (parsed_url.scheme, parsed_url.netloc, "/api/tags", "", "")
    )
    try:
        with urlopen(tags_url, timeout=1) as response:
            return response.status == 200
    except Exception:
        return False


def main() -> None:
    st.session_state.setdefault(
        "model_performance_started_at", datetime.now(timezone.utc).isoformat()
    )
    store = get_store()
    with st.sidebar:
        st.title("DocuMind AI")
        page = st.radio(
            "Navigate",
            ["Documents", "Ask", "Model performance", "Evaluation"],
            index=0 if store.document_count() == 0 else 1,
            label_visibility="collapsed",
        )
        ollama_status = "Reachable" if ollama_is_reachable() else "Unavailable"
        st.caption(f"Ollama: {ollama_status}")
        st.caption(f"Model: {config.MODEL_NAME}")
        st.caption(f"Indexed documents: {store.document_count()}")
        st.session_state["top_k"] = config.DEFAULT_TOP_K

    if page == "Documents":
        render_documents(store, get_embedder())
    elif page == "Ask":
        render_ask(store, get_embedder(), get_metrics_store())
    elif page == "Model performance":
        render_performance(store, get_metrics_store())
    else:
        render_evaluation(store, get_embedder(), get_metrics_store())


if __name__ == "__main__":
    main()
