# DocuMind Technical Architecture

## Scope and evidence

This document describes the checked-in Streamlit application, its Python modules,
the automatic benchmark configuration, and the model report in
`data/model_report/metrics.json`. Runtime behavior below is based on the application
code and configuration. Fine-tuning values are reported values from the model
report; the repository does not include the training notebook, raw trainer logs,
or a training script that independently reproduces them.

The main application is configured for local embeddings, a persistent local
ChromaDB collection, and an Ollama endpoint at `http://localhost:11434/api/chat`.
This establishes a local inference path. It is not evidence that every first-run
installation action is offline: model weights and Python packages must already be
available locally or be obtained during setup.

## 1. End-to-End Pipeline & System Architecture

### 1.1 Component map

```text
PDF / DOCX / TXT upload
        |
        v
ingestion.py: text extraction, content-hash duplicate check
        |
        v
chunking.py: page-scoped character chunks + overlap + source metadata
        |
        +--------------------------+
        |                          |
        v                          v
embeddings.py                 schemas.py
SentenceTransformer           SourcePage / TextChunk
all-MiniLM-L6-v2                    |
        |                            |
        +-------------+--------------+
                      v
vector_store.py: persistent ChromaDB collection
                      ^
                      | top-k cosine-distance search
                      |
question -> query embedding -> retrieval.py
                      |
                      v
generation.py: retrieved passages + question -> local Ollama chat API
                      |
                      v
citations.py: citation-reference validation and source rendering
conflict_detection.py: heuristic pairwise passage conflict detection
metrics.py: SQLite-backed per-query measurements
                      |
                      v
Streamlit answer, citations, conflict notice, source passages
```

The Streamlit entry point is `app.py`. It creates cached resources for the vector
store, sentence-transformer embedder, and metrics store. The UI has Documents,
Ask, Model performance, and Evaluation pages. The Ask path records normal user
query measurements. Automatic benchmark runs use `auto_eval.py` and save their own
result JSON rather than writing query records to the normal metrics database.

### 1.2 Document ingestion and parsing

`ingestion.py` accepts `.pdf`, `.docx`, and `.txt` files. Empty uploads and
unsupported extensions are reported as errors; content hashes are SHA-256 values
used internally to avoid indexing the same file contents twice.

| Input | Parser used by the repository | Text/source behavior |
|---|---|---|
| PDF | PyMuPDF (`fitz`) | Extracts text separately for each PDF page; page numbering starts at 1. A PDF with no extractable text is rejected. |
| DOCX | `python-docx` | Extracts paragraphs and table-cell text into one text block. DOCX pages are not inferred, so page metadata is unavailable. |
| TXT | Python UTF-8 decoder (`utf-8-sig`) | Decodes the file as UTF-8; text is represented as one logical source page with no page number. |

The implementation does **not** use `pypdf`. Text extraction operates on the
uploaded bytes in the application process. Scanned-PDF OCR is not implemented;
the parser requires extractable PDF text.

### 1.3 Character chunking and provenance

`chunking.py` applies `RecursiveCharacterTextSplitter` independently to each
`SourcePage`. Its current configuration uses `separators=[""]`, so splitting is
character-based rather than paragraph- or sentence-aware. Defaults are:

| Setting | Default |
|---|---:|
| Chunk size | 800 characters |
| Chunk overlap | 120 characters |

These are configurable in the Documents page for future indexing. The overlap
retains adjoining text across chunk boundaries. PDF page numbers are preserved
per chunk; DOCX and TXT chunks have no page number.

Each `TextChunk` carries the document identifier, file name, optional page number,
chunk identifier, SHA-256 content hash, text, and the chunking settings used.
Chunk identifiers follow the form
`<first-12-characters-of-document-id>-p<page-or-na>-c<zero-padded-index>`.
These identifiers and content hashes are operational provenance fields, not
evidence of semantic validity.

### 1.4 Local embeddings and persistent vector search

`embeddings.py` lazily loads
`sentence-transformers/all-MiniLM-L6-v2`. Document and query embeddings are
encoded with normalized vectors. The dependency is initialized on first use and
retained by Streamlit's resource cache.

`vector_store.py` opens a persistent ChromaDB client at `config.CHROMA_DIR`, which
defaults to `data/chroma`. It creates or reuses the collection
`documind_documents` with `hnsw:space` set to `cosine`. For each chunk, ChromaDB
stores the embedding, text, and metadata including filename, page, document ID,
chunk ID, content hash, chunk size, and overlap. A missing page is encoded in
Chroma metadata as `-1` and mapped back to `None` when retrieved.

`retrieval.py` embeds the question and requests `config.DEFAULT_TOP_K`, which is
4. The vector-store query caps the result count at the number of indexed chunks.
The returned rows are sorted by ascending cosine distance, then assigned
per-answer citation identifiers (`S1`, `S2`, and so on). The Ask page numbers
those ordered sources as `[Doc 1]`, `[Doc 2]`, etc. Generation proceeds only when
there is at least one source and the nearest source satisfies
`config.MIN_RELEVANCE_DISTANCE` (0.7). A distance is a ranking/dissimilarity
signal, not a calibrated probability or confidence score.

Top-K is fixed by application configuration for a query; it is not dynamically
chosen by a token-budget algorithm. The number of results can be lower than four
when fewer than four chunks exist.

### 1.5 Ollama generation and post-processing

`generation.py` builds a prompt containing retrieved passages labelled with
filename and, where available, PDF page. It sends a non-streaming HTTP POST to
the configured Ollama chat endpoint. The configured endpoint is
`http://localhost:11434/api/chat`; the configured model name is `policy-bot`.
The request supplies a system instruction to answer only from the passages, cite
factual claims as `[Doc N]`, describe both sides if sources disagree, and avoid
guessing when context is insufficient. The Ollama request sets temperature to
`0`, `num_ctx` to `4096`, and a 180-second timeout.

The Ask path then:

1. Calls `citations.validate_citations` with the retrieved source IDs. It maps
   `[Doc N]` to the corresponding retrieved `S<N>` identifier for validation.
   Invalid `[S<N>]` references are removed; document-style references are kept
   for display even when their number has no matching source. Therefore this is
   reference validation, not semantic entailment validation.
2. Calls `conflict_detection.detect_conflicts` on retrieved passages. Detected
   conflict details are retained with the query and rendered in a collapsed
   details panel.
3. Calls `metrics.MetricsStore.record` to persist normal Ask-query measurements
   in SQLite.
4. Renders document citations as filename/page labels and displays sources used.

The generator reports provider token counts if present. The application also
estimates token counts using `tiktoken`'s `cl100k_base` encoding. Those estimates
are not guaranteed to match the Ollama model's tokenizer.

## 2. Tech Stack & Dependencies

### Runtime and application dependencies

| Component | Repository evidence | Role |
|---|---|---|
| Python 3.11 | `requirements.txt` recommends Python 3.11 on Windows | Supported project environment used for the application and tests. |
| Streamlit 1.42.2 | `requirements.txt`, `app.py` | Four-page UI, forms, tables, charts, expanders, and session state. |
| ChromaDB 0.6.3 | `requirements.txt`, `vector_store.py` | Persistent local vector collection and cosine-space nearest-neighbor search. |
| sentence-transformers 3.4.1 | `requirements.txt`, `embeddings.py` | Local text embeddings using `all-MiniLM-L6-v2`. |
| PyMuPDF 1.25.3 | `requirements.txt`, `ingestion.py` (`fitz`) | PDF text extraction by page. |
| python-docx 1.1.2 | `requirements.txt`, `ingestion.py` | DOCX paragraph and table-cell extraction. |
| langchain-text-splitters 0.3.8 | `requirements.txt`, `chunking.py` | Recursive character splitter configured with an empty-string separator. |
| Ollama | `generation.py`, `config.py`, `run_app.bat` | Locally served chat inference through the Ollama HTTP API. The application expects the `policy-bot` model name. |
| requests | `requirements.txt`, `generation.py` | HTTP client for the Ollama chat request. |
| SQLite (`sqlite3`) | Python standard library, `metrics.py` | Persists JSON query-measurement payloads in `data/metrics.sqlite3` by default. |
| pandas 2.2.3 | `requirements.txt`, automatic evaluation UI | CSV serialization and tabular data support. |
| tiktoken 0.9.0 | `requirements.txt`, `metrics.py` | Estimated token counts using `cl100k_base`. |
| Pydantic 2.10.6 and Kubernetes 31.0.0 | `requirements.txt` | ChromaDB-related compatibility/runtime dependencies pinned by this project. |
| posthog | `requirements.txt` | Dependency included with the application environment; not called directly by project modules. Chroma's client is configured with anonymized telemetry disabled. |
| pytest 8.3.5 | `requirements.txt`, `tests/` | Unit and behavior test runner. |
| `auto_eval.py` | Project source | Custom benchmark validation, retrieval/generation orchestration, and automatic answer scoring. It is project code, not a third-party package. |

### Training tools versus application dependencies

The model report labels the base model as “Qwen2.5-1.5B-Instruct (4-bit,
Unsloth)” and describes LoRA supervised fine-tuning. Neither `requirements.txt`
nor the repository's Python source contains a training implementation or pins
Hugging Face TRL or Unsloth. Those tools must therefore be described as part of
the reported training provenance, not as confirmed runtime dependencies of this
Streamlit application. The report does not identify a TRL version.

## 3. LLM Fine-Tuning, Training Parameters & Weights

The following specifications are taken from
`data/model_report/metrics.json`. They are report metadata; training execution
cannot be reproduced from this repository alone because the training notebook,
training code, data, and raw trainer logs are not present in the project files.

| Training parameter | Reported value |
|---|---|
| Base model | Qwen2.5-1.5B-Instruct (4-bit, Unsloth) |
| Fine-tuning method | LoRA supervised fine-tuning; loss on responses only |
| Total parameters | 1,562,179,072 |
| Trainable parameters | 18,464,768 |
| Trainable fraction | 1.18% (report value; computed ratio is approximately 1.182%) |
| LoRA rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs / steps | 3 / 84 |
| Learning rate | 0.0002 (2e-4) |
| Per-device batch size | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Optimizer | `adamw_8bit` |
| Maximum sequence length | 2,048 tokens |
| Training / validation examples | 224 / 26 |
| Reported training duration | 3.7 minutes |
| Training GPU | Tesla T4 (reported as 14.6 GB) |
| GPU memory after model load | 1.66 GB |
| Final validation loss | 0.777 |
| Reported perplexity | 2.17 |
| Export | Q4_K_M GGUF, approximately 940 MB |
| Serving label | Ollama (`policy-bot`) |

### Per-epoch loss values

| Epoch | Training loss | Validation loss |
|---:|---:|---:|
| 1 | 0.741 | 0.705 |
| 2 | 0.417 | 0.694 |
| 3 | 0.263 | 0.777 |

Training loss decreases at each reported epoch. Validation loss decreases from
epoch 1 to 2 and then increases from 0.694 to 0.777 at epoch 3. This is
consistent with a possible overfitting signal; the three summary points alone
do not establish a broader generalization result. The repository includes
`data/model_report/loss_curve.png` and the summary JSON, but no raw training log
or notebook to audit how these values were produced.

## 4. Handling Large Files & Context Limits

The system handles documents by extracting page- or file-level text and indexing
smaller overlapping character chunks rather than placing an entire file in the
generation prompt. Each question embeds only the query, retrieves at most the
configured Top-K of 4 chunks, and constructs the Ollama prompt from those
retrieved passages plus the instruction and question.

| Token/context setting | Value | Evidence |
|---|---:|---|
| Ollama `num_ctx` | 4,096 | `generation.py` request options; also model report `ollama_num_ctx`. |
| Model context window | 32,768 | Model report `tokens.model_context_window`; this is report metadata, not the application's active Ollama setting. |
| Retrieval Top-K | 4 | `config.DEFAULT_TOP_K`; also model report. |
| Default chunk size | 800 characters | `config.DEFAULT_CHUNK_SIZE`. |
| Default overlap | 120 characters | `config.DEFAULT_CHUNK_OVERLAP`. |
| Maximum chunk size available in UI | 4,000 characters | `config.MAX_CHUNK_SIZE`. |

Top-K retrieval bounds the number of injected chunks, but the application does
not measure the assembled prompt against a token budget, truncate passages to a
remaining budget, or dynamically reduce Top-K when the context limit is near.
Character count is not token count: tokenization depends on text and model
tokenizer. `tiktoken` measurements in this project are estimates made with
`cl100k_base`, while generation is served by an Ollama model. Consequently, the
design reduces prompt size through chunking and Top-K retrieval, but does not
prove that every possible set of four retrieved chunks fits within 4,096 tokens.

The splitter does not index semantic paragraphs as a unit: the current
`separators=[""]` configuration makes character-level splits. PDF page boundaries
are processed separately, so chunks retain their page metadata. DOCX and TXT
inputs have no reliable page metadata in this implementation.

The report includes training/test token summaries: training input average 398,
training maximum 579, and test average input 177. These are reported data
statistics, not live prompt-budget guarantees for arbitrary uploaded documents.

## 5. Multi-Source Conflict Detection & Resolution

### Generation instruction

`generation.py` instructs Ollama to answer only from the provided passages, cite
claims with `[Doc N]`, state both values when documents disagree, and not choose
one unless a document specifies which takes precedence. This is a generation
instruction intended to encourage conflict-aware answers; it does not guarantee
the model will always follow that instruction.

### Application-layer detector

`conflict_detection.py` compares retrieved chunks pairwise, but skips pairs
belonging to the same document. Within each chunk, it compares sentence-like
statements. A candidate pair needs shared non-stop subject terms and at least
one shared topic category:

| Topic category | Terms used by the detector include |
|---|---|
| Opening hours | `hour`, `hours`, `open`, `opens`, `opening`, `close`, `closes` |
| Service frequency | `every`, `minute`, `minutes`, `hourly`, `frequency`, `interval`, `departs`, `runs`, `operates` |
| Loan duration | `loan`, `loans`, `borrow`, `borrowing`, `period`, `duration` |
| Safety requirement | `must`, `required`, `require`, `wear`, `prohibited` |

It flags a candidate when the statements differ in negation-marker presence,
contain disjoint recognized AM/PM times, or contain different quantities with
the same recognized unit (minutes, hours, days, weeks, months, or years). It
returns a reason and the passages; the Ask UI condenses this to “The sources
disagree on this.” with details in an expander.

This is a narrow heuristic, not a general contradiction solver. It can miss
conflicts expressed without configured topic terms or supported time/quantity
formats, and shared vocabulary can still produce false positives. Conflict
flagging and conflict synthesis by the LLM are separate mechanisms: the detector
does not change the model answer or resolve which source is authoritative.

## 6. LLM Performance Matrix (Base vs. Fine-Tuned)

The table below transcribes the comparison entries in
`data/model_report/metrics.json`. The report identifies its source as
“Documind.ipynb, Google Colab Tesla T4” and its test set as 100 synthetic
questions across legal, travel, telecom, and banking domains, described as never
used in training.

| Metric | Base | Fine-tuned | Direction / interpretation |
|---|---:|---:|---|
| Citation precision | 0.200 | 0.825 | Higher is better. Report caveat: the base model never produced a citation; its 0.20 precision and recall come only from questions where no citation is expected. |
| Citation recall | 0.200 | 0.865 | Higher is better; same base-model caveat applies. |
| Valid citation IDs | 1.000 | 1.000 | Higher is better. This checks ID validity, not semantic support by itself. |
| Conflict handled (both sources cited) | 0.000 | 0.667 | Higher is better. |
| Abstain accuracy | 0.750 | 0.950 | Higher is better. Re-scored with a wider refusal-phrase matcher, according to the report. |
| False-abstain rate | 0.025 | 0.100 | Lower is better. Re-scored with the wider matcher, according to the report. |
| ROUGE-L | 0.457 | 0.589 | Higher is better. |
| Claims supported (word-overlap estimate) | 0.000 | 0.547 | Higher is better; automatic word-overlap estimate, not human judgment. |
| Average output tokens | 41.17 | 36.86 | Descriptive output length; not a quality score. |
| Seconds per answer (Colab T4 GPU) | 1.711 | 2.055 | Lower is faster. These are report values for the stated Colab hardware, not live measurements on the deployment computer. |

The base-versus-fine-tuned latency row shows the fine-tuned answer taking
0.344 seconds longer on average in that reported test, despite other reported
quality metrics improving. It must not be conflated with “Live speed on this
computer” in the UI, which is computed from local SQLite query records and uses
different hardware and a different measurement population.

### Automatic application evaluation

`eval/benchmark.json` contains seven question records: six answerable items
(including one `conflict` question) and one `unanswerable` item. It lists
expected fact groups, candidate source filenames, and a minimum source count
for citation on the conflict question. This local benchmark is distinct from
the 100-question base-versus-fine-tuned test set summarized above.

`auto_eval.py` runs retrieval with the application's configured Top-K and
generates through the same `generate_answer` path when sources meet the
relevance threshold. It validates citation references and computes:

- **Retrieval hit / reciprocal rank:** based on expected filename membership and
  the rank of the first retrieved passage whose filename is expected. Retrieval
  metrics are not applicable to unanswerable questions.
- **Answer correctness:** every expected fact group must match at least one
  pipe-separated alternative after normalization of case, punctuation,
  whitespace, AM/PM times, and number words one through ten.
- **Citation validity:** cited `[Doc N]` references must map to retrieved
  passages; answerable non-refusal answers must include a citation.
- **Citation support:** for each sentence containing a citation, numbers/times/
  money amounts must appear in cited text and at least 60% of content words
  (after stop-word and short-word removal) must overlap. The score is supported
  cited sentences divided by cited sentences.
- **Grounded answer / hallucination rate:** a correct refusal is grounded;
  otherwise the answer must have valid citations and all cited sentences must
  pass the support heuristic. Hallucination rate is one minus the grounded rate.
- **Refusal and conflict handling:** refusal uses the configured
  case-insensitive phrase matcher. Conflict handling requires all expected facts
  and citations to the configured minimum number of distinct files.

Expected source filenames not present in the indexed collection cause the
corresponding question to be skipped with a UI warning. Results are saved to
`data/eval_results.json` with a UTC timestamp. The runner does not record
automatic evaluation results in the ordinary user-query metrics database.

## Repository layout

```text
app.py                         Streamlit page routing and UI
auto_eval.py                   Automatic benchmark validation, runner, scoring, UI
config.py                      Local paths, model, chunk, retrieval, and threshold configuration
ingestion.py                   PDF/DOCX/TXT text extraction and duplicate-aware ingestion
chunking.py                    Page-scoped character chunking and provenance metadata
embeddings.py                  Lazy sentence-transformers model wrapper
vector_store.py                Persistent ChromaDB wrapper
retrieval.py                   Query embedding, Top-K retrieval, relevance threshold
generation.py                  Prompt construction and Ollama chat request
citations.py                   Citation ID validation, display formatting, manual review scoring
conflict_detection.py          Pairwise heuristic disagreement detector
metrics.py                     Token estimates, query schema, SQLite persistence and summaries
evaluation.py                  Legacy chunk-ID retrieval benchmark and labels
schemas.py                     SourcePage, TextChunk, RetrievedChunk data classes
eval/benchmark.json            Automatic evaluation questions and expected facts/sources
data/model_report/metrics.json Reported training configuration and model comparison
data/model_report/loss_curve.png Loss visualization supplied with the report
data/chroma/                   Default persistent ChromaDB directory at runtime
data/metrics.sqlite3           Default SQLite query-measurements database at runtime
data/eval_results.json         Automatic evaluation output after a run
sample_docs/                   Sample text documents and demo questions
tests/                          Unit tests for ingestion, chunking, generation, retrieval,
                                evaluation, metrics, citations, conflicts, and auto_eval
requirements.txt               Python runtime and test dependencies
run_app.bat                     Windows launcher for Ollama and Streamlit
```

## Operational and interpretation notes

1. The application uses local Ollama inference and local ChromaDB persistence as
   configured. Ollama must be installed and the `policy-bot` model available to
   generate answers. Sentence-transformer weights must also be present locally
   for offline embedding; first-time package/model acquisition may require
   network access.
2. `config.GENERATION_TEMPERATURE` is 0.1 and appears in recorded model
   configuration metadata. The current `generation.py` Ollama request explicitly
   sends `options.temperature: 0`; the request value is the effective generation
   setting.
3. Normal query metrics are stored in SQLite as timestamped JSON payloads.
   Recorded fields include retrieval/generation/end-to-end latency, token
   estimates, provider usage when supplied, model name, citations, and
   evaluation-related scores. Evaluation runs from `auto_eval.py` are saved in a
   separate JSON results file.
4. Scores called “supported,” “grounded,” and “hallucination rate” in the
   automatic evaluator are heuristic estimates with the exact limitations
   described above. They are not human annotations or a formal factuality proof.
5. The model report is a fixed comparison artifact. Local live latency and
   automatic evaluation results depend on the current indexed documents,
   configured model, Ollama service, and machine; those values should be
   presented separately from the reported Colab benchmark.
