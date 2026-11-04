"""Application configuration and paths."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("DOCUMIND_DATA_DIR", BASE_DIR / "data"))
CHROMA_DIR = DATA_DIR / "chroma"
METRICS_DB = DATA_DIR / "metrics.sqlite3"
EVALUATION_DATASET_PATH = BASE_DIR / "evaluation_dataset.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "policy-bot"
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120
DEFAULT_TOP_K = 4
MAX_CHUNK_SIZE = 4000
MIN_RELEVANCE_DISTANCE = 0.7
GENERATION_TEMPERATURE = 0.1
