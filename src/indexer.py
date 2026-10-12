"""Indexing stage: walk the corpus, chunk every file and persist an index.

The index is a simple pickle of a list of :class:`~src.models.Chunk`
objects plus a fitted BM25/TF-IDF model, stored under ``data/processed/``.
It is deliberately simple (no external vector database) since the lexical
retrieval methods required by the subject (BM25 / TF-IDF) only need the
tokenized corpus in memory.
"""

from __future__ import annotations

import pickle
import re
import time
import uuid
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from src.chunking import chunk_file
from src.embedder import Embedder
from src.models import Chunk, IndexMetadata

QUERY_CACHE_FILE = "query_cache.json"
EMBEDDINGS_FILE = "embeddings.npy"

# File extensions considered part of the corpus.
INDEXABLE_EXTENSIONS = {".py", ".md", ".rst", ".txt"}

# Directories that never carry useful signal and slow down indexing.
IGNORED_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".idea",
    ".vscode",
    "dist",
    "build",
    "egg-info",
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+(?:\.[0-9]+)?")
# Word pieces inside an identifier: HTTPServer -> HTTP, Server;
# max_num_seqs -> max, num, seqs.
_PIECE_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+")


def tokenize(text: str) -> List[str]:
    """Tokenize text for the lexical index.

    Every identifier is kept whole (lowercased) and, when it is made of
    several words (``snake_case`` or ``camelCase``), its word pieces are
    emitted too. A question saying "max num seqs" can therefore match the
    identifier ``max_num_seqs`` and vice versa.

    Args:
        text: The text to tokenize.

    Returns:
        A list of lowercase tokens.
    """
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text):
        tokens.append(raw.lower())
        if not raw[0].isalpha() and raw[0] != "_":
            continue
        pieces = [p.lower() for p in _PIECE_RE.findall(raw)]
        if len(pieces) > 1:
            tokens.extend(p for p in pieces if len(p) > 1)
    return tokens


def iter_corpus_files(raw_dir: Path) -> List[Path]:
    """Enumerate the files that should be indexed under ``raw_dir``.

    Args:
        raw_dir: Root directory of the raw corpus (typically ``data/raw``).

    Returns:
        A sorted list of file paths with an indexable extension, skipping
        ignored directories.
    """
    files: List[Path] = []
    if not raw_dir.exists():
        return files
    for path in raw_dir.rglob("*"):
        if not path.is_file():
            continue
        if any(
            part in IGNORED_DIR_NAMES
            for part in path.relative_to(raw_dir).parts
        ):
            continue
        if path.suffix in INDEXABLE_EXTENSIONS:
            files.append(path)
    return sorted(files)


def _read_text(path: Path) -> str:
    """Read a file as UTF-8 text, tolerating decoding errors.

    Args:
        path: Path to the file to read.

    Returns:
        The decoded text content, or an empty string if the file cannot be
        read at all.
    """
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""


class Indexer:
    """Builds and persists the searchable index for a corpus.

    Attributes:
        raw_dir: Root directory of the raw corpus to ingest.
        processed_dir: Directory where the index is persisted.
        method: Lexical retrieval method recorded in the metadata.
        semantic: Also compute and store sentence embeddings (bonus).
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        method: str = "bm25",
        semantic: bool = False,
    ) -> None:
        """Store the indexing options.

        Args:
            raw_dir: Root directory of the raw corpus to ingest.
            processed_dir: Directory where the index is persisted.
            method: Lexical retrieval method recorded in the metadata.
            semantic: Also compute sentence embeddings (bonus).
        """
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.method = method
        self.semantic = semantic

    def _path_tokens(self, path: Path) -> List[str]:
        """Tokens of a file's location below ``raw_dir``.

        A question such as "CLI command for benchmarking latency" often
        names the file (``docs/cli/bench/latency.md``) rather than quoting
        its text, so the path is indexed together with every chunk.
        """
        try:
            rel = path.relative_to(self.raw_dir)
        except ValueError:
            rel = path
        return tokenize(rel.with_suffix("").as_posix().replace("/", " "))

    def build(self, max_chunk_size: int = 2000) -> IndexMetadata:
        """Chunk every file in the corpus and persist the resulting index.

        Args:
            max_chunk_size: Maximum number of characters per chunk. Also
                forwarded to the chunking strategies.

        Returns:
            Metadata describing the freshly built index.
        """
        start_time = time.time()
        files = iter_corpus_files(self.raw_dir)

        chunks: List[Chunk] = []
        chunk_path_tokens: List[List[str]] = []
        for path in tqdm(files, desc="Chunking", unit="file"):
            text = _read_text(path)
            if not text:
                continue
            rel_path = path.as_posix()
            path_tokens = self._path_tokens(path)
            spans = chunk_file(rel_path, text, max_chunk_size)
            chunk_type = "code" if path.suffix == ".py" else "text"
            for first, last in spans:
                snippet = text[first:last]
                if not snippet.strip():
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        file_path=rel_path,
                        first_character_index=first,
                        last_character_index=last,
                        text=snippet,
                        chunk_type=chunk_type,
                    )
                )
                chunk_path_tokens.append(path_tokens)

        tokenized_corpus = [
            path_tokens + tokenize(chunk.text)
            for chunk, path_tokens in tqdm(
                zip(chunks, chunk_path_tokens),
                total=len(chunks),
                desc="Tokenizing",
                unit="chunk",
            )
        ]

        self.processed_dir.mkdir(parents=True, exist_ok=True)
        for stale in (QUERY_CACHE_FILE, EMBEDDINGS_FILE):
            (self.processed_dir / stale).unlink(missing_ok=True)
        with (self.processed_dir / "chunks.pkl").open("wb") as fh:
            pickle.dump(chunks, fh)
        with (self.processed_dir / "tokenized_corpus.pkl").open("wb") as fh:
            pickle.dump(tokenized_corpus, fh)

        embedder = Embedder()
        semantic_ok = False
        if self.semantic:
            semantic_ok = self._build_embeddings(chunks, embedder)

        metadata = IndexMetadata(
            max_chunk_size=max_chunk_size,
            num_chunks=len(chunks),
            num_files=len(files),
            method=self.method,
            index_id=str(uuid.uuid4()),
            semantic=semantic_ok,
            embedding_model=embedder.model_name if semantic_ok else "",
        )
        with (self.processed_dir / "metadata.json").open(
            "w", encoding="utf-8"
        ) as fh:
            fh.write(metadata.model_dump_json(indent=2))

        elapsed = time.time() - start_time
        print(
            f"Ingestion complete! Indexed {len(chunks)} chunks from "
            f"{len(files)} files under {self.processed_dir}/ "
            f"in {elapsed:.1f}s"
        )
        return metadata

    def _build_embeddings(
        self, chunks: List[Chunk], embedder: Embedder
    ) -> bool:
        """Embed all chunks and store them next to the lexical index.

        Returns:
            True on success. On failure (model unavailable) a message is
            printed and the lexical index is still usable.
        """
        texts = [f"{c.file_path}\n{c.text}" for c in chunks]
        try:
            vectors = embedder.encode(texts, show_progress=True)
        except RuntimeError as exc:
            print(f"Semantic index skipped: {exc}")
            return False
        np.save(self.processed_dir / EMBEDDINGS_FILE, vectors)
        return True
