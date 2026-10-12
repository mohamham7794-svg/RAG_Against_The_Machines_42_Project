"""Retrieval stage: load the persisted index and answer top-k queries.

Implements BM25 lexical retrieval (via ``rank_bm25``), the classic ranking
function recommended by the subject. TF-IDF is easy to swap in instead by
implementing the same ``score`` interface, but BM25 generally handles the
mix of short queries and long code/doc chunks better out of the box.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import sys
from pathlib import Path
from types import TracebackType
from typing import Any, Dict, List, Optional, Type

import numpy as np
from rank_bm25 import BM25Okapi

from src.embedder import Embedder
from src.indexer import EMBEDDINGS_FILE, QUERY_CACHE_FILE, tokenize
from src.models import Chunk, IndexMetadata, MinimalSource

MODES = ("lexical", "hybrid")
# Bump whenever ranking logic changes so old cached answers are ignored.
RANKING_VERSION = "2"
# Reciprocal Rank Fusion: score = sum(1 / (RRF_K + rank)) over the rankings.
RRF_K = 60
HYBRID_POOL = 100
CACHE_MAX_ENTRIES = 20000

"""Identifiers (function/class/constant names) quoted verbatim in a question
are the strongest possible retrieval signal for code questions, but plain
BM25 treats them as just another lowercase token and lets common English
words dominate the ranking. _IDENTIFIER_RE finds candidate identifiers in
the *original*, case-preserving query text; _is_identifier_like keeps only
the ones that actually look like code (snake_case, camelCase/PascalCase,
or ALL_CAPS) rather than ordinary words.
"""
_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_IDENTIFIER_BONUS = 3.0
_IDENTIFIER_POOL_SIZE = 500


def _is_identifier_like(token: str) -> bool:
    """Whether a token looks like a code identifier rather than a word.

    Args:
        token: A word extracted from the raw (non-lowercased) query text.

    Returns:
        True if the token contains an underscore, a lower-to-upper case
        transition (camelCase/PascalCase), or is fully uppercase and at
        least 3 characters long.
    """
    if len(token) < 3:
        return False
    has_underscore = "_" in token
    has_case_transition = any(a.islower() and b.isupper() for a, b in zip(token, token[1:]))
    is_all_caps = token.isupper()
    return has_underscore or has_case_transition or is_all_caps


def _extract_identifiers(query: str) -> List[str]:
    """Extract code-identifier-like tokens from the raw query text.

    Args:
        query: The original, case-preserving query string.

    Returns:
        The distinct identifier-like substrings found in ``query``.
    """
    seen: List[str] = []
    for token in _IDENTIFIER_RE.findall(query):
        if _is_identifier_like(token) and token not in seen:
            seen.append(token)
    return seen


_QUERY_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "how", "what", "why", "when", "where", "who", "which", "does", "do",
    "did", "can", "could", "would", "should", "will", "shall", "to", "of",
    "in", "on", "for", "with", "and", "or", "it", "its", "this", "that",
    "i", "you", "we", "they", "as", "at", "by", "from", "about",
}


def _filter_stopwords(tokens: List[str]) -> List[str]:
    """Drop common function words from query tokens before scoring.

    Args:
        tokens: Tokenized query terms.

    Returns:
        The tokens with stopwords removed, or the original tokens
        unchanged if filtering would remove everything (e.g. a query
        made entirely of stopwords).
    """
    filtered = [t for t in tokens if t not in _QUERY_STOPWORDS]
    return filtered if filtered else tokens


class Retriever:
    """Loads a persisted index and serves top-k search queries.

    Two ranking modes are available:

    * ``lexical`` (default): BM25 plus a bonus for quoted identifiers.
    * ``hybrid`` (bonus): the BM25 ranking and a sentence-embedding ranking
      are merged with Reciprocal Rank Fusion. It needs an index built with
      ``index --semantic``; otherwise it falls back to ``lexical``.

    Query results are cached on disk (bonus), keyed by the index build id,
    the ranking mode, ``k`` and the query text. Rebuilding the index
    invalidates the cache.

    Attributes:
        processed_dir: Directory the index was persisted to by
            :class:`~src.indexer.Indexer`.
        mode: Requested ranking mode.
    """

    def __init__(
        self,
        processed_dir: str = "data/processed",
        mode: str = "lexical",
        use_cache: bool = True,
    ) -> None:
        """Configure the retriever; the index itself is loaded lazily.

        Args:
            processed_dir: Directory holding the persisted index.
            mode: ``"lexical"`` or ``"hybrid"``.
            use_cache: Read and write the on-disk query cache.

        Raises:
            ValueError: If ``mode`` is not one of :data:`MODES`.
        """
        if mode not in MODES:
            raise ValueError(
                f"Unknown mode '{mode}'. Choose one of: {', '.join(MODES)}."
            )
        self.processed_dir = Path(processed_dir)
        self.mode = mode
        self.use_cache = use_cache
        self._chunks: Optional[List[Chunk]] = None
        self._bm25: Optional[BM25Okapi] = None
        self._embeddings: Optional[np.ndarray] = None
        self._embedder: Optional[Embedder] = None
        self._metadata: Optional[IndexMetadata] = None
        self._effective_mode: Optional[str] = None
        self._text_by_span: Optional[Dict[Any, str]] = None
        self._cache: Optional[Dict[str, List[List[Any]]]] = None
        self._cache_dirty = False

    def __enter__(self) -> "Retriever":
        """Enter the ``with`` block.

        Returns:
            This retriever.
        """
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        """Leave the ``with`` block and save the query cache.

        Args:
            exc_type: Exception type raised in the block, if any.
            exc: Exception instance raised in the block, if any.
            tb: Traceback of that exception, if any.
        """
        self.close()

    def close(self) -> None:
        """Write pending cache entries to disk."""
        if not self._cache_dirty or self._cache is None:
            return
        path = self.processed_dir / QUERY_CACHE_FILE
        tmp = path.with_suffix(".tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(self._cache, fh)
            os.replace(tmp, path)
            self._cache_dirty = False
        except OSError as exc:
            print(f"Could not save query cache: {exc}", file=sys.stderr)

    def _load_metadata(self) -> IndexMetadata:
        """Read the small metadata file (cheap, no model or BM25 build).

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        if self._metadata is None:
            path = self.processed_dir / "metadata.json"
            if not path.exists():
                raise FileNotFoundError(
                    f"No index found under {self.processed_dir}/. "
                    "Run `index` before searching."
                )
            self._metadata = IndexMetadata.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        return self._metadata

    def _resolve_mode(self) -> str:
        """The mode actually used (hybrid needs a semantic index)."""
        if self._effective_mode is None:
            self._effective_mode = "lexical"
            if self.mode == "hybrid":
                available = (
                    self._load_metadata().semantic
                    and (self.processed_dir / EMBEDDINGS_FILE).exists()
                )
                if available:
                    self._effective_mode = "hybrid"
                else:
                    print(
                        "No semantic index found (build it with "
                        "`index --semantic`); using lexical search.",
                        file=sys.stderr,
                    )
        return self._effective_mode

    def _cache_key(self, query: str, k: int) -> str:
        """Build the cache key of a query.

        The key depends on the ranking version, the index build, the
        effective mode, ``k`` and the query text, so any change to one of
        them can never return a stale answer.

        Args:
            query: The query text.
            k: Number of results requested.

        Returns:
            A hex digest identifying this exact search.
        """
        raw = "|".join(
            [
                RANKING_VERSION,
                self._load_metadata().index_id,
                self._resolve_mode(),
                str(k),
                query.strip(),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _load_cache(self) -> Dict[str, List[List[Any]]]:
        """Load the on-disk query cache once.

        Returns:
            The cache mapping; empty if the file is missing or corrupt.
        """
        if self._cache is None:
            self._cache = {}
            path = self.processed_dir / QUERY_CACHE_FILE
            if path.exists():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        self._cache = loaded
                except (OSError, ValueError):
                    self._cache = {}  # corrupt cache: start again
        return self._cache

    def _cache_enabled(self) -> bool:
        """Whether this search may read and write the query cache.

        Returns:
            True if caching was requested and the index has a build id.
        """
        return self.use_cache and bool(self._load_metadata().index_id)

    def _ensure_loaded(self) -> None:
        """Lazily load the chunks and BM25 model from disk.

        Raises:
            FileNotFoundError: If the index has not been built yet.
        """
        if self._chunks is not None and self._bm25 is not None:
            return

        chunks_path = self.processed_dir / "chunks.pkl"
        corpus_path = self.processed_dir / "tokenized_corpus.pkl"
        if not chunks_path.exists() or not corpus_path.exists():
            raise FileNotFoundError(
                f"No index found under {self.processed_dir}/. "
                "Run `index` before searching."
            )

        with chunks_path.open("rb") as fh:
            self._chunks = pickle.load(fh)
        with corpus_path.open("rb") as fh:
            tokenized_corpus = pickle.load(fh)

        if tokenized_corpus:
            self._bm25 = BM25Okapi(tokenized_corpus)
        else:
            self._bm25 = None

        if self._resolve_mode() == "hybrid":
            self._embeddings = np.load(self.processed_dir / EMBEDDINGS_FILE)

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Return the top-k most relevant source locations for a query.

        Args:
            query: The natural-language (or code) query text.
            k: Number of results to return. Non-positive values yield an
                empty list.

        Returns:
            A list of at most ``k`` :class:`~src.models.MinimalSource`,
            ranked from most to least relevant.
        """
        if k <= 0 or not query or not query.strip():
            return []

        key = ""
        if self._cache_enabled():
            key = self._cache_key(query, k)
            hit = self._load_cache().get(key)
            if hit is not None:
                return [
                    MinimalSource(
                        file_path=path,
                        first_character_index=first,
                        last_character_index=last,
                    )
                    for path, first, last in hit
                ]

        results = self._search_uncached(query, k)

        if key:
            cache = self._load_cache()
            if len(cache) < CACHE_MAX_ENTRIES:
                cache[key] = [
                    [s.file_path, s.first_character_index,
                     s.last_character_index]
                    for s in results
                ]
                self._cache_dirty = True
        return results

    def _search_uncached(self, query: str, k: int) -> List[MinimalSource]:
        """Rank the chunks for a query without consulting the cache.

        Args:
            query: The query text.
            k: Number of results to return.

        Returns:
            At most ``k`` sources, best first.
        """
        self._ensure_loaded()
        if not self._chunks or self._bm25 is None:
            return []

        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        query_tokens = _filter_stopwords(query_tokens)

        scores = np.asarray(self._bm25.get_scores(query_tokens))

        """Boost chunks that contain an exact, case-sensitive identifier
        quoted in the query (e.g. a class or function name). This is
        re-scored only within the top BM25 candidates for speed, and only
        when the query actually contains identifier-like tokens, so plain
        natural-language questions are unaffected.
        """
        identifiers = _extract_identifiers(query)
        if identifiers:
            pool = np.argsort(-scores, kind="stable")[:_IDENTIFIER_POOL_SIZE]
            for idx in pool:
                text = self._chunks[idx].text
                bonus = sum(
                    _IDENTIFIER_BONUS for ident in identifiers if ident in text
                )
                if bonus:
                    scores[idx] += bonus

        lexical_order = np.argsort(-scores, kind="stable")

        if self._resolve_mode() == "hybrid":
            ranked = self._fuse(query, lexical_order, scores)
        else:
            ranked = [int(i) for i in lexical_order[: max(k * 3, k)]
                      if scores[i] > 0]

        results = [self._to_source(idx) for idx in ranked[:k]]

        """Fall back to top scores even if <=0, so we always try to return k
        results when the corpus is non-empty (better than an empty list).
        """
        if not results:
            results = [self._to_source(int(i)) for i in lexical_order[:k]]
        return results

    def _fuse(
        self, query: str, lexical_order: np.ndarray, scores: np.ndarray
    ) -> List[int]:
        """Merge the BM25 and embedding rankings with RRF."""
        if self._embedder is None:
            self._embedder = Embedder()
        assert self._embeddings is not None
        query_vec = self._embedder.encode([query])[0]
        semantic_scores = self._embeddings @ query_vec
        semantic_order = np.argsort(-semantic_scores, kind="stable")

        fused: Dict[int, float] = {}
        lexical_hits = (int(i) for i in lexical_order[:HYBRID_POOL]
                        if scores[i] > 0)
        for rank, idx in enumerate(lexical_hits):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(semantic_order[:HYBRID_POOL]):
            i = int(idx)
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + rank + 1)
        return sorted(fused, key=lambda i: fused[i], reverse=True)

    def _to_source(self, idx: int) -> MinimalSource:
        """Convert a chunk position into its public source location.

        Args:
            idx: Position of the chunk in the index.

        Returns:
            The chunk's file path and character range.
        """
        assert self._chunks is not None
        chunk = self._chunks[idx]
        return MinimalSource(
            file_path=chunk.file_path,
            first_character_index=chunk.first_character_index,
            last_character_index=chunk.last_character_index,
        )

    def get_text(self, source: MinimalSource) -> str:
        """Return the raw chunk text backing a given source location.

        Used by the answer-generation stage to build the model context.

        Args:
            source: A source location previously returned by :meth:`search`.

        Returns:
            The chunk text, or an empty string if it can no longer be found
            (e.g. the index was rebuilt with different chunk boundaries).
        """
        self._ensure_loaded()
        if not self._chunks:
            return ""
        if self._text_by_span is None:
            self._text_by_span = {
                (c.file_path, c.first_character_index,
                 c.last_character_index): c.text
                for c in self._chunks
            }
        return self._text_by_span.get(
            (source.file_path, source.first_character_index,
             source.last_character_index),
            "",
        )
