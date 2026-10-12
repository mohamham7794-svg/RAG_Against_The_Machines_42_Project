"""Chunking strategies for the RAG indexer.

Two distinct strategies are implemented, as required by the subject:

* :func:`chunk_python_code` splits ``.py`` files along top-level function
  and class boundaries (using the ``ast`` module), falling back to a plain
  sliding window when a file fails to parse or a single definition is
  larger than ``max_chunk_size``.
* :func:`chunk_markdown_text` splits ``.md`` / ``.txt`` files along Markdown
  headers and paragraph breaks, again falling back to a sliding window when
  a section is too large.

Both strategies guarantee that no produced chunk exceeds ``max_chunk_size``
characters, as required by the moulinette (``max_context_length``).
"""

from __future__ import annotations

import ast
from typing import List, Tuple

Span = Tuple[int, int]


def _sliding_window(text: str, max_chunk_size: int, overlap: int = 100) -> List[Span]:
    """Split ``text`` into fixed-size, slightly overlapping windows.

    Used as a fallback whenever a semantic unit (a function, a Markdown
    section, ...) is itself larger than ``max_chunk_size``.

    Args:
        text: The text to split.
        max_chunk_size: Maximum number of characters per window.
        overlap: Number of characters shared between consecutive windows,
            so a match near a window boundary is not lost.

    Returns:
        A list of ``(first_character_index, last_character_index)`` spans,
        end-exclusive, covering the whole text.
    """
    if not text:
        return []
    if max_chunk_size <= 0:
        return [(0, len(text))]

    spans: List[Span] = []
    start = 0
    step = max(1, max_chunk_size - overlap)
    text_len = len(text)
    while start < text_len:
        end = min(start + max_chunk_size, text_len)
        spans.append((start, end))
        if end >= text_len:
            break
        start += step
    return spans


def chunk_python_code(source: str, max_chunk_size: int = 2000) -> List[Span]:
    """Chunk a Python source file along function/class boundaries.

    Strategy: parse the module with ``ast``, and cut the file at the start
    of every top-level ``def``/``class``/``async def`` statement. This keeps
    a whole function (its signature, docstring and body) inside a single
    chunk whenever it fits in ``max_chunk_size``, which lexical retrieval
    methods rely on to match identifiers and comments together.

    If the file cannot be parsed (syntax error) or a single top-level block
    is larger than ``max_chunk_size``, that block is further split with a
    sliding window fallback.

    Args:
        source: The full text content of the ``.py`` file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of ``(first_character_index, last_character_index)`` spans,
        end-exclusive, covering the whole source.
    """
    if not source:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return _sliding_window(source, max_chunk_size)

    lines = source.splitlines(keepends=True)
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))

    def line_to_offset(lineno: int) -> int:
        idx = lineno - 1
        if idx < 0:
            return 0
        if idx >= len(line_offsets):
            return len(source)
        return line_offsets[idx]

    top_level_starts: List[int] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            top_level_starts.append(line_to_offset(node.lineno))

    if not top_level_starts:
        return _sliding_window(source, max_chunk_size)

    top_level_starts = sorted(set(top_level_starts))
    boundaries = [0] + top_level_starts + [len(source)]
    boundaries = sorted(set(boundaries))

    spans: List[Span] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        block_len = end - start
        if block_len <= max_chunk_size:
            spans.append((start, end))
        else:
            fallback = _sliding_window(source[start:end], max_chunk_size)
            spans.extend((s + start, e + start) for s, e in fallback)
    return spans


def chunk_markdown_text(source: str, max_chunk_size: int = 2000) -> List[Span]:
    """Chunk a Markdown or plain-text file along headers and paragraphs.

    Strategy: first split on Markdown headers (lines starting with ``#``),
    which usually delimit a self-contained topic. Each header section is
    then further split on blank-line-separated paragraphs if it still
    exceeds ``max_chunk_size``, and finally with a sliding window as a last
    resort.

    Args:
        source: The full text content of the ``.md``/``.txt`` file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of ``(first_character_index, last_character_index)`` spans,
        end-exclusive, covering the whole source.
    """
    if not source:
        return []

    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    header_starts: List[int] = [0]
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            header_starts.append(offsets[i])

    header_starts = sorted(set(header_starts))
    boundaries = header_starts + [len(source)]
    boundaries = sorted(set(boundaries))

    sections: List[Span] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end > start:
            sections.append((start, end))
    if not sections:
        sections = [(0, len(source))]

    spans: List[Span] = []
    for start, end in sections:
        block = source[start:end]
        if len(block) <= max_chunk_size:
            spans.append((start, end))
            continue

        paragraphs = _split_paragraphs(block)
        if len(paragraphs) <= 1:
            spans.extend((s + start, e + start) for s, e in _sliding_window(block, max_chunk_size))
            continue

        spans.extend(_pack_spans(block, paragraphs, max_chunk_size, base_offset=start))

    return spans


def _pack_spans(
    block: str, atomic_spans: List[Span], max_chunk_size: int, base_offset: int = 0
) -> List[Span]:
    """Greedily group atomic spans into chunks bounded by ``max_chunk_size``.

    Consecutive atomic spans (e.g. paragraphs) are merged together as long
    as the running group stays within ``max_chunk_size``. An atomic span
    that alone exceeds ``max_chunk_size`` is further split with a sliding
    window.

    Args:
        block: The text the atomic spans are relative to.
        atomic_spans: Non-overlapping ``(start, end)`` spans covering
            ``block``, in order.
        max_chunk_size: Maximum number of characters per output chunk.
        base_offset: Offset to add to every produced span, so callers can
            work in the coordinate system of a larger enclosing text.

    Returns:
        A list of ``(start, end)`` spans (offset by ``base_offset``)
        covering the whole ``block``, each at most ``max_chunk_size`` long.
    """
    spans: List[Span] = []
    group_start: int | None = None
    group_end: int | None = None

    def flush() -> None:
        if group_start is not None and group_end is not None and group_end > group_start:
            spans.append((group_start + base_offset, group_end + base_offset))

    for p_start, p_end in atomic_spans:
        p_len = p_end - p_start
        if p_len > max_chunk_size:
            flush()
            group_start = None
            group_end = None
            spans.extend(
                (s + p_start + base_offset, e + p_start + base_offset)
                for s, e in _sliding_window(block[p_start:p_end], max_chunk_size)
            )
            continue

        if group_start is None:
            group_start, group_end = p_start, p_end
            continue

        if p_end - group_start <= max_chunk_size:
            group_end = p_end
        else:
            flush()
            group_start, group_end = p_start, p_end

    flush()
    return spans


def _split_paragraphs(text: str) -> List[Span]:
    """Split text into paragraph spans separated by blank lines.

    Args:
        text: The text to split.

    Returns:
        A list of ``(start, end)`` spans, one per paragraph, covering the
        whole text (including separators, attached to the following
        paragraph) so that concatenation reconstructs the original text.
    """
    if not text:
        return []
    spans: List[Span] = []
    start = 0
    i = 0
    n = len(text)
    while i < n - 1:
        if text[i] == "\n" and text[i + 1] == "\n":
            end = i + 2
            spans.append((start, end))
            start = end
            i = end
        else:
            i += 1
    if start < n:
        spans.append((start, n))
    return spans


def chunk_file(path: str, source: str, max_chunk_size: int = 2000) -> List[Span]:
    """Dispatch to the right chunking strategy based on file extension.

    Args:
        path: Path to the file (used only to pick a strategy by extension).
        source: The full text content of the file.
        max_chunk_size: Maximum number of characters per chunk.

    Returns:
        A list of ``(first_character_index, last_character_index)`` spans.
    """
    if path.endswith(".py"):
        return chunk_python_code(source, max_chunk_size)
    return chunk_markdown_text(source, max_chunk_size)
