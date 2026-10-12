"""Pydantic data models exchanged between the RAG pipeline stages.

These models mirror the ones defined in the project subject. They are the
contract between indexing, retrieval, generation and evaluation: every
stage reads and writes objects that validate against these classes.
"""

from __future__ import annotations

import uuid
from typing import List, Optional, Union

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """A single source location inside the ingested corpus.

    Attributes:
        file_path: Path to the source file, relative to the project root and
            written exactly as in the ingested corpus (e.g.
            ``data/raw/vllm-0.10.1/docs/features/lora.md``). Compared
            verbatim by the grader.
        first_character_index: Starting character position of the chunk in
            the source file.
        last_character_index: Ending character position of the chunk in the
            source file.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question without an answer yet.

    Attributes:
        question_id: Unique identifier for the question.
        question: The natural-language question text.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question paired with its ground-truth sources and answer.

    Attributes:
        sources: The ground-truth source locations that answer the question.
        answer: The reference natural-language answer.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of RAG questions, answered or not.

    Attributes:
        rag_questions: The list of questions in the dataset.
    """

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """The retrieved sources for a single question.

    Attributes:
        question_id: Identifier of the question this result belongs to.
        question: The natural-language question text.
        retrieved_sources: The ranked list of retrieved source locations.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search results augmented with a generated answer.

    Attributes:
        answer: The natural-language answer produced by the model.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Top-level output of a ``search`` / ``search_dataset`` run.

    Attributes:
        search_results: One entry per question with its retrieved sources.
        k: Number of results requested per question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Top-level output of an ``answer`` / ``answer_dataset`` run.

    Attributes:
        search_results: One entry per question with sources and answer.
        k: Number of results requested per question.
    """

    search_results: List[MinimalAnswer]
    k: int


class Chunk(BaseModel):
    """An indexed chunk of text extracted from a source file.

    This is an internal model (not part of the subject's minimal set) used
    to persist the index built at indexing time.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        file_path: Path to the source file, relative to the project root.
        first_character_index: Starting character position in the file.
        last_character_index: Ending character position in the file.
        text: The raw text content of the chunk.
        chunk_type: Either ``"code"`` or ``"text"`` depending on the
            chunking strategy that produced it.
    """

    chunk_id: str
    file_path: str
    first_character_index: int
    last_character_index: int
    text: str
    chunk_type: str


class IndexMetadata(BaseModel):
    """Metadata persisted alongside the index for reproducibility.

    Attributes:
        max_chunk_size: The ``--max_chunk_size`` used at indexing time.
        num_chunks: Total number of indexed chunks.
        num_files: Total number of indexed files.
        method: The lexical retrieval method used (``"bm25"`` or
            ``"tfidf"``).
    """

    max_chunk_size: int
    num_chunks: int
    num_files: int
    method: str
    index_id: str = ""
    semantic: bool = False
    embedding_model: str = ""


class EvaluationResult(BaseModel):
    """Recall@k evaluation summary for a dataset run.

    Attributes:
        questions_evaluated: Number of questions scored.
        recall_at_k: Mapping of k values to recall scores.
    """

    questions_evaluated: int
    recall_at_k: dict[int, float]
    student_data_valid: Optional[bool] = None
