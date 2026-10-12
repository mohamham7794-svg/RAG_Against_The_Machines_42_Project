"""Command-Line Interface for the RAG pipeline, built with Python Fire.

Every command is invoked as ``uv run python -m src <command> [options]``.
All input/output paths are configurable CLI arguments; nothing is
hard-coded. Degenerate inputs (empty query, k=0, missing files, malformed
JSON) are handled gracefully: the CLI never crashes with an unhandled
traceback.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

from src.evaluator import evaluate as run_evaluate
from src.generator import AnswerGenerator
from src.indexer import Indexer
from src.models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSearchResults,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.retriever import Retriever

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_PROCESSED_DIR = "data/processed"
DEFAULT_K = 5
EVAL_KS = [1, 3, 5, 10]


def _load_json(path: str) -> Any:
    """Load and parse a JSON file, raising a clear error on failure.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"No such file: {path}")
    try:
        with file_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in {path}: {exc}") from exc


def _load_dataset(dataset_path: str) -> RagDataset:
    """Load a question dataset, accepting either raw-list or wrapped form.

    Args:
        dataset_path: Path to the dataset JSON file. Accepts either
            ``{"rag_questions": [...]}`` (the :class:`RagDataset` shape) or
            a bare JSON list of question objects.

    Returns:
        The parsed :class:`~src.models.RagDataset`.
    """
    raw = _load_json(dataset_path)
    if isinstance(raw, list):
        raw = {"rag_questions": raw}
    return RagDataset.model_validate(raw)


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    """Write a JSON payload to disk, creating parent directories as needed.

    Args:
        path: Destination file path.
        payload: JSON-serializable payload (typically a
            ``model_dump(mode="json")`` result).
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


class RagCLI:
    """Retrieval-Augmented Generation command-line interface."""

    def index(
        self,
        raw_dir: str = DEFAULT_RAW_DIR,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        max_chunk_size: int = 2000,
        method: str = "bm25",
        semantic: bool = False,
    ) -> None:
        """Ingest ``raw_dir`` and build the searchable index.

        Args:
            raw_dir: Root directory of the raw corpus to ingest.
            processed_dir: Directory where the index is persisted.
            max_chunk_size: Maximum number of characters per chunk. Must
                not exceed the moulinette's ``max_context_length`` (2000).
            method: Lexical retrieval method to record in the metadata
                (``"bm25"`` is the one actually used by the retriever).
            semantic: Also embed every chunk with a small CPU sentence
                model so ``--mode hybrid`` can be used when searching
                (slower: embedding all chunks takes a few minutes).
        """
        if max_chunk_size <= 0:
            print("max_chunk_size must be positive; nothing to index.")
            return
        indexer = Indexer(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            method=method,
            semantic=semantic,
        )
        try:
            indexer.build(max_chunk_size=max_chunk_size)
        except Exception as exc:  # noqa: BLE001 - CLI must never crash
            print(f"Indexing failed: {exc}")

    def search(
        self,
        query: str,
        k: int = DEFAULT_K,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        mode: str = "lexical",
        use_cache: bool = True,
    ) -> None:
        """Return the top-k sources for a single query.

        Args:
            query: The natural-language (or code) query text.
            k: Number of results to return.
            processed_dir: Directory holding the persisted index.
            mode: ``"lexical"`` (BM25) or ``"hybrid"`` (BM25 + embeddings,
                needs ``index --semantic``).
            use_cache: Reuse / store results in the on-disk query cache.
        """
        if not isinstance(query, str) or not query.strip():
            print("Empty or invalid query; returning no results.")
            return
        if k <= 0:
            print("k must be positive; returning no results.")
            return

        try:
            with Retriever(processed_dir, mode, use_cache) as retriever:
                results = retriever.search(query, k=k)
        except FileNotFoundError as exc:
            print(exc)
            return
        except Exception as exc:  # noqa: BLE001 - CLI must never crash
            print(f"Search failed: {exc}")
            return

        if not results:
            print("No results found.")
            return
        for source in results:
            print(
                f"{source.file_path} "
                f"[{source.first_character_index}:{source.last_character_index}]"
            )

    def search_dataset(
        self,
        dataset_path: str,
        k: int = DEFAULT_K,
        save_directory: str = "data/output/search_results",
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        mode: str = "lexical",
        use_cache: bool = True,
    ) -> None:
        """Run search over a whole dataset and write a StudentSearchResults file.

        Args:
            dataset_path: Path to the input question dataset JSON file.
            k: Number of results to return per question.
            save_directory: Directory the output JSON is written into,
                under the same filename as ``dataset_path``.
            processed_dir: Directory holding the persisted index.
            mode: ``"lexical"`` or ``"hybrid"`` (see :meth:`search`).
            use_cache: Reuse / store results in the on-disk query cache.
        """
        try:
            dataset = _load_dataset(dataset_path)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return

        try:
            retriever = Retriever(processed_dir, mode, use_cache)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load index: {exc}")
            return

        results: List[MinimalSearchResults] = []
        try:
            for question in tqdm(
                dataset.rag_questions, desc="Searching", unit="question"
            ):
                try:
                    sources = retriever.search(question.question, k=k)
                except FileNotFoundError as exc:
                    print(exc)
                    return
                except Exception as exc:  # noqa: BLE001
                    print(
                        "Search failed for question "
                        f"{question.question_id}: {exc}"
                    )
                    sources = []
                results.append(
                    MinimalSearchResults(
                        question_id=question.question_id,
                        question=question.question,
                        retrieved_sources=sources,
                    )
                )
        finally:
            retriever.close()

        output = StudentSearchResults(search_results=results, k=k)
        out_path = Path(save_directory) / Path(dataset_path).name
        _write_json(str(out_path), output.model_dump(mode="json"))
        print(f"Saved student_search_results to {out_path}")

    def answer(
        self,
        query: str,
        k: int = DEFAULT_K,
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        model_name: str = "Qwen/Qwen3-0.6B",
        mode: str = "lexical",
        use_cache: bool = True,
    ) -> None:
        """Answer a single query using the retrieved context.

        Args:
            query: The natural-language question.
            k: Number of sources to retrieve and pass as context.
            processed_dir: Directory holding the persisted index.
            model_name: HuggingFace model identifier used for generation.
            mode: ``"lexical"`` or ``"hybrid"`` (see :meth:`search`).
            use_cache: Reuse / store results in the on-disk query cache.
        """
        if not isinstance(query, str) or not query.strip():
            print("Empty or invalid query; nothing to answer.")
            return
        if k <= 0:
            print("k must be positive; nothing to retrieve.")
            return

        try:
            with Retriever(processed_dir, mode, use_cache) as retriever:
                sources = retriever.search(query, k=k)
                texts = [retriever.get_text(source) for source in sources]
        except FileNotFoundError as exc:
            print(exc)
            return
        except Exception as exc:  # noqa: BLE001
            print(f"Search failed: {exc}")
            return

        generator = AnswerGenerator(model_name=model_name)
        answer_text = generator.generate(query, sources, texts)
        print(answer_text)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer",
        processed_dir: str = DEFAULT_PROCESSED_DIR,
        model_name: str = "Qwen/Qwen3-0.6B",
    ) -> None:
        """Generate answers for a dataset of already-retrieved search results.

        Args:
            student_search_results_path: Path to a ``StudentSearchResults``
                JSON file (typically produced by ``search_dataset``).
            save_directory: Directory the output JSON is written into,
                under the same filename as the input file.
            processed_dir: Directory holding the persisted index (used to
                recover chunk text for the retrieved sources).
            model_name: HuggingFace model identifier used for generation.
        """
        try:
            raw = _load_json(student_search_results_path)
            student_results = StudentSearchResults.model_validate(raw)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return

        try:
            retriever = Retriever(processed_dir=processed_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not load index: {exc}")
            return

        generator = AnswerGenerator(model_name=model_name)

        answers: List[MinimalAnswer] = []
        total = len(student_results.search_results)
        print(f"Loaded {total} questions ...")
        for result in tqdm(student_results.search_results, desc="Answering", unit="question"):
            texts = [retriever.get_text(source) for source in result.retrieved_sources]
            answer_text = generator.generate(result.question, result.retrieved_sources, texts)
            answers.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer_text,
                )
            )
        print(f"Processed {len(answers)} of {total} questions")

        output = StudentSearchResultsAndAnswer(search_results=answers, k=student_results.k)
        out_path = Path(save_directory) / Path(student_search_results_path).name
        _write_json(str(out_path), output.model_dump(mode="json"))
        print(f"Saved student_search_results_and_answer to {out_path}")

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> None:
        """Report your own recall@k against a ground-truth dataset.

        This is for local iteration only: the official recall@k used
        during the defense is computed by the moulinette executable, never
        by this command.

        Args:
            student_search_results_path: Path to a ``StudentSearchResults``
                (or ``StudentSearchResultsAndAnswer``) JSON file.
            dataset_path: Path to the ground-truth ``AnsweredQuestions``
                dataset JSON file.
        """
        try:
            student_raw = _load_json(student_search_results_path)
            _load_json(dataset_path)  # validate the reference file exists and parses
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return

        try:
            search_results = student_raw.get("search_results", [])
            student_results = [
                MinimalSearchResults.model_validate(
                    {
                        "question_id": r["question_id"],
                        "question": r["question"],
                        "retrieved_sources": r["retrieved_sources"],
                    }
                )
                for r in search_results
            ]
        except (KeyError, TypeError) as exc:
            print(f"Malformed student results: {exc}")
            return

        try:
            reference_dataset = _load_dataset(dataset_path)
        except (FileNotFoundError, ValueError) as exc:
            print(exc)
            return

        reference_questions: List[AnsweredQuestion] = [
            q for q in reference_dataset.rag_questions if isinstance(q, AnsweredQuestion)
        ]
        if not reference_questions:
            print("Reference dataset contains no answered (ground-truth) questions.")
            return

        result = run_evaluate(student_results, reference_questions, EVAL_KS)
        print("Evaluation Results")
        print("=" * 40)
        print(f"Questions evaluated: {result.questions_evaluated}")
        for k in EVAL_KS:
            score = result.recall_at_k.get(k, 0.0)
            print(f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)")


__all__ = ["RagCLI"]
