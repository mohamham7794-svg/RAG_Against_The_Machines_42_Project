"""Local evaluation helper: recall@k against a ground-truth dataset.

This mirrors (a simplified version of) the metric the official moulinette
computes, so contributors can iterate locally. It is NOT the official
grading tool: the moulinette executable remains authoritative during the
defense, and this module must never be imported by it or used to bypass it.
"""

from __future__ import annotations

from typing import Dict, List

from src.models import AnsweredQuestion, EvaluationResult, MinimalSearchResults, MinimalSource

IOU_THRESHOLD = 0.05


def _overlap_iou(a: MinimalSource, b: MinimalSource) -> float:
    """Compute the character-range IoU between two sources in the same file.

    Args:
        a: First source location.
        b: Second source location.

    Returns:
        The intersection-over-union of the two character ranges, or 0.0 if
        the sources are in different files or do not overlap.
    """
    if a.file_path != b.file_path:
        return 0.0
    start = max(a.first_character_index, b.first_character_index)
    end = min(a.last_character_index, b.last_character_index)
    intersection = max(0, end - start)
    if intersection == 0:
        return 0.0
    union = (
        (a.last_character_index - a.first_character_index)
        + (b.last_character_index - b.first_character_index)
        - intersection
    )
    if union <= 0:
        return 0.0
    return intersection / union


def _is_match(predicted: MinimalSource, reference: MinimalSource) -> bool:
    """Whether a predicted source counts as retrieving a reference source.

    Args:
        predicted: A source returned by the student's retriever.
        reference: A ground-truth source for the question.

    Returns:
        True if they are in the same file and overlap with IoU >= 0.05.
    """
    return _overlap_iou(predicted, reference) >= IOU_THRESHOLD


def recall_at_k(
    predicted: List[MinimalSource], reference: List[MinimalSource], k: int
) -> float:
    """Compute recall@k for a single question.

    Args:
        predicted: Ranked list of retrieved sources (only the first ``k``
            are considered).
        reference: Ground-truth sources for the question.
        k: Cutoff rank.

    Returns:
        The fraction of reference sources found within the top-k predicted
        sources. Returns 1.0 if there are no reference sources at all.
    """
    if not reference:
        return 1.0
    top_k = predicted[:k]
    found = 0
    for ref in reference:
        if any(_is_match(pred, ref) for pred in top_k):
            found += 1
    return found / len(reference)


def evaluate(
    student_results: List[MinimalSearchResults],
    reference_questions: List[AnsweredQuestion],
    ks: List[int],
) -> EvaluationResult:
    """Evaluate a batch of student search results against ground truth.

    Args:
        student_results: The student's retrieved sources, one entry per
            question.
        reference_questions: The ground-truth answered questions.
        ks: The list of k cutoffs to report recall for.

    Returns:
        An :class:`~src.models.EvaluationResult` with recall@k for each
        requested k, averaged over all matched questions.
    """
    reference_by_id = {q.question_id: q for q in reference_questions}
    scores: Dict[int, List[float]] = {k: [] for k in ks}

    for result in student_results:
        reference = reference_by_id.get(result.question_id)
        if reference is None:
            continue
        for k in ks:
            scores[k].append(recall_at_k(result.retrieved_sources, reference.sources, k))

    recall_summary = {
        k: (sum(vals) / len(vals) if vals else 0.0) for k, vals in scores.items()
    }
    questions_evaluated = len(scores[ks[0]]) if ks else 0
    return EvaluationResult(
        questions_evaluated=questions_evaluated,
        recall_at_k=recall_summary,
    )
