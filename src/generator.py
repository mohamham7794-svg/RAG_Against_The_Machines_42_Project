"""Answer generation stage: build a grounded prompt and call Qwen3-0.6B.

The model is loaded lazily on first use (it is only needed by the
``answer`` / ``answer_dataset`` commands, not by ``index`` / ``search``),
and kept in memory for subsequent calls within the same process.
"""

from __future__ import annotations

from typing import Any, List

from src.models import MinimalSource

DEFAULT_MODEL_NAME = "Qwen/Qwen3-0.6B"

# Leave headroom for the prompt scaffolding and the model's own reasoning
# tokens within Qwen3-0.6B's context window.
MAX_NEW_TOKENS = 512
MAX_CONTEXT_CHARS = 2000


class AnswerGenerator:
    """Wraps a small local causal LM to answer questions from context.

    Attributes:
        model_name: HuggingFace model identifier to load.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        self.model_name = model_name
        self._tokenizer: Any = None
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        """Lazily load the tokenizer and model on first use."""
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment issue
            raise RuntimeError(
                "transformers and torch are required for answer generation. "
                "Run `uv sync` to install project dependencies."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float32,
        )
        self._model.eval()

    @staticmethod
    def build_context(sources: List[MinimalSource], texts: List[str]) -> str:
        """Assemble the retrieved snippets into a single context block.

        Args:
            sources: The retrieved source locations, used for labelling.
            texts: The raw chunk text for each source, same order/length.

        Returns:
            A formatted context string, one labelled section per source.
        """
        blocks = []
        for source, text in zip(sources, texts):
            if not text:
                continue
            snippet = text[:MAX_CONTEXT_CHARS]
            blocks.append(f"[Source: {source.file_path}]\n{snippet}")
        return "\n\n".join(blocks)

    @staticmethod
    def build_prompt(question: str, context: str) -> str:
        """Build the instruction prompt handed to the model.

        Args:
            question: The user's natural-language question.
            context: The formatted retrieved context (see
                :meth:`build_context`).

        Returns:
            The full prompt string.
        """
        if not context.strip():
            return (
                "You are a helpful assistant answering questions about the "
                "vLLM codebase. No relevant context was retrieved, so say "
                "you don't have enough information to answer confidently.\n\n"
                f"Question: {question}\nAnswer:"
            )
        return (
            "You are a helpful assistant answering questions about the "
            "vLLM codebase. Use ONLY the context below to answer. Be "
            "concise and factual, and do not invent information that is "
            "not supported by the context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\nAnswer:"
        )

    def generate(self, question: str, sources: List[MinimalSource], texts: List[str]) -> str:
        """Generate a grounded natural-language answer.

        Args:
            question: The user's natural-language question.
            sources: The retrieved source locations.
            texts: The raw chunk text for each source, same order/length.

        Returns:
            The generated answer text, or a graceful fallback message if
            generation fails for any reason (model unavailable, OOM, ...).
        """
        context = self.build_context(sources, texts)
        prompt = self.build_prompt(question, context)

        try:
            self._ensure_loaded()
        except RuntimeError as exc:
            return f"[generation unavailable: {exc}]"

        try:
            return self._run_model(prompt)
        except Exception as exc:  # noqa: BLE001 - never crash the CLI
            return f"[generation failed: {exc}]"

    def _run_model(self, prompt: str) -> str:
        """Run the loaded model on a chat-formatted prompt.

        Args:
            prompt: The full instruction prompt.

        Returns:
            The decoded, stripped model completion.
        """
        import torch

        messages = [{"role": "user", "content": prompt}]
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer configs without the enable_thinking kwarg.
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        inputs = self._tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        answer = self._tokenizer.decode(generated, skip_special_tokens=True)
        return str(answer).strip()
