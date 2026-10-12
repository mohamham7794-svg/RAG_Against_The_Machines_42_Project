*This project has been created as part of the 42 curriculum by mohhammo.*

# RAG against the machine

## Description

This project is a **Retrieval-Augmented Generation (RAG)** system that answers natural-language questions about the [vLLM](https://github.com/vllm-project/vllm) codebase.

Instead of relying on a language model's frozen training data, the system:

1. **Indexes** the vLLM repository (Python source and Markdown/docs) into a searchable, chunked corpus.
2. **Retrieves** the top-k most relevant snippets for a question using a lexical ranking method (BM25).
3. **Generates** a grounded, natural-language answer with a small local model (`Qwen/Qwen3-0.6B`), using only the retrieved snippets as context.
4. **Evaluates** retrieval quality with a Recall@k metric, computed against a ground-truth dataset of questions and their reference sources.

The goal is not to build the best possible chatbot, but to build a correct, end-to-end retrieval pipeline: the right chunks, for the right question, fed to the model in the right way.

## System architecture

```text
data/raw/  --(chunk)-->  Indexer  --(persist)-->  data/processed/
                                                       |
                                                       v
question --(tokenize + BM25 score)--> Retriever --(top-k MinimalSource)-->
                                                       |
                                         (fetch chunk text for each source)
                                                       |
                                                       v
                                           AnswerGenerator
                                            (Qwen/Qwen3-0.6B)
                                                       |
                                                       v
                                      answer (StudentSearchResultsAndAnswer)
```

The pipeline is split into independent, testable stages, each with a single responsibility:

* `src/chunking.py` — pure functions that split raw text into character spans. No I/O or state, making the functions easy to unit-test in isolation.
* `src/indexer.py` — walks `data/raw/`, chunks every file, tokenizes the corpus, and persists everything under `data/processed/` (chunk metadata, tokenized corpus, and a small JSON metadata summary).
* `src/retriever.py` — loads the persisted index lazily, fits a BM25 model in memory, and answers single queries with ranked `MinimalSource` results.
* `src/generator.py` — lazily loads `Qwen/Qwen3-0.6B`, builds a grounded prompt from the retrieved chunk text, and generates an answer.
* `src/evaluator.py` — computes Recall@k locally against a ground-truth dataset. This mirrors, but does not replace, the official moulinette.
* `src/models.py` — the Pydantic contracts exchanged between the different stages.
* `src/cli.py` — the Python Fire CLI wiring the components together.

Every stage communicates through the Pydantic models in `src/models.py`, so the interface between stages is validated rather than assumed.

## Chunking strategy

A Python file and a Markdown page carry structure in very different ways, so two distinct chunking strategies are implemented in `src/chunking.py`.

* **Python code (`chunk_python_code`)**: the file is parsed with the `ast` module and cut at the boundary of every top-level `def`, `async def`, and `class` statement. This keeps a whole function — signature, docstring, and body — inside a single chunk whenever it fits within `--max_chunk_size`. If the file fails to parse because of a syntax error, or a single top-level block is larger than `max_chunk_size`, a sliding-window fallback with a small overlap is used.

* **Markdown / text (`chunk_markdown_text`)**: the file is first split along Markdown headers (`#`, `##`, ...), since a header usually delimits a self-contained topic. Any section that still exceeds `max_chunk_size` is further split along blank-line-separated paragraphs, which are then greedily packed back together up to the size limit (`_pack_spans`). A sliding window is used as a last-resort fallback for an oversized paragraph.

Both strategies produce chunks no larger than `--max_chunk_size` characters (default 2000, matching the moulinette's `max_context_length`) and preserve the character offsets required for retrieval evaluation.

## Retrieval method

Retrieval uses **BM25** via the `rank_bm25` package, one of the two lexical methods allowed by the subject.

At indexing time, every chunk's text is tokenized using an identifier/word-boundary regular expression and lowercased, so `snake_case` identifiers and ordinary English words can both be matched. The tokenized corpus is persisted alongside the chunk metadata.

At query time, the query is tokenized using the same method. BM25 scores every chunk against the query, and the top-k highest-scoring chunks are returned as ranked `MinimalSource` objects. Each result contains the `file_path` copied from the ingested corpus and the chunk's exact character span.

If no chunk scores above zero, the retriever still returns the highest-scoring chunks rather than an empty list, allowing the downstream generation stage to receive potentially useful partial matches.

## Instructions

### Requirements

* Python 3.10+
* [`uv`](https://docs.astral.sh/uv/) as the package/project manager
* The vLLM repository placed under `data/raw/` and not committed to this repository

### Install

```bash
make install
```

This runs:

```bash
uv sync
```

### Index the corpus

Using the Makefile:

```bash
make index
```

The default maximum chunk size is 2000 characters.

It can be changed with:

```bash
make index MAX_CHUNK_SIZE=1500
```

The equivalent direct command is:

```bash
uv run python -m src index \
  --max_chunk_size 2000
```

### Search

Using the Makefile:

```bash
make search
```

The default query is:

```text
How to configure the OpenAI server?
```

and the default `K` value is `5`.

You can override both:

```bash
make search QUERY="How to configure the OpenAI server?" K=5
```

The equivalent direct command is:

```bash
uv run python -m src search "How to configure the OpenAI server?" --k 5
```

### Generate answers

Using the Makefile:

```bash
make answer
```

The default query and `K` value are the same as for `make search`.

You can override them:

```bash
make answer QUERY="How to configure the OpenAI server?" K=5
```

The equivalent direct command is:

```bash
uv run python -m src answer "How to configure the OpenAI server?" --k 5
```

### Search and answer together

The `run` target executes both the search and answer commands:

```bash
make run
```

It is equivalent to running:

```bash
make search
make answer
```

### Debug

The search command can be run under Python's debugger:

```bash
make debug
```

This uses:

```bash
uv run python -m pdb -m src search $(QUERY) --k $(K)
```

### Dataset search

The CLI can run retrieval over the provided question dataset:

```bash
uv run python -m src search_dataset \
  --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
  --k 10 \
  --save_directory data/output/search_results/UnansweredQuestions
```

### Dataset answers

Answers can be generated from a `search_dataset` result:

```bash
uv run python -m src answer_dataset \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

### Local evaluation

The local evaluator can be used during development:

```bash
uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

The official Recall@k used during the defense is computed by the provided `moulinette` executable, not by this command. This project never imports or calls the moulinette.

## Makefile targets

| Target             | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `make install`     | Run `uv sync`                                                |
| `make index`       | Index the vLLM corpus using `MAX_CHUNK_SIZE` (default: 2000) |
| `make search`      | Run a search query using `QUERY` and `K`                     |
| `make answer`      | Generate an answer using `QUERY` and `K`                     |
| `make run`         | Run both `search` and `answer`                               |
| `make debug`       | Run the search command under `pdb`                           |
| `make clean`       | Remove Python caches and generated lint/test caches          |
| `make lint`        | Run `flake8` and `mypy` with the required flags              |
| `make lint-strict` | Run `flake8` and `mypy --strict`                             |

The default Makefile variables are:

```make
QUERY ?= "How to configure the OpenAI server?"
K ?= 5
MAX_CHUNK_SIZE ?= 2000
```

For example:

```bash
make index
make search QUERY="How to configure the OpenAI server?" K=5
make answer QUERY="How to configure the OpenAI server?" K=5
make run
```

## Performance analysis

Indexing and retrieval were tested on the full vLLM corpus and remained within the project's required execution budgets.

Recall@k depends heavily on `--max_chunk_size`. Smaller chunks increase the number of candidate spans and can split related content across chunks, while also making each retrieved span more precise. The default of 2000 characters matches the moulinette's `max_context_length` and was used as the baseline.

## Evaluation results

Official moulinette results on the provided datasets:

| Dataset       | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
| ------------- | -------: | -------: | -------: | --------: |
| Documentation |    65.0% |    81.0% |    85.0% |     89.0% |
| Code          |    42.4% |    60.6% |    63.6% |     68.7% |

The subject requirements at Recall@5 are:

* Documentation: at least 80%
* Code: at least 50%

The official results above were obtained using the provided moulinette executable.

## Design decisions

* **Lexical retrieval for the mandatory part**: BM25 was chosen over TF-IDF because it naturally accounts for term frequency and document length, and works well for natural-language questions against code and documentation without requiring an additional embedding model.

* **Two separate chunking strategies**: code and prose break apart along different structural boundaries, so Python syntax and Markdown/document structure are handled separately.

* **Simple persisted index**: chunks and the tokenized corpus are persisted as pickles, and BM25 is refit in memory from the tokenized corpus when the index is loaded. This keeps the index format simple and avoids an external database or indexing service.

* **CLI input validation**: commands validate common invalid inputs such as missing files, malformed JSON, empty queries, and invalid `k` values, and report these errors with clear messages.

* **Lazy model loading**: the generator is loaded only when `answer` or `answer_dataset` is invoked. Commands such as `index`, `search`, and `evaluate` do not load the language model.

## Challenges faced

* **Character-index bookkeeping**: keeping `first_character_index` and `last_character_index` consistent with the raw file content across two chunking strategies and their fallbacks required careful testing. Incorrect offsets can affect Recall@k evaluation because the official comparison uses the retrieved character spans.

* **flake8 vs. `pyproject.toml`**: flake8 does not read `[tool.flake8]` from `pyproject.toml` without additional support. By default it can also walk into `.venv/`, including third-party packages. A dedicated `setup.cfg` configuration excludes `.venv`, caches, and `data/`.

* **mypy and dynamically typed ML libraries**: `transformers` objects do not provide precise type information for all model and tokenizer attributes. These internal attributes are therefore typed as `Any` where necessary, while the public methods expose precise signatures and remain compatible with `mypy --strict`.

## Example usage

```bash
$ uv run python -m src index --max_chunk_size 2000
Ingestion complete! Indexed 13466 chunks from 1965 files under data/processed/ in 42.1s

$ uv run python -m src search "How to configure the OpenAI server?" --k 5
data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_with_reasoning.py [...]
data/raw/vllm-0.10.1/examples/online_serving/openai_transcription_client.py [...]
data/raw/vllm-0.10.1/benchmarks/auto_tune/README.md [...]
...

$ uv run python -m src answer "How to configure the OpenAI server?" --k 5
To configure the OpenAI-compatible server in vLLM, launch it with the
`--port` and `--model` flags ...
```

The exact retrieved files and character spans can vary depending on the indexed corpus.

## Resources

* [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
* [BM25 — Okapi ranking function overview](https://en.wikipedia.org/wiki/Okapi_BM25)
* [rank_bm25 documentation](https://github.com/dorianbrown/rank_bm25)
* [Hugging Face Transformers documentation](https://huggingface.co/docs/transformers)
* [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B)
* [Pydantic documentation](https://docs.pydantic.dev/)
* [Python Fire documentation](https://github.com/google/python-fire)
* [uv documentation](https://docs.astral.sh/uv/)

### AI usage disclosure

AI assistance was used during this project for:

* Reviewing the generated code against the subject's requirements, including data model shapes, CLI command signatures, output paths, and lint/type issues surfaced by actually running the tooling.
* Drafting and reviewing the README structure and documentation.

All code was read, understood, and tested end-to-end before submission. No part of the pipeline was used without being able to explain how it works.
