.PHONY: install index search answer run debug clean lint lint-strict

QUERY ?= "How to configure the OpenAI server?"
K ?= 5
MAX_CHUNK_SIZE ?= 2000

install:
	uv sync

index:
	uv run python -m src index --max_chunk_size $(MAX_CHUNK_SIZE)

search:
	uv run python -m src search $(QUERY) --k $(K)

answer:
	uv run python -m src answer $(QUERY) --k $(K)

run: search answer

debug:
	uv run python -m pdb -m src search $(QUERY) --k $(K)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .mypy_cache .pytest_cache .ruff_cache

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports \
		--disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 .
	uv run mypy . --strict
