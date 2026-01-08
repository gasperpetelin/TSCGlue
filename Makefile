.PHONY: help install-uv setup list clean tests format
.ONESHELL:

help:   ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

install-uv:  ## Install uv package manager
	curl -LsSf https://astral.sh/uv/install.sh | sh

setup:  ## Sets up everything needed for a new deployment
	uv sync --all-extras

list:
	@LC_ALL=C $(MAKE) -pRrq -f $(firstword $(MAKEFILE_LIST)) : 2>/dev/null | awk -v RS= -F: '/(^|\n)# Files(\n|$$)/,/(^|\n)# Finished Make data base/ {if ($$1 !~ "^[#.]") {print $$1}}' | sort | grep -E -v -e '^[^[:alnum:]]' -e '^$@$$'

clean: ## Removes env, docs and caches
	rm -rf build/docs
	rm -rf ~/.exturion
	rm -rf .venv
	uv clean all
	uv cache clean

tests: ## Run the unit tests
	uv run --extra dev pytest tests/ -vv -W ignore::DeprecationWarning --capture=no --durations=0 --cache-clear --maxfail=1

format: ## Format the code with isort and ruff
	uv run --extra dev isort . --profile black
	uv run --extra dev ruff format .
	uv run --extra dev ruff check . --fix
