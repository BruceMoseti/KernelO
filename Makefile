# KernelForge task runner.
#
# `make help` lists every target. Targets that need a GPU say so; everything
# else runs on a CPU-only machine, including the kernel compilation checks.

PYTHON  ?= python3
PYTEST  ?= $(PYTHON) -m pytest
KF      ?= $(PYTHON) -m kernelforge.cli.main

# Flagship problem sizes, overridable: `make tune M=4096 N=4096 K=4096`
M     ?= 2048
N     ?= 4096
K     ?= 4096
DTYPE ?= fp16
SUITE ?= sweep

.DEFAULT_GOAL := help
.PHONY: help install install-cpu test test-gpu test-all test-slow compile-check \
        lint format check env tune tune-decode explain bench compare profile \
        report experiments cache clean

help: ## List available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- setup ---------------------------------------------------------------
install: ## Install the package with dev and report extras
	pip install -e '.[dev,report]'

install-cpu: ## Install with CPU PyTorch, for machines without a GPU
	pip install torch --index-url https://download.pytorch.org/whl/cpu
	pip install -e '.[dev,report,gpu]'

# --- verification --------------------------------------------------------
test: ## Run the CPU-safe suite (no GPU required)
	$(PYTEST) -m "not gpu" -q

test-gpu: ## Run only the tests that require a CUDA device
	$(PYTEST) -m gpu -q

test-all: ## Run everything (needs a CUDA device for full coverage)
	$(PYTEST) -q

test-slow: ## Compile every budgeted GEMM candidate (no GPU required)
	$(PYTEST) -m slow -q

compile-check: ## Lower every kernel to PTX/cubin for sm80 and sm90, no GPU
	$(PYTEST) tests/test_triton_compile.py tests/test_cuda_rmsnorm.py -m "not gpu" -q

lint: ## Check formatting and lint rules
	ruff check .
	ruff format --check .

format: ## Apply formatting and auto-fixable lint rules
	ruff check --fix .
	ruff format .

check: lint test ## Lint plus the CPU-safe suite: the pre-push gate

# --- running (CUDA device required) --------------------------------------
env: ## Print GPU, driver and library versions
	$(KF) env

explain: ## Show candidate generation for the flagship GEMM (no GPU required)
	$(KF) tune matmul --m $(M) --n $(N) --k $(K) --dtype $(DTYPE) --explain

tune: ## Autotune a GEMM (GPU)
	$(KF) tune matmul --m $(M) --n $(N) --k $(K) --dtype $(DTYPE)

tune-decode: ## Autotune the single-token decode GEMM (GPU)
	$(KF) tune matmul --m 1 --n 11008 --k 4096 --dtype $(DTYPE)

bench: ## Benchmark a workload suite (GPU)
	$(KF) benchmark matmul --suite $(SUITE) --dtype $(DTYPE)

compare: ## Compare implementations of the fused kernel (GPU)
	$(KF) compare fused_linear --m 4096 --n 11008 --k 4096 --dtype $(DTYPE)

profile: ## Attribute device time and count kernel launches (GPU)
	$(KF) profile fused_linear --m 4096 --n 11008 --k 4096

report: ## Render tables and figures from the results database
	$(KF) report

experiments: ## Full reproduction: tune, benchmark, profile, report (GPU, long)
	bash scripts/run_experiments.sh

cache: ## List tuned configurations held in the cache
	$(KF) cache list

# --- housekeeping --------------------------------------------------------
clean: ## Remove caches and generated artefacts
	rm -rf .pytest_cache .ruff_cache reports build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
