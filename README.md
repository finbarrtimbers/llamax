# llamax



![Lint](https://github.com/finbarrtimbers/llamax/actions/workflows/lint.yml/badge.svg)
![Tests](https://github.com/finbarrtimbers/llamax/actions/workflows/test.yml/badge.svg)
![Coverage](https://gist.githubusercontent.com/finbarrtimbers/12ba425b48b5fe95dce24fba21bcbf70/raw/coverage.svg)

A basic Flax library which contains reference implementations of various LLMs. Currently, just Llama3.

The code is very much a work in-progress; use at your own risk.

## TODO

- [x] Verify that Flax code matches [reference implementation](https://github.com/meta-llama/llama3/blob/main/llama/model.py) at 6 significant digits in float64 on CPU.
- [x] Support passing in `kv_mask` and `pairwise_mask` to support padded sequences. Test that this works properly.
- [ ] Implement sharding, at least model-parallel + data-parallel.
- [ ] Change the main transformer loop to use a scan instead.
- [ ] Add throughput benchmarks.
- [x] Test against the actual llama3 weights.
- [ ] Add code that can run tests using Modal Labs.
- [ ] (Stretch goal) Implement some of the fancier sharding methods listed in Meta's [Movie Gen](https://ai.meta.com/research/movie-gen/) paper.
- [ ] Add tests on GPU.
- [x] Pin dependency versions.

## Setup

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) for Python package management:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

1. Clone the repository and install dependencies:

```bash
uv sync --all-extras
```

2. Install pre-commit hooks:

```bash
uv run pre-commit install
```

## Usage

### Running Tests

```bash
uv run pytest
```

### Linting and Formatting

Run the linter and formatter:

```bash
uv run ruff check .
uv run ruff format .
```

Or use the Makefile:

```bash
make style && make quality
```

### Docker

Run tests with `docker build -t llamax . && docker run -e JAX_ENABLE_X64=True -e HF_TOKEN=$HF_TOKEN llamax pytest`.

Get your Huggingface token by following [their instructions](https://huggingface.co/docs/hub/en/security-tokens). If you want to run the integration test, download weights as per instructions below, and then pass them in via `-v $WEIGHTS_DIR:/data` in the Docker command above.


### Weights

I don't include any of the weights needed to test the implementations. You'll
 need to get them yourself. Here's where I got them:

1. **Llama 3.2 1B**: Copy the "original" folder from [Huggingface](https://huggingface.co/meta-llama/Llama-3.2-1B/tree/main/original).
