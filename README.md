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
- [ ] Add benchmarks.
- [ ] Test against the actual llama3 weights.
- [ ] Add code that can run tests using Modal Labs.
- [ ] (Stretch goal) Implement some of the fancier sharding listed in Meta's [Movie Gen](https://ai.meta.com/research/movie-gen/) paper.
- [ ] Add tests on GPU.
- [x] Pin dependency versions.

## Usage

Run tests with `docker build -t llamax . && docker run -e JAX_ENABLE_X64=True llamax pytest`.
