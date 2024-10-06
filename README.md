# llamax

A llama3 implementation in Flax, which aims to be compatible with the official torch version up to floating point error.

This is still a work in-progress; use at your own risk.

## TODO

- [ ] Verify that Flax code matches [reference implementation](https://github.com/meta-llama/llama3/blob/main/llama/model.py) at 6 significant digits in float64 on CPU.
- [ ] Support passing in `kv_mask` and `pairwise_mask` to support padded sequences. Test that this works properly.
- [ ] Implement flax sharding, at least model-parallel + data-parallel.
- [ ] Change the main transformer loop to use a scan instead.
- [ ] Add benchmarks.
- [ ] Test against the actual llama3 weights.
- [ ] Add code that can run tests using Modal Labs.
- [ ] (Stretch goal) Implement some of the fancier sharding listed in Meta's [Movie Gen](https://ai.meta.com/research/movie-gen/) paper.
- [ ] Add tests on GPU.
- [ ] Pin dependency versions.

## Usage

Run tests with `docker build -t llamax . && docker run -e JAX_ENABLE_X64=True llamax pytest`.