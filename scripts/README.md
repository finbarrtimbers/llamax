# Scripts

## GPU Tests

### `run_gpu_tests.sh`

Run GPU tests on Modal using H100 or A100 instances.

#### Usage

```bash
# Run all GPU tests on any available GPU
./scripts/run_gpu_tests.sh

# Run all GPU tests on H100
./scripts/run_gpu_tests.sh --gpu=h100

# Run all GPU tests on A100
./scripts/run_gpu_tests.sh --gpu=a100

# Run specific test pattern
./scripts/run_gpu_tests.sh test_pattern

# Combine GPU type and test pattern
./scripts/run_gpu_tests.sh --gpu=h100 test_pattern
```

#### Examples

```bash
# Run all GPU tests on H100
./scripts/run_gpu_tests.sh --gpu=h100

# Run only tests with "integration" in the name on A100
./scripts/run_gpu_tests.sh --gpu=a100 integration

# Run tests matching "model" pattern
./scripts/run_gpu_tests.sh model
```

#### Requirements

- Modal account and authentication (`modal setup`)
- `uv` package manager installed
- Modal secrets configured (optional, for HuggingFace access)

#### GPU Types

- `any`: Use any available GPU (default)
- `h100`: NVIDIA H100 GPU
- `a100`: NVIDIA A100 GPU

The test filter uses pytest's `-k` option, so you can use any pytest filter expression.
