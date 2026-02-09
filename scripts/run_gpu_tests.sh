#!/bin/bash
# Run GPU tests on Modal using H100 or A100 instances
#
# Usage:
#   ./scripts/run_gpu_tests.sh                    # Run all GPU tests
#   ./scripts/run_gpu_tests.sh test_pattern       # Run tests matching pattern
#   ./scripts/run_gpu_tests.sh --gpu=h100         # Specify GPU type (h100 or a100)
#   ./scripts/run_gpu_tests.sh --gpu=a100 pattern # Combine GPU type and pattern

set -e

# Default values
GPU_TYPE="any"
TEST_FILTER=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu=*)
            GPU_TYPE="${1#*=}"
            shift
            ;;
        --gpu)
            GPU_TYPE="$2"
            shift 2
            ;;
        *)
            TEST_FILTER="$1"
            shift
            ;;
    esac
done

# Validate GPU type
if [[ ! "$GPU_TYPE" =~ ^(any|h100|a100|H100|A100)$ ]]; then
    echo "Error: Invalid GPU type '$GPU_TYPE'. Must be one of: any, h100, a100"
    exit 1
fi

# Convert to lowercase for Modal
GPU_TYPE=$(echo "$GPU_TYPE" | tr '[:upper:]' '[:lower:]')

echo "Running GPU tests on Modal..."
echo "GPU Type: $GPU_TYPE"
if [ -n "$TEST_FILTER" ]; then
    echo "Test Filter: $TEST_FILTER"
    uv run modal run -m llamax.ci_gpu --gpu-type "$GPU_TYPE" --filter "$TEST_FILTER"
else
    echo "Test Filter: all GPU tests"
    uv run modal run -m llamax.ci_gpu --gpu-type "$GPU_TYPE"
fi
