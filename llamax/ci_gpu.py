"""Modal CI configuration for running GPU tests."""

import modal
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Create a Modal image with all necessary dependencies
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]==0.4.34",
        "jaxlib==0.4.34",
        "flax==0.10.0",
        "torch==2.5.0",
        "numpy==2.1.2",
        "pytest==8.3.3",
        "pytest-cov>=6.0.0",
        "pytest-xdist>=3.6.1",
        "parameterized==0.9.0",
        "psutil==6.1.0",
        "transformers==4.45.2",
        "sentencepiece==0.2.0",
        "jax-dataclasses==1.6.1",
    )
    .env({"JAX_ENABLE_X64": "True"})
)

app = modal.App("llamax-gpu-tests", image=image)

# Mount the project code
code_mount = modal.Mount.from_local_dir(
    project_root,
    remote_path="/root/llamax",
)


@app.function(
    gpu="any",  # Request any available GPU
    timeout=3600,  # 1 hour timeout for tests
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
    mounts=[code_mount],
)
def run_gpu_tests():
    """Run pytest on all tests ending with _gpu.py"""
    import subprocess
    import sys

    # Add the project to Python path
    sys.path.insert(0, "/root/llamax")

    # Run pytest on files ending with _gpu.py
    result = subprocess.run(
        [
            "pytest",
            "-v",
            "/root/llamax/llamax",
            "-k",
            "gpu",  # Only run tests with 'gpu' in the name
            "--tb=short",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    # Print output
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    # Exit with the same code as pytest
    if result.returncode != 0:
        raise SystemExit(result.returncode)

    return result.returncode


@app.local_entrypoint()
def main():
    """Main entrypoint for running GPU tests locally via Modal."""
    return run_gpu_tests.remote()
