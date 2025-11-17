"""Modal CI configuration for running GPU tests."""

import modal
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Create a Modal image with all necessary dependencies using uv
image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && apt-get install -y curl",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
    )
    .env({"PATH": "/root/.cargo/bin:$PATH", "JAX_ENABLE_X64": "True"})
    .copy_local_file(project_root / "pyproject.toml", "/root/llamax/pyproject.toml")
    .run_commands(
        "cd /root/llamax && /root/.cargo/bin/uv sync --extra dev",
    )
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

    # Run pytest using uv on files with 'gpu' in the name
    result = subprocess.run(
        [
            "/root/.cargo/bin/uv",
            "run",
            "pytest",
            "-v",
            "/root/llamax/llamax",
            "-k",
            "gpu",  # Only run tests with 'gpu' in the name
            "--tb=short",
        ],
        check=False,
        cwd="/root/llamax",
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
