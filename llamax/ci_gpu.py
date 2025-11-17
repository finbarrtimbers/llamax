"""Modal CI configuration for running GPU tests."""

import modal
from pathlib import Path

# Get the project root directory
project_root = Path(__file__).parent.parent

# Create a Modal image with all necessary dependencies using uv
image = (
    modal.Image.debian_slim(python_version="3.12")
    .add_local_file(project_root / "pyproject.toml", "/root/pyproject.toml")
    .uv_sync(
        local_path=project_root,
        extras=["dev"],
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

    # Run pytest on files with 'gpu' in the name
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
