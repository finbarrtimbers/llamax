"""Modal CI configuration for running GPU tests."""

from pathlib import Path

import modal

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


def create_gpu_test_function(gpu_type: str = "any"):
    """Create a Modal function with the specified GPU type."""

    @app.function(
        gpu=gpu_type,
        timeout=3600,  # 1 hour timeout for tests
        secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
        mounts=[code_mount],
    )
    def run_tests(test_filter: str = "gpu"):
        """Run pytest on GPU tests with optional filter."""
        import subprocess

        # Build pytest command
        cmd = [
            "pytest",
            "-v",
            "/root/llamax/llamax",
            "-k",
            test_filter,
            "--tb=short",
        ]

        # Run pytest
        result = subprocess.run(
            cmd,
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

    return run_tests


# Default function for backward compatibility (used by GitHub Actions)
@app.function(
    gpu="any",
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
    mounts=[code_mount],
)
def run_gpu_tests():
    """Run pytest on all tests with 'gpu' in the name (backward compatible)."""
    import subprocess

    result = subprocess.run(
        [
            "pytest",
            "-v",
            "/root/llamax/llamax",
            "-k",
            "gpu",
            "--tb=short",
        ],
        check=False,
        cwd="/root/llamax",
        capture_output=True,
        text=True,
    )

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise SystemExit(result.returncode)

    return result.returncode


@app.local_entrypoint()
def main(gpu_type: str = "any", filter: str = "gpu"):
    """
    Main entrypoint for running GPU tests locally via Modal.

    Args:
        gpu_type: GPU type to use (any, h100, a100, etc.)
        filter: Test filter pattern for pytest -k option
    """
    print(f"Running GPU tests with GPU type: {gpu_type}")
    print(f"Test filter: {filter}")

    # Create function with specified GPU type
    test_func = create_gpu_test_function(gpu_type)

    # Run the tests
    return test_func.remote(test_filter=filter)
