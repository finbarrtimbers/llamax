"""Example GPU test file to demonstrate the GPU testing pattern."""

import unittest

import jax
import jax.numpy as jnp


class ExampleGPUTest(unittest.TestCase):
    """Example test class to verify GPU functionality."""

    def test_gpu_available(self):
        """Test that JAX can see GPU devices."""
        devices = jax.devices()
        # This test will pass on both CPU and GPU, but on Modal it should see GPUs
        self.assertGreater(len(devices), 0)
        print(f"Available devices: {devices}")

    def test_simple_gpu_computation(self):
        """Test a simple computation that would benefit from GPU."""
        # Create a simple matrix multiplication
        key = jax.random.PRNGKey(0)
        a = jax.random.normal(key, (1000, 1000))
        b = jax.random.normal(key, (1000, 1000))

        # Perform computation
        c = jnp.dot(a, b)

        # Verify shape
        self.assertEqual(c.shape, (1000, 1000))
        print(f"Computation device: {c.device()}")


if __name__ == "__main__":
    unittest.main()
