"""Tests for the DeepSpeed sparse attention JAX wrapper.

Unit tests exercise configuration, tensor shapes, and module structure
without requiring DeepSpeed or a GPU.  Integration tests (marked with
``@unittest.skipUnless``) call through to the real DeepSpeed kernels.
"""

from __future__ import annotations

import unittest
from unittest import mock

import jax
import jax.numpy as jnp
import numpy as np

from llamax.deepspeed_sparse_attention import (
    BigBirdSparsityConfig,
    BSLongformerSparsityConfig,
    DenseSparsityConfig,
    FixedSparsityConfig,
    SparseAttention,
    _jax_to_torch,
    _torch_to_jax,
    sparse_attention,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_deepspeed() -> bool:
    try:
        import deepspeed.ops.sparse_attention  # noqa: F401
        return True
    except Exception:
        return False


def _has_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _has_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Sparsity config tests (no external deps needed)
# ---------------------------------------------------------------------------


class TestSparsityConfigs(unittest.TestCase):
    """Test that sparsity config dataclasses can be created and are frozen."""

    def test_fixed_defaults(self):
        cfg = FixedSparsityConfig(num_heads=8)
        self.assertEqual(cfg.num_heads, 8)
        self.assertEqual(cfg.block, 16)
        self.assertEqual(cfg.attention, "unidirectional")
        self.assertEqual(cfg.num_local_blocks, 4)
        self.assertEqual(cfg.num_global_blocks, 1)

    def test_fixed_custom(self):
        cfg = FixedSparsityConfig(
            num_heads=16,
            block=64,
            num_local_blocks=8,
            num_global_blocks=2,
            attention="bidirectional",
        )
        self.assertEqual(cfg.block, 64)
        self.assertEqual(cfg.num_local_blocks, 8)
        self.assertEqual(cfg.attention, "bidirectional")

    def test_fixed_is_frozen(self):
        cfg = FixedSparsityConfig(num_heads=4)
        with self.assertRaises(AttributeError):
            cfg.num_heads = 8  # type: ignore[misc]

    def test_fixed_is_hashable(self):
        cfg1 = FixedSparsityConfig(num_heads=4, block=32)
        cfg2 = FixedSparsityConfig(num_heads=4, block=32)
        self.assertEqual(hash(cfg1), hash(cfg2))
        self.assertEqual(cfg1, cfg2)

    def test_bigbird_defaults(self):
        cfg = BigBirdSparsityConfig(num_heads=8)
        self.assertEqual(cfg.num_random_blocks, 1)
        self.assertEqual(cfg.num_sliding_window_blocks, 3)
        self.assertEqual(cfg.attention, "bidirectional")

    def test_longformer_defaults(self):
        cfg = BSLongformerSparsityConfig(num_heads=8)
        self.assertEqual(cfg.num_sliding_window_blocks, 3)
        self.assertEqual(cfg.num_global_blocks, 1)

    def test_dense_defaults(self):
        cfg = DenseSparsityConfig(num_heads=4)
        self.assertEqual(cfg.block, 16)
        self.assertFalse(cfg.different_layout_per_head)


# ---------------------------------------------------------------------------
# Tensor conversion tests (requires torch)
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_torch(), "PyTorch not available")
class TestTensorConversion(unittest.TestCase):
    """Test JAX <-> PyTorch tensor conversion via DLPack."""

    def setUp(self):
        jax.config.update("jax_platform_name", "cpu")

    def test_jax_to_torch_shape_dtype(self):
        import torch

        x_jax = jnp.ones((2, 4, 8), dtype=jnp.float32)
        x_pt = _jax_to_torch(x_jax)
        self.assertEqual(x_pt.shape, (2, 4, 8))
        self.assertEqual(x_pt.dtype, torch.float32)

    def test_torch_to_jax_shape_dtype(self):
        import torch

        x_pt = torch.ones(3, 5, dtype=torch.float16)
        x_jax = _torch_to_jax(x_pt)
        self.assertEqual(x_jax.shape, (3, 5))
        self.assertEqual(x_jax.dtype, jnp.float16)

    def test_roundtrip_preserves_values(self):
        x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        x_pt = _jax_to_torch(x)
        x_back = _torch_to_jax(x_pt)
        np.testing.assert_array_equal(np.array(x), np.array(x_back))

    def test_float16_roundtrip(self):
        x = jnp.ones((2, 3), dtype=jnp.float16)
        x_pt = _jax_to_torch(x)
        x_back = _torch_to_jax(x_pt)
        np.testing.assert_array_equal(np.array(x), np.array(x_back))


# ---------------------------------------------------------------------------
# Flax module structure tests (no GPU or DeepSpeed needed)
# ---------------------------------------------------------------------------


class TestSparseAttentionModule(unittest.TestCase):
    """Test that the SparseAttention Flax module has the correct structure."""

    def setUp(self):
        jax.config.update("jax_platform_name", "cpu")

    def test_module_creation(self):
        cfg = FixedSparsityConfig(num_heads=4, block=16)
        module = SparseAttention(
            n_heads=4,
            dim=64,
            max_batch_size=2,
            max_seq_len=128,
            sparsity_config=cfg,
        )
        self.assertEqual(module.n_heads, 4)
        self.assertEqual(module.dim, 64)
        self.assertIsInstance(module.sparsity_config, FixedSparsityConfig)

    def test_module_with_gqa(self):
        cfg = FixedSparsityConfig(num_heads=8, block=16)
        module = SparseAttention(
            n_heads=8,
            dim=128,
            max_batch_size=2,
            max_seq_len=256,
            n_kv_heads=2,
            sparsity_config=cfg,
        )
        self.assertEqual(module.n_kv_heads, 2)

    def test_module_default_config(self):
        module = SparseAttention(
            n_heads=4,
            dim=64,
            max_batch_size=2,
            max_seq_len=128,
        )
        self.assertIsNone(module.sparsity_config)


# ---------------------------------------------------------------------------
# Functional API tests with mocked DeepSpeed
# ---------------------------------------------------------------------------


class TestSparseAttentionFunctional(unittest.TestCase):
    """Test the sparse_attention function using a mock DeepSpeed backend."""

    def setUp(self):
        jax.config.update("jax_platform_name", "cpu")

    @unittest.skipUnless(_has_torch(), "PyTorch not available")
    def test_forward_calls_deepspeed(self):
        """Verify that the forward path calls through to DeepSpeed."""
        import torch

        batch, heads, seq_len, head_dim = 2, 4, 32, 16
        q = jnp.ones((batch, heads, seq_len, head_dim), dtype=jnp.float16)
        k = jnp.ones((batch, heads, seq_len, head_dim), dtype=jnp.float16)
        v = jnp.ones((batch, heads, seq_len, head_dim), dtype=jnp.float16)

        expected_np = np.ones(
            (batch, heads, seq_len, head_dim), dtype=np.float16
        )

        mock_module = mock.MagicMock()
        mock_module.return_value = torch.from_numpy(expected_np)

        cfg = FixedSparsityConfig(num_heads=heads, block=16)

        with mock.patch(
            "llamax.deepspeed_sparse_attention._get_sparse_attn_module",
            return_value=mock_module,
        ):
            result = sparse_attention(q, k, v, cfg, max_seq_length=128)

        self.assertEqual(result.shape, (batch, heads, seq_len, head_dim))
        self.assertEqual(result.dtype, jnp.float16)
        mock_module.assert_called_once()

    @unittest.skipUnless(_has_torch(), "PyTorch not available")
    def test_output_shape_matches_input(self):
        """Output shape must match query shape."""
        import torch

        shapes = [
            (1, 2, 16, 8),
            (4, 8, 64, 32),
            (2, 16, 128, 64),
        ]
        cfg = FixedSparsityConfig(num_heads=2, block=16)

        for shape in shapes:
            q = jnp.zeros(shape, dtype=jnp.float16)
            k = jnp.zeros(shape, dtype=jnp.float16)
            v = jnp.zeros(shape, dtype=jnp.float16)

            expected_np = np.zeros(shape, dtype=np.float16)
            mock_module = mock.MagicMock()
            mock_module.return_value = torch.from_numpy(expected_np)

            with mock.patch(
                "llamax.deepspeed_sparse_attention._get_sparse_attn_module",
                return_value=mock_module,
            ):
                result = sparse_attention(
                    q, k, v, cfg, max_seq_length=256
                )

            self.assertEqual(
                result.shape,
                shape,
                f"Shape mismatch for input shape {shape}",
            )


# ---------------------------------------------------------------------------
# Integration tests (require DeepSpeed + CUDA GPU)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _has_deepspeed() and _has_cuda(),
    "DeepSpeed sparse attention and CUDA GPU required",
)
class TestSparseAttentionIntegration(unittest.TestCase):
    """End-to-end tests against the real DeepSpeed kernels."""

    def test_fixed_sparsity_forward(self):
        batch, heads, seq_len, head_dim = 2, 4, 128, 64
        cfg = FixedSparsityConfig(
            num_heads=heads,
            block=16,
            attention="unidirectional",
        )
        rng = jax.random.PRNGKey(0)
        k1, k2, k3 = jax.random.split(rng, 3)
        shape = (batch, heads, seq_len, head_dim)

        q = jax.random.normal(k1, shape, dtype=jnp.float16)
        k = jax.random.normal(k2, shape, dtype=jnp.float16)
        v = jax.random.normal(k3, shape, dtype=jnp.float16)

        result = sparse_attention(q, k, v, cfg, max_seq_length=seq_len)

        self.assertEqual(result.shape, shape)
        self.assertEqual(result.dtype, jnp.float16)
        self.assertTrue(jnp.isfinite(result).all())

    def test_bigbird_forward(self):
        batch, heads, seq_len, head_dim = 1, 2, 64, 32
        cfg = BigBirdSparsityConfig(num_heads=heads, block=16)
        rng = jax.random.PRNGKey(1)
        k1, k2, k3 = jax.random.split(rng, 3)
        shape = (batch, heads, seq_len, head_dim)

        q = jax.random.normal(k1, shape, dtype=jnp.float16)
        k = jax.random.normal(k2, shape, dtype=jnp.float16)
        v = jax.random.normal(k3, shape, dtype=jnp.float16)

        result = sparse_attention(q, k, v, cfg, max_seq_length=seq_len)
        self.assertEqual(result.shape, shape)
        self.assertTrue(jnp.isfinite(result).all())

    def test_gradient_flow(self):
        """Verify that gradients flow through sparse_attention."""
        batch, heads, seq_len, head_dim = 1, 2, 32, 16
        cfg = FixedSparsityConfig(num_heads=heads, block=16)
        shape = (batch, heads, seq_len, head_dim)

        rng = jax.random.PRNGKey(2)
        k1, k2, k3 = jax.random.split(rng, 3)
        q = jax.random.normal(k1, shape, dtype=jnp.float16)
        k = jax.random.normal(k2, shape, dtype=jnp.float16)
        v = jax.random.normal(k3, shape, dtype=jnp.float16)

        def loss_fn(q, k, v):
            out = sparse_attention(q, k, v, cfg, max_seq_length=seq_len)
            return jnp.sum(out)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2))(q, k, v)
        for i, g in enumerate(grads):
            self.assertEqual(g.shape, shape, f"Grad {i} shape mismatch")
            self.assertTrue(
                jnp.isfinite(g).all(), f"Grad {i} contains non-finite"
            )


if __name__ == "__main__":
    unittest.main(failfast=True)
