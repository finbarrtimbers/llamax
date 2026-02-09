"""Tests for the Native Sparse Attention JAX wrapper.

Unit tests exercise configuration, tensor conversion, and module structure
without requiring ``native-sparse-attention`` or a GPU.  Integration tests
(marked with ``@unittest.skipUnless``) call the real Triton kernels.
"""

from __future__ import annotations

import unittest
from unittest import mock

import jax
import jax.numpy as jnp

from llamax.nsa import (
    NativeSparseAttention,
    NSAConfig,
    _jax_to_torch,
    _torch_to_jax,
    nsa_attention,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_nsa() -> bool:
    try:
        from native_sparse_attention.ops import parallel_nsa  # noqa: F401

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
# Config tests
# ---------------------------------------------------------------------------


class TestNSAConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = NSAConfig()
        self.assertEqual(cfg.block_size, 64)
        self.assertEqual(cfg.block_counts, 16)
        self.assertEqual(cfg.window_size, 512)
        self.assertIsNone(cfg.scale)

    def test_custom(self):
        cfg = NSAConfig(block_size=32, block_counts=8, window_size=256)
        self.assertEqual(cfg.block_size, 32)
        self.assertEqual(cfg.block_counts, 8)
        self.assertEqual(cfg.window_size, 256)

    def test_frozen(self):
        cfg = NSAConfig()
        with self.assertRaises(AttributeError):
            cfg.block_size = 128  # type: ignore[misc]

    def test_hashable(self):
        a = NSAConfig(block_size=64, block_counts=16)
        b = NSAConfig(block_size=64, block_counts=16)
        self.assertEqual(hash(a), hash(b))
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# Tensor conversion tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(_has_torch(), "PyTorch not available")
class TestTensorConversion(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_platform_name", "cpu")

    def test_jax_to_torch_shape_dtype(self):
        import torch

        x = jnp.ones((2, 4, 8), dtype=jnp.float32)
        t = _jax_to_torch(x)
        self.assertEqual(t.shape, (2, 4, 8))
        self.assertEqual(t.dtype, torch.float32)

    def test_torch_to_jax_shape_dtype(self):
        import torch

        t = torch.ones(3, 5, dtype=torch.float16)
        x = _torch_to_jax(t)
        self.assertEqual(x.shape, (3, 5))
        self.assertEqual(x.dtype, jnp.float16)

    def test_roundtrip(self):
        import numpy as np

        x = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
        np.testing.assert_array_equal(
            np.array(x), np.array(_torch_to_jax(_jax_to_torch(x)))
        )

    def test_float16_roundtrip(self):
        import numpy as np

        x = jnp.ones((2, 3), dtype=jnp.float16)
        np.testing.assert_array_equal(
            np.array(x), np.array(_torch_to_jax(_jax_to_torch(x)))
        )


# ---------------------------------------------------------------------------
# Flax module structure tests
# ---------------------------------------------------------------------------


class TestNativeSparseAttentionModule(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_platform_name", "cpu")

    def test_creation(self):
        cfg = NSAConfig(block_size=64)
        m = NativeSparseAttention(
            n_heads=4,
            dim=64,
            max_batch_size=2,
            max_seq_len=128,
            nsa_config=cfg,
        )
        self.assertEqual(m.n_heads, 4)
        self.assertEqual(m.dim, 64)
        self.assertIsInstance(m.nsa_config, NSAConfig)

    def test_gqa(self):
        m = NativeSparseAttention(
            n_heads=8,
            dim=128,
            max_batch_size=2,
            max_seq_len=256,
            n_kv_heads=2,
        )
        self.assertEqual(m.n_kv_heads, 2)

    def test_default_config(self):
        m = NativeSparseAttention(
            n_heads=4, dim=64, max_batch_size=2, max_seq_len=128
        )
        self.assertEqual(m.nsa_config, NSAConfig())


# ---------------------------------------------------------------------------
# Functional API tests (mocked backend)
# ---------------------------------------------------------------------------


class TestNsaAttentionFunctional(unittest.TestCase):
    def setUp(self):
        jax.config.update("jax_platform_name", "cpu")

    @unittest.skipUnless(_has_torch(), "PyTorch not available")
    def test_forward_calls_parallel_nsa(self):
        import numpy as np

        B, T, HQ, H, D = 2, 64, 8, 2, 32
        q = jnp.ones((B, T, HQ, D), dtype=jnp.float32)
        k = jnp.ones((B, T, H, D), dtype=jnp.float32)
        v = jnp.ones((B, T, H, D), dtype=jnp.float32)
        g_cmp = jnp.ones((B, T, HQ), dtype=jnp.float32)
        g_slc = jnp.ones((B, T, HQ), dtype=jnp.float32)
        g_swa = jnp.ones((B, T, HQ), dtype=jnp.float32)

        expected = np.ones((B, T, HQ, D), dtype=np.float32)
        cfg = NSAConfig()

        with mock.patch(
            "llamax.nsa._forward_impl",
            side_effect=lambda *a, **kw: jnp.array(expected),
        ) as mock_fwd:
            result = nsa_attention(q, k, v, g_cmp, g_slc, g_swa, cfg)

        self.assertEqual(result.shape, (B, T, HQ, D))
        mock_fwd.assert_called_once()

    @unittest.skipUnless(_has_torch(), "PyTorch not available")
    def test_output_shape_matches_query(self):
        import numpy as np

        cfg = NSAConfig()
        shapes_qkv = [
            # (q_shape, kv_shape)
            ((1, 32, 4, 16), (1, 32, 2, 16)),
            ((2, 64, 8, 32), (2, 64, 2, 32)),
        ]

        for q_shape, kv_shape in shapes_qkv:
            B, T, HQ, D = q_shape
            q = jnp.zeros(q_shape, dtype=jnp.float32)
            k = jnp.zeros(kv_shape, dtype=jnp.float32)
            v = jnp.zeros(kv_shape, dtype=jnp.float32)
            g = jnp.zeros((B, T, HQ), dtype=jnp.float32)

            expected = np.zeros(q_shape, dtype=np.float32)

            with mock.patch(
                "llamax.nsa._forward_impl",
                return_value=jnp.array(expected),
            ):
                result = nsa_attention(q, k, v, g, g, g, cfg)

            self.assertEqual(result.shape, q_shape)


# ---------------------------------------------------------------------------
# Integration tests (require native-sparse-attention + CUDA)
# ---------------------------------------------------------------------------


@unittest.skipUnless(
    _has_nsa() and _has_cuda(),
    "native-sparse-attention and CUDA GPU required",
)
class TestNsaIntegration(unittest.TestCase):
    def test_forward(self):
        B, T, HQ, H, D = 1, 128, 16, 4, 32
        cfg = NSAConfig(block_size=64, block_counts=2, window_size=64)

        rng = jax.random.PRNGKey(0)
        keys = jax.random.split(rng, 6)
        q = jax.random.normal(keys[0], (B, T, HQ, D), dtype=jnp.float16)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float16)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float16)
        g_cmp = jax.nn.sigmoid(jax.random.normal(keys[3], (B, T, HQ)))
        g_slc = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, HQ)))
        g_swa = jax.nn.sigmoid(jax.random.normal(keys[5], (B, T, HQ)))

        result = nsa_attention(q, k, v, g_cmp, g_slc, g_swa, cfg)
        self.assertEqual(result.shape, (B, T, HQ, D))
        self.assertTrue(jnp.isfinite(result).all())

    def test_gradient_flow(self):
        B, T, HQ, H, D = 1, 64, 16, 4, 32
        cfg = NSAConfig(block_size=64, block_counts=1, window_size=64)

        rng = jax.random.PRNGKey(1)
        keys = jax.random.split(rng, 6)
        q = jax.random.normal(keys[0], (B, T, HQ, D), dtype=jnp.float16)
        k = jax.random.normal(keys[1], (B, T, H, D), dtype=jnp.float16)
        v = jax.random.normal(keys[2], (B, T, H, D), dtype=jnp.float16)
        g_cmp = jax.nn.sigmoid(jax.random.normal(keys[3], (B, T, HQ)))
        g_slc = jax.nn.sigmoid(jax.random.normal(keys[4], (B, T, HQ)))
        g_swa = jax.nn.sigmoid(jax.random.normal(keys[5], (B, T, HQ)))

        def loss_fn(q, k, v, g_cmp, g_slc, g_swa):
            out = nsa_attention(q, k, v, g_cmp, g_slc, g_swa, cfg)
            return jnp.sum(out)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(
            q, k, v, g_cmp, g_slc, g_swa
        )
        for i, g in enumerate(grads):
            self.assertTrue(
                jnp.isfinite(g).all(), f"Grad {i} has non-finite values"
            )


if __name__ == "__main__":
    unittest.main(failfast=True)
