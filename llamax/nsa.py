"""JAX wrapper for DeepSeek's Native Sparse Attention (NSA) Triton kernels.

Wraps the ``fla-org/native-sparse-attention`` Triton implementation of NSA
from the DeepSeek paper (arXiv 2502.11089).  Uses ``jax.pure_callback`` for
PyTorch interop, ``jax.custom_vjp`` for gradient support, and DLPack for
zero-copy GPU tensor sharing.

NSA computes attention via three gated branches:
  1. **Compressed** — coarse-grained attention over block-pooled KV.
  2. **Selected**   — fine-grained attention over top-k selected blocks.
  3. **Sliding window** — local context window.

Requirements (only needed at runtime)::

    pip install native-sparse-attention

Usage::

    from llamax.nsa import NativeSparseAttention, NSAConfig, nsa_attention

    # Functional API — expects pre-projected Q/K/V and gates
    config = NSAConfig(block_size=64, block_counts=16, window_size=512)
    output = nsa_attention(q, k, v, g_cmp, g_slc, g_swa, config)

    # Flax module — drop-in replacement for llamax.model.Attention
    attn = NativeSparseAttention(
        n_heads=32, n_kv_heads=4, dim=2048,
        max_batch_size=1, max_seq_len=2048,
        nsa_config=NSAConfig(),
    )
"""

from __future__ import annotations

import dataclasses
import functools

import flax.linen as nn
import jax
import jax.numpy as jnp

from llamax.model import apply_rotary_emb

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class NSAConfig:
    """Configuration for Native Sparse Attention.

    Args:
        block_size: Block size used for KV pooling and block selection.
        block_counts: Number of top-k blocks selected per query position.
        window_size: Sliding-window size in tokens (0 disables the window).
        scale: Softmax scale factor.  ``None`` → ``1 / sqrt(head_dim)``.
    """

    block_size: int = 64
    block_counts: int = 16
    window_size: int = 512
    scale: float | None = None


# ---------------------------------------------------------------------------
# Tensor conversion helpers (JAX ↔ PyTorch via DLPack)
# ---------------------------------------------------------------------------


def _jax_to_torch(x: jnp.ndarray):
    """Convert a JAX array to a PyTorch tensor (zero-copy on GPU)."""
    import torch

    return torch.from_dlpack(x)


def _torch_to_jax(x) -> jnp.ndarray:
    """Convert a PyTorch tensor to a JAX array (zero-copy on GPU)."""
    return jnp.from_dlpack(x.detach().contiguous())


# ---------------------------------------------------------------------------
# Forward / backward implementations (run inside jax.pure_callback)
# ---------------------------------------------------------------------------


def _forward_impl(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g_cmp: jnp.ndarray,
    g_slc: jnp.ndarray,
    g_swa: jnp.ndarray,
    config: NSAConfig,
) -> jnp.ndarray:
    """Run ``parallel_nsa`` forward without autograd."""
    import torch
    from native_sparse_attention.ops import parallel_nsa

    with torch.no_grad():
        out = parallel_nsa(
            q=_jax_to_torch(q),
            k=_jax_to_torch(k),
            v=_jax_to_torch(v),
            g_cmp=_jax_to_torch(g_cmp),
            g_slc=_jax_to_torch(g_slc),
            g_swa=_jax_to_torch(g_swa),
            block_size=config.block_size,
            block_counts=config.block_counts,
            window_size=config.window_size,
            scale=config.scale,
            head_first=False,
        )
    return _torch_to_jax(out)


def _backward_impl(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g_cmp: jnp.ndarray,
    g_slc: jnp.ndarray,
    g_swa: jnp.ndarray,
    grad_output: jnp.ndarray,
    config: NSAConfig,
) -> tuple[jnp.ndarray, ...]:
    """Re-run forward with autograd and call ``.backward()`` to get grads."""
    from native_sparse_attention.ops import parallel_nsa

    q_pt = _jax_to_torch(q).clone().requires_grad_(True)
    k_pt = _jax_to_torch(k).clone().requires_grad_(True)
    v_pt = _jax_to_torch(v).clone().requires_grad_(True)
    g_cmp_pt = _jax_to_torch(g_cmp).clone().requires_grad_(True)
    g_slc_pt = _jax_to_torch(g_slc).clone().requires_grad_(True)
    g_swa_pt = _jax_to_torch(g_swa).clone().requires_grad_(True)

    out = parallel_nsa(
        q=q_pt,
        k=k_pt,
        v=v_pt,
        g_cmp=g_cmp_pt,
        g_slc=g_slc_pt,
        g_swa=g_swa_pt,
        block_size=config.block_size,
        block_counts=config.block_counts,
        window_size=config.window_size,
        scale=config.scale,
        head_first=False,
    )
    out.backward(_jax_to_torch(grad_output))

    return tuple(
        _torch_to_jax(t.grad)
        for t in (q_pt, k_pt, v_pt, g_cmp_pt, g_slc_pt, g_swa_pt)
    )


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(6,))
def nsa_attention(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g_cmp: jnp.ndarray,
    g_slc: jnp.ndarray,
    g_swa: jnp.ndarray,
    config: NSAConfig,
) -> jnp.ndarray:
    """Native Sparse Attention using DeepSeek's Triton kernels.

    Calls ``fla-org/native-sparse-attention``'s ``parallel_nsa`` via
    ``jax.pure_callback``, with DLPack for zero-copy GPU tensor sharing.

    Block indices for the *selected* branch are computed internally via
    top-k scoring over compressed keys.

    Args:
        q: Queries  ``(batch, seq_len, num_heads, head_dim)``.
        k: Keys     ``(batch, seq_len, num_kv_heads, head_dim)``.
        v: Values   ``(batch, seq_len, num_kv_heads, head_dim)``.
        g_cmp: Compression gate ``(batch, seq_len, num_heads)``.
        g_slc: Selection gate   ``(batch, seq_len, num_heads)``.
        g_swa: Sliding-window gate ``(batch, seq_len, num_heads)``.
        config: :class:`NSAConfig` controlling block size, counts, etc.

    Returns:
        Attention output ``(batch, seq_len, num_heads, head_dim)``.
    """

    def _callback(q, k, v, g_cmp, g_slc, g_swa):
        return _forward_impl(q, k, v, g_cmp, g_slc, g_swa, config)

    result_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    return jax.pure_callback(
        _callback, result_shape, q, k, v, g_cmp, g_slc, g_swa
    )


def _nsa_attention_fwd(q, k, v, g_cmp, g_slc, g_swa, config):
    """Forward rule for :func:`nsa_attention` custom VJP."""
    result = nsa_attention(q, k, v, g_cmp, g_slc, g_swa, config)
    return result, (q, k, v, g_cmp, g_slc, g_swa)


def _nsa_attention_bwd(config, residuals, grad_output):
    """Backward rule for :func:`nsa_attention` custom VJP."""
    q, k, v, g_cmp, g_slc, g_swa = residuals

    def _callback(q, k, v, g_cmp, g_slc, g_swa, g_out):
        return _backward_impl(
            q, k, v, g_cmp, g_slc, g_swa, g_out, config
        )

    result_shapes = (
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct(g_cmp.shape, g_cmp.dtype),
        jax.ShapeDtypeStruct(g_slc.shape, g_slc.dtype),
        jax.ShapeDtypeStruct(g_swa.shape, g_swa.dtype),
    )
    return jax.pure_callback(
        _callback,
        result_shapes,
        q,
        k,
        v,
        g_cmp,
        g_slc,
        g_swa,
        grad_output,
    )


nsa_attention.defvjp(_nsa_attention_fwd, _nsa_attention_bwd)


# ---------------------------------------------------------------------------
# Flax module
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class NativeSparseAttention(nn.Module):
    """Drop-in replacement for :class:`llamax.model.Attention` using
    DeepSeek's Native Sparse Attention.

    The call signature matches ``Attention`` so it can be swapped at the
    ``TransformerBlock`` level.  Internally it computes three branch gates
    via a learned projection and delegates to :func:`nsa_attention`.

    Args:
        n_heads: Number of query attention heads.
        dim: Model dimension.
        max_batch_size: Maximum batch size (kept for API compat).
        max_seq_len: Maximum sequence length.
        n_kv_heads: Number of key/value heads (``None`` → ``n_heads``).
        nsa_config: :class:`NSAConfig` for block size, counts, window.
    """

    n_heads: int
    dim: int
    max_batch_size: int
    max_seq_len: int
    n_kv_heads: int | None = None
    nsa_config: NSAConfig = dataclasses.field(default_factory=NSAConfig)

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: jnp.ndarray | None,
    ):
        del start_pos  # No KV-cache yet.
        del mask  # Sparsity pattern replaces the explicit mask.

        n_kv_heads = self.n_kv_heads or self.n_heads
        head_dim = self.dim // self.n_heads

        wq = nn.Dense(
            features=self.n_heads * head_dim, use_bias=False, name="wq"
        )
        wk = nn.Dense(
            features=n_kv_heads * head_dim, use_bias=False, name="wk"
        )
        wv = nn.Dense(
            features=n_kv_heads * head_dim, use_bias=False, name="wv"
        )
        wg = nn.Dense(
            features=self.n_heads * 3, use_bias=False, name="wg"
        )
        wo = nn.Dense(features=self.dim, use_bias=False, name="wo")

        bsz, seqlen, _ = x.shape

        xq = wq(x).reshape(bsz, seqlen, self.n_heads, head_dim)
        xk = wk(x).reshape(bsz, seqlen, n_kv_heads, head_dim)
        xv = wv(x).reshape(bsz, seqlen, n_kv_heads, head_dim)

        # Gate projection → (batch, seq_len, n_heads, 3) → sigmoid → split.
        gates = jax.nn.sigmoid(
            wg(x).reshape(bsz, seqlen, self.n_heads, 3)
        )
        g_cmp = gates[..., 0]  # (batch, seq_len, n_heads)
        g_slc = gates[..., 1]
        g_swa = gates[..., 2]

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # No transpose needed — parallel_nsa uses (B, T, H, D) layout.
        # No repeat_kv needed — parallel_nsa handles GQA natively.
        output = nsa_attention(
            xq, xk, xv, g_cmp, g_slc, g_swa, self.nsa_config
        )

        output = output.reshape(bsz, seqlen, -1)
        return wo(output)
