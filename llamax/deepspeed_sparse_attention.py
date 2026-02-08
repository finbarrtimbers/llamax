"""JAX wrapper for DeepSpeed's block-sparse attention kernels.

This module provides a JAX-compatible interface to DeepSpeed's block-sparse
attention implementation, which uses Triton kernels for efficient GPU
computation. The wrapper uses ``jax.pure_callback`` to call through to the
PyTorch/DeepSpeed implementation and ``jax.custom_vjp`` for gradient
computation.

Tensor conversion between JAX and PyTorch uses DLPack for zero-copy sharing
on GPU.

Requirements (optional, only needed at runtime)::

    pip install deepspeed[sparse-attn] triton

Usage::

    from llamax.deepspeed_sparse_attention import (
        SparseAttention,
        FixedSparsityConfig,
        sparse_attention,
    )

    # Functional API
    config = FixedSparsityConfig(num_heads=8, block=64)
    output = sparse_attention(query, key, value, config)

    # Flax module (drop-in replacement for llamax.model.Attention)
    attn = SparseAttention(
        n_heads=8, dim=512, max_seq_len=2048,
        sparsity_config=FixedSparsityConfig(num_heads=8, block=64),
    )
"""

from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np

from llamax.model import apply_rotary_emb, repeat_kv

# ---------------------------------------------------------------------------
# Sparsity configuration dataclasses
# ---------------------------------------------------------------------------
# These are pure-Python dataclasses that mirror DeepSpeed's sparsity config
# classes.  They can be created without importing DeepSpeed; the actual
# DeepSpeed config object is built lazily via ``_to_deepspeed()``.
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class FixedSparsityConfig:
    """Fixed-pattern sparsity with local and global blocks.

    Args:
        num_heads: Number of attention heads.
        block: Block size for the sparsity pattern (typically 16, 32, or 64).
        different_layout_per_head: Use a different layout for each head.
        num_local_blocks: Number of local (sliding-window) blocks.
        num_global_blocks: Number of global attention blocks.
        attention: ``"unidirectional"`` for causal or ``"bidirectional"``.
        horizontal_global_attention: Add horizontal global attention.
        num_different_global_patterns: Number of distinct global patterns.
    """

    num_heads: int
    block: int = 16
    different_layout_per_head: bool = False
    num_local_blocks: int = 4
    num_global_blocks: int = 1
    attention: str = "unidirectional"
    horizontal_global_attention: bool = False
    num_different_global_patterns: int = 1

    def _to_deepspeed(self):
        from deepspeed.ops.sparse_attention import (
            FixedSparsityConfig as _DS,
        )

        return _DS(
            num_heads=self.num_heads,
            block=self.block,
            different_layout_per_head=self.different_layout_per_head,
            num_local_blocks=self.num_local_blocks,
            num_global_blocks=self.num_global_blocks,
            attention=self.attention,
            horizontal_global_attention=self.horizontal_global_attention,
            num_different_global_patterns=self.num_different_global_patterns,
        )


@dataclasses.dataclass(frozen=True)
class BigBirdSparsityConfig:
    """BigBird-style sparsity with random, sliding-window, and global blocks.

    Args:
        num_heads: Number of attention heads.
        block: Block size for the sparsity pattern.
        different_layout_per_head: Use a different layout for each head.
        num_random_blocks: Number of random attention blocks.
        num_sliding_window_blocks: Number of sliding-window blocks.
        num_global_blocks: Number of global attention blocks.
        attention: ``"unidirectional"`` for causal or ``"bidirectional"``.
    """

    num_heads: int
    block: int = 16
    different_layout_per_head: bool = False
    num_random_blocks: int = 1
    num_sliding_window_blocks: int = 3
    num_global_blocks: int = 1
    attention: str = "bidirectional"

    def _to_deepspeed(self):
        from deepspeed.ops.sparse_attention import (
            BigBirdSparsityConfig as _DS,
        )

        return _DS(
            num_heads=self.num_heads,
            block=self.block,
            different_layout_per_head=self.different_layout_per_head,
            num_random_blocks=self.num_random_blocks,
            num_sliding_window_blocks=self.num_sliding_window_blocks,
            num_global_blocks=self.num_global_blocks,
            attention=self.attention,
        )


@dataclasses.dataclass(frozen=True)
class BSLongformerSparsityConfig:
    """Longformer-style sparsity with sliding-window and global attention.

    Args:
        num_heads: Number of attention heads.
        block: Block size for the sparsity pattern.
        different_layout_per_head: Use a different layout for each head.
        num_sliding_window_blocks: Number of sliding-window blocks.
        num_global_blocks: Number of global attention blocks.
        attention: ``"unidirectional"`` for causal or ``"bidirectional"``.
    """

    num_heads: int
    block: int = 16
    different_layout_per_head: bool = False
    num_sliding_window_blocks: int = 3
    num_global_blocks: int = 1
    attention: str = "bidirectional"

    def _to_deepspeed(self):
        from deepspeed.ops.sparse_attention import (
            BSLongformerSparsityConfig as _DS,
        )

        return _DS(
            num_heads=self.num_heads,
            block=self.block,
            different_layout_per_head=self.different_layout_per_head,
            num_sliding_window_blocks=self.num_sliding_window_blocks,
            num_global_blocks=self.num_global_blocks,
            attention=self.attention,
        )


@dataclasses.dataclass(frozen=True)
class DenseSparsityConfig:
    """Fully-dense attention (no sparsity) -- useful as a baseline.

    Args:
        num_heads: Number of attention heads.
        block: Block size (still used for tiling, but all blocks are active).
        different_layout_per_head: Use a different layout for each head.
    """

    num_heads: int
    block: int = 16
    different_layout_per_head: bool = False

    def _to_deepspeed(self):
        from deepspeed.ops.sparse_attention import (
            DenseSparsityConfig as _DS,
        )

        return _DS(
            num_heads=self.num_heads,
            block=self.block,
            different_layout_per_head=self.different_layout_per_head,
        )


SparsityConfig = (
    FixedSparsityConfig
    | BigBirdSparsityConfig
    | BSLongformerSparsityConfig
    | DenseSparsityConfig
)


# ---------------------------------------------------------------------------
# Tensor conversion helpers (JAX <-> PyTorch via DLPack)
# ---------------------------------------------------------------------------


def _jax_to_torch(x: jnp.ndarray):
    """Convert a JAX array to a PyTorch tensor (zero-copy on GPU via DLPack)."""
    import torch  # noqa: F811

    return torch.from_dlpack(x)


def _torch_to_jax(x) -> jnp.ndarray:
    """Convert a PyTorch tensor to a JAX array (zero-copy on GPU via DLPack)."""
    return jnp.from_dlpack(x.detach().contiguous())


# ---------------------------------------------------------------------------
# Cached DeepSpeed module construction
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=32)
def _get_sparse_attn_module(
    sparsity_config: SparsityConfig,
    max_seq_length: int,
):
    """Return a cached ``SparseSelfAttention`` module for the given config."""
    from deepspeed.ops.sparse_attention import SparseSelfAttention

    ds_config = sparsity_config._to_deepspeed()
    return SparseSelfAttention(
        sparsity_config=ds_config,
        key_padding_mask_mode="add",
        attn_mask_mode="mul",
        max_seq_length=max_seq_length,
    )


# ---------------------------------------------------------------------------
# Forward / backward implementations (called inside jax.pure_callback)
# ---------------------------------------------------------------------------


def _forward_impl(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    sparsity_config: SparsityConfig,
    max_seq_length: int,
) -> jnp.ndarray:
    """Run DeepSpeed sparse attention forward (no autograd)."""
    import torch

    module = _get_sparse_attn_module(sparsity_config, max_seq_length)

    q_pt = _jax_to_torch(query)
    k_pt = _jax_to_torch(key)
    v_pt = _jax_to_torch(value)

    with torch.no_grad():
        out_pt = module(q_pt, k_pt, v_pt)

    return _torch_to_jax(out_pt)


def _backward_impl(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    grad_output: jnp.ndarray,
    sparsity_config: SparsityConfig,
    max_seq_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute gradients through DeepSpeed sparse attention via autograd.

    This re-runs the forward pass with autograd enabled and then calls
    ``.backward()`` to obtain input gradients.
    """
    import torch

    module = _get_sparse_attn_module(sparsity_config, max_seq_length)

    # Clone so that autograd can track these as leaf tensors.
    q_pt = _jax_to_torch(query).clone().requires_grad_(True)
    k_pt = _jax_to_torch(key).clone().requires_grad_(True)
    v_pt = _jax_to_torch(value).clone().requires_grad_(True)
    g_pt = _jax_to_torch(grad_output)

    out_pt = module(q_pt, k_pt, v_pt)
    out_pt.backward(g_pt)

    return (
        _torch_to_jax(q_pt.grad),
        _torch_to_jax(k_pt.grad),
        _torch_to_jax(v_pt.grad),
    )


# ---------------------------------------------------------------------------
# Public functional API
# ---------------------------------------------------------------------------


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def sparse_attention(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    sparsity_config: SparsityConfig,
    max_seq_length: int = 2048,
) -> jnp.ndarray:
    """Block-sparse attention using DeepSpeed's Triton kernels.

    Calls through to DeepSpeed's ``SparseSelfAttention`` via
    ``jax.pure_callback``, converting tensors between JAX and PyTorch with
    DLPack (zero-copy on GPU).

    Args:
        query: ``(batch, num_heads, seq_len, head_dim)``, float16.
        key: ``(batch, num_heads, seq_len, head_dim)``, float16.
        value: ``(batch, num_heads, seq_len, head_dim)``, float16.
        sparsity_config: One of the ``*SparsityConfig`` dataclasses that
            controls the block-sparse attention pattern.
        max_seq_length: Maximum sequence length (used for internal padding).

    Returns:
        Attention output with the same shape and dtype as *query*.
    """

    def _callback(q, k, v):
        return _forward_impl(q, k, v, sparsity_config, max_seq_length)

    result_shape = jax.ShapeDtypeStruct(query.shape, query.dtype)
    return jax.pure_callback(_callback, result_shape, query, key, value)


def _sparse_attention_fwd(
    query: jnp.ndarray,
    key: jnp.ndarray,
    value: jnp.ndarray,
    sparsity_config: SparsityConfig,
    max_seq_length: int,
):
    """Forward rule for ``sparse_attention`` custom VJP."""
    result = sparse_attention(
        query, key, value, sparsity_config, max_seq_length
    )
    # Save inputs as residuals for the backward pass.
    return result, (query, key, value)


def _sparse_attention_bwd(
    sparsity_config: SparsityConfig,
    max_seq_length: int,
    residuals: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray],
    grad_output: jnp.ndarray,
):
    """Backward rule for ``sparse_attention`` custom VJP."""
    query, key, value = residuals

    def _callback(q, k, v, g):
        return _backward_impl(
            q, k, v, g, sparsity_config, max_seq_length
        )

    result_shapes = (
        jax.ShapeDtypeStruct(query.shape, query.dtype),
        jax.ShapeDtypeStruct(key.shape, key.dtype),
        jax.ShapeDtypeStruct(value.shape, value.dtype),
    )
    grad_q, grad_k, grad_v = jax.pure_callback(
        _callback, result_shapes, query, key, value, grad_output
    )
    return grad_q, grad_k, grad_v


sparse_attention.defvjp(_sparse_attention_fwd, _sparse_attention_bwd)


# ---------------------------------------------------------------------------
# Flax module
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class SparseAttention(nn.Module):
    """Drop-in replacement for :class:`llamax.model.Attention` that uses
    DeepSpeed block-sparse attention.

    The interface intentionally mirrors ``Attention`` so that it can be
    swapped in at the ``TransformerBlock`` level.  The only addition is
    ``sparsity_config``, which controls the block-sparse pattern.

    Args:
        n_heads: Number of query attention heads.
        dim: Model dimension.
        max_batch_size: Maximum batch size (unused, kept for API compat).
        max_seq_len: Maximum sequence length.
        n_kv_heads: Number of key/value heads (defaults to ``n_heads``).
        sparsity_config: Block-sparse pattern configuration.  Defaults to
            :class:`FixedSparsityConfig` with the module's ``n_heads``.
    """

    n_heads: int
    dim: int
    max_batch_size: int
    max_seq_len: int
    n_kv_heads: int | None = None
    sparsity_config: SparsityConfig | None = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: jnp.ndarray | None,
    ):
        del start_pos  # Unused (no KV-cache yet).
        del mask  # Sparsity pattern replaces the explicit mask.

        n_kv_heads = self.n_kv_heads or self.n_heads
        head_dim = self.dim // self.n_heads
        n_rep = self.n_heads // n_kv_heads

        wq = nn.Dense(
            features=self.n_heads * head_dim, use_bias=False, name="wq"
        )
        wk = nn.Dense(
            features=n_kv_heads * head_dim, use_bias=False, name="wk"
        )
        wv = nn.Dense(
            features=n_kv_heads * head_dim, use_bias=False, name="wv"
        )
        wo = nn.Dense(features=self.dim, use_bias=False, name="wo")

        bsz, seqlen, _ = x.shape
        xq, xk, xv = wq(x), wk(x), wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_heads, head_dim)
        xk = xk.reshape(bsz, seqlen, n_kv_heads, head_dim)
        xv = xv.reshape(bsz, seqlen, n_kv_heads, head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # Repeat k/v heads if n_kv_heads < n_heads (GQA).
        xk = repeat_kv(xk, n_rep)
        xv = repeat_kv(xv, n_rep)

        # Transpose to (batch, heads, seq_len, head_dim) for DeepSpeed.
        xq = jnp.transpose(xq, (0, 2, 1, 3))
        xk = jnp.transpose(xk, (0, 2, 1, 3))
        xv = jnp.transpose(xv, (0, 2, 1, 3))

        # DeepSpeed sparse attention requires float16.
        orig_dtype = xq.dtype
        xq = xq.astype(jnp.float16)
        xk = xk.astype(jnp.float16)
        xv = xv.astype(jnp.float16)

        config = self.sparsity_config or FixedSparsityConfig(
            num_heads=self.n_heads,
            attention="unidirectional",
        )

        output = sparse_attention(
            xq, xk, xv, config, self.max_seq_len
        )

        output = output.astype(orig_dtype)

        # Transpose back and project.
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(
            bsz, seqlen, -1
        )
        return wo(output)
