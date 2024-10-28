from flax import struct

from typing import Optional

import jax.numpy as jnp


@struct.dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000

    max_batch_size: int = 32
    max_seq_len: int = 2048


def make_causal_mask(seq_len: int) -> jnp.ndarray:
    mask = jnp.full((seq_len, seq_len), 1, dtype=bool)
    mask = jnp.triu(mask, k=1)
    return mask
