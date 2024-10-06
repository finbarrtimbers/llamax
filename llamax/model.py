import math
from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import dataclasses

from llamax import reference_model_torch

@dataclasses.dataclass
class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        weight = self.param('weight', nn.initializers.ones, (self.dim,))
        return x * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps) * weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (jnp.arange(0, dim, 2)[:dim // 2].astype(jnp.float32) / dim))
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)

def apply_rotary_emb(xq: jnp.ndarray, xk: jnp.ndarray, freqs_cis: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    xq_ = jnp.stack([xq[..., ::2], xq[..., 1::2]], axis=-1)
    xk_ = jnp.stack([xk[..., ::2], xk[..., 1::2]], axis=-1)
    freqs_cis = jax.lax.broadcast_in_dim(freqs_cis, xq_.shape, (1, 2))
    xq_out = jnp.stack([xq_[..., 0] * freqs_cis.real - xq_[..., 1] * freqs_cis.imag,
                        xq_[..., 1] * freqs_cis.real + xq_[..., 0] * freqs_cis.imag], axis=-1)
    xk_out = jnp.stack([xk_[..., 0] * freqs_cis.real - xk_[..., 1] * freqs_cis.imag,
                        xk_[..., 1] * freqs_cis.real + xk_[..., 0] * freqs_cis.imag], axis=-1)
    return xq_out.reshape(xq.shape), xk_out.reshape(xk.shape)

def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.repeat(x[:, :, None, :, :], n_rep, axis=2).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


@dataclasses.dataclass
class Attention(nn.Module):
    n_heads: int
    dim: int
    max_batch_size: int
    max_seq_len: int
    n_kv_heads: Optional[int] = None

    @nn.compact
    def __call__(self, x: jnp.ndarray, start_pos: int, freqs_cis: jnp.ndarray, mask: Optional[jnp.ndarray]):
        self.n_kv_heads = self.n_kv_heads or self.n_heads
        head_dim = self.dim // self.n_heads
        n_rep = self.n_heads // self.n_kv_heads

        wq = nn.Dense(features=self.n_heads * head_dim, use_bias=False, name='wq')
        wk = nn.Dense(features=self.n_kv_heads * head_dim, use_bias=False, name='wk')
        wv = nn.Dense(features=self.n_kv_heads * head_dim, use_bias=False, name='wv')
        wo = nn.Dense(features=self.dim, use_bias=False, name='wo')

        bsz, seqlen, _ = x.shape
        xq, xk, xv = wq(x), wk(x), wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_heads, head_dim)
        xk = xk.reshape(bsz, seqlen, self.n_kv_heads, head_dim)
        xv = xv.reshape(bsz, seqlen, self.n_kv_heads, head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        cache_k = self.variable('cache', 'key', lambda: jnp.zeros((self.max_batch_size, self.max_seq_len, self.n_kv_heads, head_dim)))
        cache_v = self.variable('cache', 'value', lambda: jnp.zeros((self.max_batch_size, self.max_seq_len, self.n_kv_heads, head_dim)))

        cache_k.value = cache_k.value.at[:bsz, start_pos:start_pos + seqlen].set(xk)
        cache_v.value = cache_v.value.at[:bsz, start_pos:start_pos + seqlen].set(xv)

        keys = cache_k.value[:bsz, :start_pos + seqlen]
        values = cache_v.value[:bsz, :start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, n_rep)
        values = repeat_kv(values, n_rep)

        xq = jnp.transpose(xq, (0, 2, 1, 3))
        keys = jnp.transpose(keys, (0, 2, 1, 3))
        values = jnp.transpose(values, (0, 2, 1, 3))
        scores = jnp.matmul(xq, jnp.swapaxes(keys, -1, -2)) / math.sqrt(head_dim)
        if mask is not None:
            scores = scores + mask
        scores = jax.nn.softmax(scores, axis=-1)
        output = jnp.matmul(scores, values)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(bsz, seqlen, -1)
        return wo(output)


@dataclasses.dataclass
class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float] = None

    @nn.compact
    def __call__(self, x):
        hidden_dim = int(2 * self.hidden_dim / 3)
        if self.ffn_dim_multiplier is not None:
            hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
        hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        w1 = nn.Dense(features=hidden_dim, use_bias=False, name='w1')
        w2 = nn.Dense(features=self.dim, use_bias=False, name='w2')
        w3 = nn.Dense(features=hidden_dim, use_bias=False, name='w3')
        return w3(x)
        return jax.nn.silu(w1(x)) * w3(x)
        return w2(jax.nn.silu(w1(x)) * w3(x))


def feedforward_params_from_torch(torch_module: reference_model_torch.FeedForward) -> Dict[str, Any]:
    return {'params':
            {'w1': {'kernel': torch_module.w1.weight.detach().numpy().T},
             'w2': {'kernel': torch_module.w2.weight.detach().numpy().T},
             'w3': {'kernel': torch_module.w3.weight.detach().numpy().T}}}


@dataclasses.dataclass
class TransformerBlock(nn.Module):
    layer_id: int
    n_heads: int
    dim: int
    multiple_of: int
    max_batch_size: int
    max_seq_len: int
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5

    @nn.compact
    def __call__(self, x: jnp.ndarray, start_pos: int, freqs_cis: jnp.ndarray, mask: Optional[jnp.ndarray]):
        attention = Attention(n_heads=self.n_heads, dim=self.dim, max_batch_size=self.max_batch_size, max_seq_len=self.max_seq_len)
        feed_forward = FeedForward(dim=self.dim, hidden_dim=4 * self.dim, multiple_of=self.multiple_of, ffn_dim_multiplier=self.ffn_dim_multiplier)
        attention_norm = RMSNorm(self.dim, eps=self.norm_eps)
        ffn_norm = RMSNorm(self.dim, eps=self.norm_eps)

        h = x + attention(attention_norm(x), start_pos, freqs_cis, mask)
        out = h + feed_forward(ffn_norm(h))
        return out


@dataclasses.dataclass
class Transformer(nn.Module):
    vocab_size: int
    n_layers: int
    n_heads: int
    dim: int
    max_seq_len: int
    max_batch_size: int
    multiple_of: int
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0

    @nn.compact
    def __call__(self, tokens: jnp.ndarray, start_pos: int):
        tok_embeddings = nn.Embed(self.vocab_size, self.dim)

        freqs_cis = precompute_freqs_cis(self.dim // self.n_heads, self.max_seq_len * 2, self.rope_theta)

        h = tok_embeddings(tokens)
        freqs_cis = jax.lax.dynamic_slice_in_dim(freqs_cis, start_pos, tokens.shape[1], axis=0)

        mask = None
        if tokens.shape[1] > 1:
            mask = jnp.full((tokens.shape[1], tokens.shape[1]), float('-inf'))
            mask = jnp.triu(mask, k=1)
            mask = jnp.concatenate([jnp.zeros((tokens.shape[1], start_pos)), mask], axis=1)

        for layer_id in range(self.n_layers):
            h = TransformerBlock(
                layer_id=layer_id,
                n_heads=self.n_heads,
                dim=self.dim,
                multiple_of=self.multiple_of,
                ffn_dim_multiplier=self.ffn_dim_multiplier,
                norm_eps=self.norm_eps,
                max_batch_size=self.max_batch_size,
                max_seq_len=self.max_seq_len
            )(h, start_pos, freqs_cis, mask)

        h = RMSNorm(self.dim, eps=self.norm_eps)(h)
        output = nn.Dense(features=self.vocab_size, use_bias=False)(h)
        return output
