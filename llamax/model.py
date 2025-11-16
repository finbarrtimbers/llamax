import dataclasses
import math
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax_dataclasses as jdc

import llamax
from llamax import reference_model_torch


@dataclasses.dataclass
class RMSNorm(nn.Module):
    dim: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x):
        weight = self.param("weight", nn.initializers.ones, (self.dim,))
        return (
            x
            * jax.lax.rsqrt(jnp.mean(x**2, axis=-1, keepdims=True) + self.eps)
            * weight
        )


def rmsnorm_params_from_torch(
    torch_module: reference_model_torch.RMSNorm,
) -> dict[str, Any]:
    return {"weight": torch_module.weight.detach().numpy()}


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (
        theta ** (jnp.arange(0, dim, 2)[: dim // 2].astype(jnp.float32) / dim)
    )
    t = jnp.arange(end, dtype=jnp.float32)
    freqs = jnp.outer(t, freqs)
    return jnp.exp(1j * freqs)


def reshape_for_broadcast(freqs_cis: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (
        x.shape[1],
        x.shape[-1],
    ), f"{freqs_cis.shape=}, {x.shape=}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return jnp.reshape(freqs_cis, shape)


def apply_rotary_emb(
    xq: jnp.ndarray,
    xk: jnp.ndarray,
    freqs_cis: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    xq_ = jnp.reshape(xq.astype(jnp.float32), (*xq.shape[:-1], -1, 2))
    xk_ = jnp.reshape(xk.astype(jnp.float32), (*xk.shape[:-1], -1, 2))
    xq_complex = xq_[..., 0] + 1j * xq_[..., 1]
    xk_complex = xk_[..., 0] + 1j * xk_[..., 1]

    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)

    xq_out = xq_complex * freqs_cis
    xk_out = xk_complex * freqs_cis

    xq_out = jnp.stack([jnp.real(xq_out), jnp.imag(xq_out)], axis=-1).reshape(xq.shape)
    xk_out = jnp.stack([jnp.real(xk_out), jnp.imag(xk_out)], axis=-1).reshape(xk.shape)
    return xq_out.astype(xq.dtype), xk_out.astype(xk.dtype)


def repeat_kv(x: jnp.ndarray, n_rep: int) -> jnp.ndarray:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return jnp.repeat(x[:, :, :, None, :], n_rep, axis=2).reshape(
        bs, slen, n_kv_heads * n_rep, head_dim
    )


@dataclasses.dataclass
class Attention(nn.Module):
    n_heads: int
    dim: int
    max_batch_size: int
    max_seq_len: int
    n_kv_heads: int | None = None

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: jnp.ndarray | None,
    ):
        # This is unused until we have a KV cache.
        del start_pos
        n_kv_heads = self.n_kv_heads or self.n_heads
        head_dim = self.dim // self.n_heads
        n_rep = self.n_heads // n_kv_heads

        wq = nn.Dense(features=self.n_heads * head_dim, use_bias=False, name="wq")
        wk = nn.Dense(features=n_kv_heads * head_dim, use_bias=False, name="wk")
        wv = nn.Dense(features=n_kv_heads * head_dim, use_bias=False, name="wv")
        wo = nn.Dense(features=self.dim, use_bias=False, name="wo")

        bsz, seqlen, _ = x.shape
        xq, xk, xv = wq(x), wk(x), wv(x)

        xq = xq.reshape(bsz, seqlen, self.n_heads, head_dim)
        xk = xk.reshape(bsz, seqlen, n_kv_heads, head_dim)
        xv = xv.reshape(bsz, seqlen, n_kv_heads, head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # repeat k/v heads if n_kv_heads < n_heads
        xk = repeat_kv(xk, n_rep)
        xv = repeat_kv(xv, n_rep)

        xq = jnp.transpose(xq, (0, 2, 1, 3))
        xk = jnp.transpose(xk, (0, 2, 1, 3))
        xv = jnp.transpose(xv, (0, 2, 1, 3))
        scores = jnp.matmul(xq, jnp.swapaxes(xk, -1, -2)) / math.sqrt(head_dim)
        if mask is not None:
            scores = jnp.where(mask, float("-inf"), scores)
        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(xq.dtype)
        output = jnp.matmul(scores, xv)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(bsz, seqlen, -1)
        return wo(output)


def attention_params_from_torch(
    torch_module: reference_model_torch.Attention,
) -> dict[str, Any]:
    return {
        "wk": {"kernel": torch_module.wk.weight.detach().numpy().T},
        "wo": {"kernel": torch_module.wo.weight.detach().numpy().T},
        "wq": {"kernel": torch_module.wq.weight.detach().numpy().T},
        "wv": {"kernel": torch_module.wv.weight.detach().numpy().T},
    }


@dataclasses.dataclass
class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    multiple_of: int
    ffn_dim_multiplier: float | None = None

    @nn.compact
    def __call__(self, x):
        hidden_dim = int(2 * self.hidden_dim / 3)
        if self.ffn_dim_multiplier is not None:
            hidden_dim = int(self.ffn_dim_multiplier * hidden_dim)
        hidden_dim = self.multiple_of * (
            (hidden_dim + self.multiple_of - 1) // self.multiple_of
        )

        w1 = nn.Dense(features=hidden_dim, use_bias=False, name="w1")
        w2 = nn.Dense(features=self.dim, use_bias=False, name="w2")
        w3 = nn.Dense(features=hidden_dim, use_bias=False, name="w3")
        return w2(jax.nn.silu(w1(x)) * w3(x))


def feedforward_params_from_torch(
    torch_module: reference_model_torch.FeedForward,
) -> dict[str, Any]:
    return {
        "w1": {"kernel": torch_module.w1.weight.detach().numpy().T},
        "w2": {"kernel": torch_module.w2.weight.detach().numpy().T},
        "w3": {"kernel": torch_module.w3.weight.detach().numpy().T},
    }


@dataclasses.dataclass
class TransformerBlock(nn.Module):
    layer_id: int
    config: llamax.ModelArgs

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        start_pos: int,
        freqs_cis: jnp.ndarray,
        mask: jnp.ndarray | None,
    ):
        attention = Attention(
            n_heads=self.config.n_heads,
            dim=self.config.dim,
            max_batch_size=self.config.max_batch_size,
            max_seq_len=self.config.max_seq_len,
            n_kv_heads=self.config.n_kv_heads,
        )
        feed_forward = FeedForward(
            dim=self.config.dim,
            hidden_dim=4 * self.config.dim,
            multiple_of=self.config.multiple_of,
            ffn_dim_multiplier=self.config.ffn_dim_multiplier,
        )
        attention_norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)
        ffn_norm = RMSNorm(self.config.dim, eps=self.config.norm_eps)

        h = x + attention(attention_norm(x), start_pos, freqs_cis, mask)
        out = h + feed_forward(ffn_norm(h))
        return out


def block_params_from_module(
    torch_module: reference_model_torch.TransformerBlock,
) -> dict[str, Any]:
    return {
        "Attention_0": attention_params_from_torch(torch_module.attention),
        "FeedForward_0": feedforward_params_from_torch(torch_module.feed_forward),
        "RMSNorm_0": rmsnorm_params_from_torch(torch_module.attention_norm),
        "RMSNorm_1": rmsnorm_params_from_torch(torch_module.ffn_norm),
    }


@dataclasses.dataclass
class Transformer(nn.Module):
    config: jdc.Static[llamax.ModelArgs]

    @nn.compact
    def __call__(
        self, tokens: jnp.ndarray, start_pos: int, mask: jnp.ndarray | None = None
    ):
        tok_embeddings = nn.Embed(self.config.vocab_size, self.config.dim)

        freqs_cis = precompute_freqs_cis(
            self.config.dim // self.config.n_heads,
            self.config.max_seq_len * 2,
            self.config.rope_theta,
        )

        h = tok_embeddings(tokens)
        freqs_cis = jax.lax.dynamic_slice_in_dim(
            freqs_cis, start_pos, tokens.shape[1], axis=0
        )

        for layer_id in range(self.config.n_layers):
            h = TransformerBlock(
                layer_id=layer_id,
                config=self.config,
            )(h, start_pos, freqs_cis, mask)

        h = RMSNorm(self.config.dim, eps=self.config.norm_eps)(h)
        output = nn.Dense(features=self.config.vocab_size, use_bias=False)(h)
        return output


def transformer_params_from_module(
    torch_module: reference_model_torch.Transformer,
) -> dict[str, Any]:
    params = {
        "params": {
            "Dense_0": {"kernel": torch_module.output.weight.detach().numpy().T},
            "Embed_0": {
                "embedding": torch_module.tok_embeddings.weight.detach().numpy()
            },
            "RMSNorm_0": rmsnorm_params_from_torch(torch_module.norm),
        }
    }
    for layer_id in range(torch_module.n_layers):
        params["params"][f"TransformerBlock_{layer_id}"] = block_params_from_module(
            torch_module.layers[layer_id]
        )
    return params
