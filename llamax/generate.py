"""
Example usage:
# Initialize your model and tokenizer
model = YourFlaxModel()
tokenizer = YourTokenizer.from_pretrained("model_name")
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 1)))

# Generate text
prompt = "Once upon a time"
generated_text = generate_text(
    params=params,
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_length=100,
    temperature=0.7,
    top_k=50,
    top_p=0.9
)
print(generated_text)
"""

import functools as ft
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

import llamax


def apply_top_k(logits: jax.Array, *, top_k: int) -> jax.Array:
    if top_k is not None:
        _, top_k_indices = jax.lax.top_k(logits, top_k)

        # Create indices for all dimensions
        batch_size = logits.shape[0]
        row_indices = jnp.arange(batch_size)[:, None]
        # Broadcast row_indices to match top_k_indices shape
        row_indices = jnp.broadcast_to(row_indices, top_k_indices.shape)

        # Create mask initialized to negative infinity
        mask = jnp.full_like(logits, float("-inf"))

        # Use advanced indexing to set the top-k positions to their original values
        mask = mask.at[row_indices, top_k_indices].set(0)

        return jnp.where(mask == 0, logits, float("-inf"))
    return logits


@ft.partial(jax.jit, static_argnames=["p"])
def apply_top_p(logits: jax.Array, *, p: int, temperature: float = 1.0) -> jax.Array:
    logits = logits / temperature
    sorted_logits, sorted_indices = jax.lax.top_k(logits, logits.shape[-1])
    probs = jax.nn.softmax(sorted_logits)
    cumulative_probs = jnp.cumsum(probs, axis=-1)
    mask = (cumulative_probs <= p) | (jnp.arange(logits.shape[-1]) == 0)

    # We need to transform the indices so that they map onto the flat
    # equivalent in order to work with jnp.take.
    offset = (sorted_indices.shape[1] * jnp.arange(sorted_indices.shape[0]))[:, None]
    offset_indices = sorted_indices + offset
    original_mask = jnp.take(mask, offset_indices)
    return jnp.where(original_mask, logits, float("-inf"))


@ft.partial(jax.jit, static_argnames=("model", "max_length", "top_k", "top_p"))
def generate_tokens(
    params: Dict,
    model: nn.Module,
    input_ids: jax.Array,
    key: jax.Array,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
) -> Tuple[jax.Array, jax.Array]:
    """
    JIT-compiled function for text generation using JAX structured control flow.

    Args:
        params: Model parameters
        model: Flax model instance
        input_ids: Initial token ids (batch_size, seq_len)
        key: PRNG key for sampling
        max_length: Maximum length of generated sequence
        temperature: Sampling temperature
        top_k: Number of highest probability tokens to consider
        top_p: Cumulative probability cutoff for nucleus sampling

    Returns:
        Generated token arrays with shape (batch_size, max_length) and type jnp.int32.
    """
    batch_size, init_seq_len = input_ids.shape

    # Initialize output arrays
    output_tokens = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
    output_tokens = output_tokens.at[:, :init_seq_len].set(input_ids)
    attention_mask = llamax.make_causal_mask(max_length)
    start_pos = 0

    @struct.dataclass
    class LoopState:
        i: int
        tokens: jax.Array
        key: jax.Array

    def sample_token(
        logits: jax.Array, temperature: float, key: jax.Array
    ) -> jax.Array:
        """Sample next token from logits with temperature and optional top-k/p."""
        # Apply temperature
        logits = logits / jnp.maximum(temperature, 1e-6)

        # Apply top-k and top-p filtering
        logits = apply_top_k(logits, top_k=top_k)
        logits = apply_top_p(logits, p=top_p)

        # Sample from the modified distribution
        next_token = jax.random.categorical(key, logits, axis=-1)
        return next_token

    def generation_loop(state):
        """Single step of the generation loop."""
        # Get model output for current sequence
        outputs = model.apply(
            params,
            state.tokens,
            start_pos,
            # Because the mask is causal, we can just use it
            # as-is, and then only look at the valid logits.
            # Causality will mask out the problematic outputs.
            attention_mask,
        )
        next_token_logits = outputs[:, state.i, :]

        # Sample next token
        cur_key, sample_key = jax.random.split(state.key)
        next_token = sample_token(next_token_logits, temperature, sample_key)

        # Update output arrays
        tokens = state.tokens.at[:, state.i].set(next_token)

        return LoopState(state.i + 1, tokens, cur_key)

    def cond_fn(state):
        """Condition for continuing generation."""
        not_max_len = state.i < max_length - init_seq_len
        # TODO(finbarrtimbers): Add support to check if all the sequences are EOS.
        return not_max_len

    # Initialize loop state
    init_state = LoopState(init_seq_len, output_tokens, key)

    # Run the generation loop
    final_state = jax.lax.while_loop(cond_fn, generation_loop, init_state)

    # Return final output and attention mask
    # TODO(finbarrtimbers): Include EOS in attention mask.
    return final_state.tokens, attention_mask


def generate_text(
    params: Dict,
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
    seed: int = 42,
) -> str:
    """
    High-level text generation function that handles tokenization and decoding.
    """
    # Encode the prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = jnp.array(input_ids)[None, :]  # Add batch dimension

    # Set up PRNG key
    key = jax.random.PRNGKey(seed)

    # Generate tokens
    generated_ids, _ = generate_tokens(
        params=params,
        model=model,
        input_ids=input_ids,
        key=key,
        # We add one to account for <bos>.
        max_length=max_length + 1,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Extract valid tokens using attention mask
    return tokenizer.decode(generated_ids[0, 1:])
