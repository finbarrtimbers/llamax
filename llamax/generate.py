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

import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Optional, Tuple, Dict

import functools as ft


@ft.partial(jax.jit, static_argnames=("model", "max_length", "top_k", "top_p"))
def generate_tokens(
    params: Dict,
    model: nn.Module,
    input_ids: jnp.ndarray,
    key: jnp.ndarray,
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = 50,
    top_p: Optional[float] = 0.9,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
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
        Tuple of (generated token array, attention mask)
    """
    batch_size, init_seq_len = input_ids.shape

    # Initialize output arrays
    output_tokens = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
    output_tokens = output_tokens.at[:, :init_seq_len].set(input_ids)
    attention_mask = jnp.zeros((batch_size, max_length), dtype=jnp.int32)
    attention_mask = attention_mask.at[:, :init_seq_len].set(1)

    def sample_token(
        logits: jnp.ndarray, temperature: float, key: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Sample next token from logits with temperature and optional top-k/p."""
        # Apply temperature
        logits = logits / jnp.maximum(temperature, 1e-6)

        def apply_top_k(logits_vals):
            if top_k is not None:
                _, top_k_indices = jax.lax.top_k(logits_vals, top_k)
                mask = jnp.zeros_like(logits_vals)
                mask = mask.at[top_k_indices].set(1)
                return jnp.where(mask, logits_vals, -1e10)
            return logits_vals

        def apply_top_p(logits_vals):
            if top_p is not None:
                sorted_logits = jnp.sort(logits_vals)[::-1]
                cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits))
                mask = cumulative_probs <= top_p
                mask = mask.at[0].set(True)
                return jnp.where(mask, logits_vals, -1e10)
            return logits_vals

        # Apply top-k and top-p filtering
        logits = apply_top_k(logits)
        logits = apply_top_p(logits)

        # Sample from the modified distribution
        next_token = jax.random.categorical(key, logits, axis=-1)
        return next_token, key

    def generation_loop(state):
        """Single step of the generation loop."""
        i, cur_output, cur_mask, cur_key = state

        # Get model output for current sequence
        valid_len = jnp.sum(cur_mask, axis=1)
        cur_input = jax.lax.dynamic_slice_in_dim(
            cur_output,
            0,
        )
        cur_input = cur_output[:, : valid_len[0]]  # Use only valid tokens
        outputs = model.apply({"params": params}, cur_input)
        next_token_logits = outputs[:, -1, :]

        # Sample next token
        cur_key, sample_key = jax.random.split(cur_key)
        next_token, _ = sample_token(next_token_logits, temperature, sample_key)

        # Update output arrays
        cur_output = cur_output.at[:, valid_len[0]].set(next_token)
        cur_mask = cur_mask.at[:, valid_len[0]].set(1)

        return i + 1, cur_output, cur_mask, cur_key

    def cond_fn(state):
        """Condition for continuing generation."""
        i, _, cur_mask, _ = state
        not_max_len = i < max_length - init_seq_len
        # Check if any sequence has not generated an EOS token
        has_active = jnp.any(cur_mask.sum(axis=1) < max_length)
        return jnp.logical_and(not_max_len, has_active)

    # Initialize loop state
    init_state = (0, output_tokens, attention_mask, key)

    # Run the generation loop
    final_state = jax.lax.while_loop(cond_fn, generation_loop, init_state)

    # Return final output and attention mask
    return final_state[1], final_state[2]


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
    generated_ids, attention_mask = generate_tokens(
        params=params,
        model=model,
        input_ids=input_ids,
        key=key,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    # Extract valid tokens using attention mask
    valid_tokens = generated_ids[0, : jnp.sum(attention_mask[0])]

    # Decode generated tokens
    generated_text = tokenizer.decode(valid_tokens)
    return generated_text
