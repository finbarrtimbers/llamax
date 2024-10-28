import unittest
import llamax

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import torch
from parameterized import parameterized

from llamax import model
from llamax import reference_model_torch

SMALL_MODEL_CONFIG = llamax.ModelArgs(
    vocab_size=1_000,
    dim=128,
    n_layers=1,
    n_heads=4,
)
LLAMA_32_1B_CONFIG = llamax.ModelArgs(
    dim=2048,
    ffn_dim_multiplier=1.5,
    multiple_of=256,
    n_heads=32,
    n_kv_heads=8,
    n_layers=16,
    vocab_size=128256,
    norm_eps=1e-5,
    # use_scaled_rope=True,
)

# This is the number of params in Llama 3.2 1B.
NUM_WEIGHTS = 1_498_482_688

# These are all length 128 after the prompt, and
KNOWN_TEXT = {
    "llama_3.2_1B": {
        "Hello, world!": (
            " I’m a 20-something year old who loves to write. I’m a huge fan "
            "of the Harry Potter series, and I’m also a huge fan"
        ),
    }
}


# We use numbers that are mutually prime and >1 to ensure no weirdness.
BATCH_SIZE = 2
SEQ_LEN = 3
MODEL_DIM = 5


def assert_modules_output_same_code(
    inputs: np.ndarray, flax_module: nn.Module, torch_module: torch.nn.Module
):
    # Convert input to appropriate types
    jax_inputs = jnp.array(inputs)
    torch_inputs = torch.tensor(inputs)

    # Get outputs from both models
    jax_output = flax_module(jax_inputs)
    torch_output = torch_module(torch_inputs)

    # Convert outputs to numpy arrays
    jax_output_np = np.array(jax_output)
    torch_output_np = torch_output.detach().numpy()

    # First, check that the output is finite:
    assert np.isfinite(jax_output_np).all(), f"{jax_output_np=}"
    assert np.isfinite(torch_output_np).all(), f"{torch_output_np=}"

    # Check if outputs are equal within a small tolerance
    np.testing.assert_array_almost_equal(jax_output_np, torch_output_np)


class TestModelEquivalence(unittest.TestCase):
    def setUp(self):
        # Set both JAX and PyTorch to use CPU
        jax.config.update("jax_platform_name", "cpu")

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        self.config = SMALL_MODEL_CONFIG
        self.freqs = model.precompute_freqs_cis(
            dim=self.config.dim // self.config.n_heads,
            end=self.config.max_seq_len * 2,
            theta=self.config.rope_theta,
        )

    @parameterized.expand(
        [
            (4, 4096),
        ]
    )
    def test_precompute_freqs_cis_equality(
        self, dim, end, theta=SMALL_MODEL_CONFIG.rope_theta
    ):
        # Get outputs from both functions
        torch_output = reference_model_torch.precompute_freqs_cis(dim, end, theta)
        jax_output = model.precompute_freqs_cis(dim, end, theta)

        # Compare real and imaginary parts separately within a tolerance
        np.testing.assert_allclose(torch_output.numpy(), jax_output)

    def test_rmsnorm_matches(self):
        inputs = np.random.randn(BATCH_SIZE, SEQ_LEN, MODEL_DIM)
        flax_rmsnorm = model.RMSNorm(dim=MODEL_DIM, eps=self.config.norm_eps)
        params = flax_rmsnorm.init(jax.random.key(0), inputs)
        torch_rmsnorm = reference_model_torch.RMSNorm(
            dim=MODEL_DIM, eps=self.config.norm_eps
        ).to(torch.float64)
        params = {"params": model.rmsnorm_params_from_torch(torch_rmsnorm)}

        def flax_module(x):
            return jax.jit(flax_rmsnorm.apply)(params, x)

        assert_modules_output_same_code(inputs, flax_module, torch_rmsnorm)

    def test_feedforward_matches(self):
        inputs = np.random.randn(BATCH_SIZE, SEQ_LEN, self.config.dim)
        flax_ffn = model.FeedForward(
            dim=self.config.dim,
            hidden_dim=4 * self.config.dim,
            multiple_of=self.config.multiple_of,
            ffn_dim_multiplier=self.config.ffn_dim_multiplier,
        )
        torch_ffn = reference_model_torch.FeedForward(
            dim=self.config.dim,
            hidden_dim=4 * self.config.dim,
            multiple_of=self.config.multiple_of,
            ffn_dim_multiplier=self.config.ffn_dim_multiplier,
        ).to(torch.float64)
        params = {"params": model.feedforward_params_from_torch(torch_ffn)}

        def flax_module(x):
            return jax.jit(flax_ffn.apply)(params, x)

        assert_modules_output_same_code(inputs, flax_module, torch_ffn)

    def test_attention_matches(self):
        inputs = np.random.randn(BATCH_SIZE, SEQ_LEN, self.config.dim)
        torch_attn = reference_model_torch.Attention(self.config).double()
        flax_attn = model.Attention(
            n_heads=self.config.n_heads,
            dim=self.config.dim,
            max_batch_size=self.config.max_batch_size,
            max_seq_len=self.config.max_seq_len,
            n_kv_heads=self.config.n_kv_heads,
        )
        start_pos = 0
        mask = llamax.make_causal_mask(SEQ_LEN)
        print(f"{mask=}")
        print(f"{jnp.any(jnp.all(~mask, axis=-1))=}")
        freqs_cis = self.freqs[start_pos : start_pos + SEQ_LEN]
        params = {"params": model.attention_params_from_torch(torch_attn)}

        def flax_module(x):
            return jax.jit(flax_attn.apply)(
                params, x, start_pos=start_pos, freqs_cis=freqs_cis, mask=mask
            )

        def torch_module(x):
            return torch_attn(
                x,
                start_pos,
                torch.from_numpy(np.array(freqs_cis)),
                torch.from_numpy(np.array(mask)),
            )

        with torch.no_grad():
            assert_modules_output_same_code(inputs, flax_module, torch_module)

    def test_transformer_block(self):
        inputs = np.random.randn(BATCH_SIZE, SEQ_LEN, self.config.dim)
        torch_block = reference_model_torch.TransformerBlock(
            layer_id=0, args=self.config
        ).double()
        flax_block = model.TransformerBlock(layer_id=0, config=self.config)
        start_pos = 0
        mask = jnp.full((SEQ_LEN, SEQ_LEN), float("-inf"))

        # Standard causal mask.
        mask = jnp.triu(mask, k=1)
        freqs_cis = self.freqs[start_pos : start_pos + SEQ_LEN]
        params = {"params": model.block_params_from_module(torch_block)}

        def torch_module(x):
            return torch_block(
                x,
                start_pos,
                torch.from_numpy(np.array(freqs_cis)),
                torch.from_numpy(np.array(mask)),
            )

        def flax_module(x):
            return flax_block.apply(params, x, start_pos, freqs_cis, mask)

        assert_modules_output_same_code(inputs, flax_module, torch_module)

    def test_forward_pass(self):
        inputs = np.random.randint(
            0, self.config.vocab_size, size=(BATCH_SIZE, SEQ_LEN)
        )
        torch_model = reference_model_torch.Transformer(self.config)

        flax_model = model.Transformer(config=self.config)
        start_pos = 0

        mask = llamax.make_causal_mask(SEQ_LEN)

        params = model.transformer_params_from_module(torch_model)

        def torch_module(x):
            return torch_model(x, start_pos)

        def flax_module(x):
            return flax_model.apply(params, x, start_pos, mask)

        assert_modules_output_same_code(inputs, flax_module, torch_module)

    def test_model_behavior_with_padding(self):
        inputs = np.random.randint(
            0, self.config.vocab_size, size=(BATCH_SIZE, SEQ_LEN + 1)
        )
        flax_model = model.Transformer(config=self.config)
        start_pos = 0

        mask = llamax.make_causal_mask(SEQ_LEN + 1)

        # Shape is (SEQ_LEN + 1, SEQ_LEN + 1)
        # Nothing should attend to the padding token.
        mask = mask.at[:, -1].set(float("-inf"))

        # The padding token shouldn't attend to anything.
        mask = mask.at[-1, :].set(float("-inf"))

        params = flax_model.init(jax.random.PRNGKey(0), inputs, start_pos, mask)
        apply_fn = jax.jit(flax_model.apply)
        unpadded_output = apply_fn(params, inputs[:, :-1], start_pos, mask[:-1, :-1])
        padded_output = apply_fn(params, inputs, start_pos, mask)

        # And, of course, we throw away the last entry, as it corresponds to the output.
        np.testing.assert_array_almost_equal(unpadded_output, padded_output[:, :-1, :])

    @unittest.skip("This isn't actually implemented, and is just a stub from Claude.")
    def test_gradient_computation(self):
        # Create random input data
        input_data = np.random.randn(1, 3, 224, 224)

        # JAX gradient computation
        def jax_loss_fn(params):
            output = self.flax_model.apply(
                {"params": params}, jnp.array(input_data, dtype=jnp.float64)
            )
            return jnp.mean(output)

        jax_grad = jax.grad(jax_loss_fn)(self.flax_model.params)

        # PyTorch gradient computation
        torch_input = torch.tensor(input_data, dtype=torch.float64, requires_grad=True)
        torch_output = self.torch_model(torch_input)
        torch_loss = torch.mean(torch_output)
        torch_loss.backward()

        # Compare gradients for a specific layer (e.g., first layer weights)
        jax_grad_np = np.array(jax_grad["layer1"]["kernel"])
        torch_grad_np = self.torch_model.layer1.weight.grad.numpy()

        np.testing.assert_allclose(jax_grad_np, torch_grad_np, rtol=1e-5, atol=1e-5)


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        # Set both JAX and PyTorch to use CPU
        jax.config.update("jax_platform_name", "cpu")

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        self.config = LLAMA_32_1B_CONFIG
        self.freqs = model.precompute_freqs_cis(
            dim=self.config.dim // self.config.n_heads,
            end=self.config.max_seq_len * 2,
            theta=self.config.rope_theta,
        )
        self.torch_model = reference_model_torch.Transformer(self.config)
        checkpoint = torch.load(
            "/data/llama-3.2-1B/consolidated.00.pth",
            map_location="cpu",
            weights_only=True,
        )
        jax.tree.map(
            lambda x, y: np.testing.assert_array_equal(x.shape, y.shape),
            dict(self.torch_model.state_dict()),
            checkpoint,
        )
        self.torch_model.load_state_dict(checkpoint)
        self.params = model.transformer_params_from_module(self.torch_model)
        flax_model = model.Transformer(self.config)
        self.apply_fn = jax.jit(flax_model.apply)

    def test_logits_match(self):
        inputs = np.random.randint(
            0, self.config.vocab_size, size=(BATCH_SIZE, SEQ_LEN)
        )
        mask = llamax.make_causal_mask(SEQ_LEN)
        torch_logits = self.torch_model(
            torch.from_numpy(inputs),
            start_pos=0,
            mask=torch.from_numpy(np.array(mask, copy=True)),
        )
        torch_logits = np.array(torch_logits, copy=True)
        flax_logits = self.apply_fn(self.params, inputs, start_pos=0, mask=mask)

        # First, we check that both tensors have no nans/infs. Otherwise, we get
        # spurious tests passing.
        self.assertTrue(np.isfinite(torch_logits).all())
        self.assertTrue(np.isfinite(flax_logits).all())
        np.testing.assert_array_almost_equal(torch_logits, flax_logits)

    @unittest.skip("This is slow so we don't run it by default.")
    def test_checkpoint_matches_torch(self):
        torch_model = reference_model_torch.Transformer(self.config)
        checkpoint = torch.load(
            "/data/llama-3.2-1B/consolidated.00.pth",
            map_location="cpu",
            weights_only=True,
        )
        jax.tree.map(
            lambda x, y: np.testing.assert_array_equal(x.shape, y.shape),
            dict(torch_model.state_dict()),
            checkpoint,
        )
        self.torch_model.load_state_dict(checkpoint)

        def count_leaves(tree) -> int:
            shape_dict = jax.tree.map(lambda x: jnp.prod(jnp.array(x.shape)), tree)
            return jnp.sum(jnp.array(list(shape_dict.values())))

        self.assertEqual(
            count_leaves(torch_model.state_dict()), count_leaves(checkpoint)
        )
        self.assertEqual(count_leaves(checkpoint), NUM_WEIGHTS)
        torch_model.load_state_dict(checkpoint)

    def test_num_parameters_match(self):
        flax_model = model.Transformer(config=self.config)
        inputs = np.random.randint(
            0, self.config.vocab_size, size=(BATCH_SIZE, SEQ_LEN)
        )
        mask = llamax.make_causal_mask(SEQ_LEN)
        params = flax_model.init(jax.random.PRNGKey(0), inputs, start_pos=0, mask=mask)
        num_params = jnp.sum(
            jnp.array(
                jax.tree.map(
                    lambda x: jnp.prod(jnp.array(x.shape)), jax.tree.flatten(params)[0]
                )
            )
        )
        self.assertEqual(num_params, NUM_WEIGHTS)


if __name__ == "__main__":
    unittest.main(failfast=True)
