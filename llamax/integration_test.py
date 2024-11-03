import unittest
import llamax

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import os

import transformers
import torch

from llamax import generate
from llamax import model
from llamax import reference_model_torch

import gc


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

# This is the number of params in Llama 3.2 1B, from Huggingface.
NUM_WEIGHTS = 1_498_482_688
MAX_LENGTH = 32

# These are all length 128 after the prompt, and
KNOWN_TEXT = {
    "meta-llama/Llama-3.1-8B": {
        "Hello, world!": (
            " I’m a 20-something year old who loves to write. I’m a huge fan "
            "of the Harry Potter series, and I’m also a huge fan of"
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


# We intentionally don't support this in Github CI. You'll have to download this
# locally and modify the docker command accordingly.
CHECKPOINT_PATH = "/data/llama-3.2-1B/consolidated.00.pth"


def checkpoint_exists():
    return os.path.isfile(CHECKPOINT_PATH)


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = "Hello, world!"
        cls.model = "meta-llama/Llama-3.1-8B"
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            cls.model, token=os.environ["HF_TOKEN"]
        )

        # Set random seed for reproducibility
        cls.seed = 42
        np.random.seed(cls.seed)
        torch.manual_seed(cls.seed)

        cls.config = LLAMA_32_1B_CONFIG

        # Ensure freqs computation is in float64
        cls.freqs = jnp.array(
            model.precompute_freqs_cis(
                dim=cls.config.dim // cls.config.n_heads,
                end=cls.config.max_seq_len * 2,
                theta=cls.config.rope_theta,
            ),
            dtype=jnp.float64,
        )

        cls.torch_model = reference_model_torch.Transformer(cls.config)

        if checkpoint_exists():
            checkpoint = torch.load(
                CHECKPOINT_PATH,
                map_location="cpu",
                weights_only=True,
            )

            # Convert checkpoint to double precision
            checkpoint = {k: v.double() for k, v in checkpoint.items()}

            jax.tree.map(
                lambda x, y: np.testing.assert_array_equal(x.shape, y.shape),
                dict(cls.torch_model.state_dict()),
                checkpoint,
            )
            cls.torch_model.load_state_dict(checkpoint)

        # Ensure model is in double precision. Probably not needed?
        cls.torch_model.double()

        # Finally, create the jax model:
        cls.params = model.transformer_params_from_module(cls.torch_model)
        cls.flax_model = model.Transformer(cls.config)

    def tearDown(self):
        # Explicitly clear any cached computations and release memory
        jax.clear_caches()
        gc.collect()

    @unittest.skipIf(not checkpoint_exists(), "checkpoint doesn't exist")
    def test_logits_match(self):
        inputs = np.random.randint(
            0, self.config.vocab_size, size=(BATCH_SIZE, SEQ_LEN)
        )
        mask = llamax.make_causal_mask(SEQ_LEN)

        # Ensure inputs and mask are float64
        torch_logits = self.torch_model(
            torch.from_numpy(inputs),
            start_pos=0,
            mask=torch.from_numpy(np.array(mask, dtype=bool)),
        )
        torch_logits = np.array(torch_logits, dtype=np.float64, copy=True)

        # Jitting here doesn't offer any performance benefits, of course, but
        # we want to test the jitted behaviour, as we don't (really) care about
        # the un-jitted behaviour.
        flax_logits = jax.jit(self.flax_model.apply)(
            self.params, inputs, start_pos=0, mask=mask
        )

        torch_argmax = jnp.argmax(torch_logits, axis=-1)
        flax_argmax = jnp.argmax(flax_logits, axis=-1)
        np.testing.assert_array_equal(torch_argmax, flax_argmax)

        # First, we check that both tensors have no nans/infs
        self.assertTrue(np.isfinite(torch_logits).all())
        self.assertTrue(np.isfinite(flax_logits).all())

        # Fails at 6 decimal points with float64, passes at 5.
        np.testing.assert_array_almost_equal(torch_logits, flax_logits, decimal=5)

    @unittest.skipIf(not checkpoint_exists(), "checkpoint doesn't exist")
    def test_known_text_generation(self):
        """Test that the model generates expected tokens for known prompts."""
        text = generate.generate_text(
            params=self.params,
            model=self.flax_model,
            tokenizer=self.tokenizer,
            prompt=self.prompt,
            max_length=32,
            temperature=0.0,
            seed=self.seed,
        )
        self.assertEqual(text, self.prompt + KNOWN_TEXT[self.model][self.prompt])

    @unittest.skip("This is slow so we don't run it by default.")
    def test_checkpoint_matches_torch(self):
        torch_model = reference_model_torch.Transformer(self.config)
        checkpoint = torch.load(
            "/data/llama-3.2-1B/consolidated.00.pth",
            map_location="cpu",
            weights_only=True,
        )
        # Convert checkpoint to double precision
        checkpoint = {k: v.double() for k, v in checkpoint.items()}

        jax.tree.map(
            lambda x, y: np.testing.assert_array_equal(x.shape, y.shape),
            dict(torch_model.state_dict()),
            checkpoint,
        )

        self.torch_model.double()  # Ensure model is in double precision
        self.torch_model.load_state_dict(checkpoint)

        def count_leaves(tree) -> int:
            shape_dict = jax.tree.map(
                lambda x: jnp.prod(jnp.array(x.shape, dtype=jnp.float64)), tree
            )
            return jnp.sum(jnp.array(list(shape_dict.values()), dtype=jnp.float64))

        self.assertEqual(
            count_leaves(torch_model.state_dict()), count_leaves(checkpoint)
        )
        self.assertEqual(count_leaves(checkpoint), NUM_WEIGHTS)
        torch_model.load_state_dict(checkpoint)

    def test_num_parameters_match(self):
        num_params = jnp.sum(
            jnp.array(
                jax.tree.map(
                    lambda x: x.size,
                    jax.tree.flatten(self.params)[0],
                )
            )
        )
        self.assertEqual(num_params, NUM_WEIGHTS)


if __name__ == "__main__":
    unittest.main(failfast=True)
