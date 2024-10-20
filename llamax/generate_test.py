import unittest

import jax
import jax.numpy as jnp

import numpy as np

from transformers import LlamaTokenizer


import llamax
from llamax import model
from llamax import generate


class TestTextGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up any necessary resources that will be shared across tests."""
        # Initialize tokenizer
        cls.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        # Initialize model
        cls.config = llamax.ModelArgs(
            dim=128,
            n_layers=1,
            n_heads=4,
            vocab_size=cls.tokenizer.vocab_size,
            max_batch_size=4,
            max_seq_len=100,
        )

        cls.model = model.Transformer(cls.config)
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
        cls.params = cls.model.init(key, dummy_input, 0)

        # Common parameters
        cls.max_length = 20
        cls.temperature = 1.0
        cls.seed = 42
        cls.test_prompt = "Hello, world!"

    def setUp(self):
        """Set up any necessary resources for each individual test."""
        self.key = jax.random.PRNGKey(self.seed)

    def test_basic_generation(self):
        """Test basic text generation functionality."""
        generated = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            seed=self.seed,
        )

        self.assertIsInstance(generated, str)
        self.assertTrue(len(generated) > 0)
        self.assertTrue(self.test_prompt in generated)

    def test_temperature_variation(self):
        """Test that different temperatures produce different outputs."""
        high_temp = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=2.0,
            seed=self.seed,
        )

        low_temp = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=0.1,
            seed=self.seed,
        )

        self.assertNotEqual(high_temp, low_temp)

    def test_reproducibility(self):
        """Test that the same seed produces the same output."""
        generated1 = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            seed=self.seed,
        )

        generated2 = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            seed=self.seed,
        )

        self.assertEqual(generated1, generated2)

    def test_max_length_constraint(self):
        """Test that max_length parameter is respected."""
        max_tokens = 10
        generated = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=max_tokens,
            temperature=self.temperature,
            seed=self.seed,
        )

        # Get token count of generated text
        tokens = self.tokenizer.encode(generated)
        prompt_tokens = self.tokenizer.encode(self.test_prompt)

        self.assertLessEqual(
            len(tokens),
            len(prompt_tokens) + max_tokens,
            "Generated text length exceeds max_length + prompt length",
        )

    def test_top_k_sampling(self):
        """Test that top_k parameter affects generation."""
        small_k = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=2,
            seed=self.seed,
        )

        large_k = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            top_k=50,
            seed=self.seed,
        )

        self.assertNotEqual(small_k, large_k)

    def test_top_p_sampling(self):
        """Test that top_p parameter affects generation."""
        small_p = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=0.1,
            seed=self.seed,
        )

        large_p = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=self.test_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            top_p=0.9,
            seed=self.seed,
        )

        self.assertNotEqual(small_p, large_p)

    def test_empty_prompt(self):
        """Test generation with empty prompt."""
        generated = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt="",
            max_length=self.max_length,
            temperature=self.temperature,
            seed=self.seed,
        )

        self.assertIsInstance(generated, str)

    def test_batch_consistency(self):
        """Test that batched generation produces consistent results."""
        # Encode prompt
        input_ids = jnp.array(
            [self.tokenizer.encode(self.test_prompt)], dtype=jnp.int32
        )

        # Generate single sequence
        output1, mask1 = generate.generate_text(
            params=self.params,
            model=self.model,
            input_ids=input_ids,
            key=self.key,
            max_length=self.max_length,
            temperature=self.temperature,
        )

        # Generate batch with same sequence duplicated
        batch_input = jnp.tile(input_ids, (2, 1))
        output2, mask2 = generate.generate_text(
            params=self.params,
            model=self.model,
            input_ids=batch_input,
            key=self.key,
            max_length=self.max_length,
            temperature=self.temperature,
        )

        np.testing.assert_array_equal(output2[0], output2[1])

    def test_start_pos_handling(self):
        """Test that the model correctly handles the start_pos parameter."""
        input_ids = jnp.array(
            [self.tokenizer.encode(self.test_prompt)], dtype=jnp.int32
        )

        # Get outputs with different start positions
        outputs1 = self.model.apply({"params": self.params}, input_ids, start_pos=0)
        outputs2 = self.model.apply({"params": self.params}, input_ids, start_pos=2)

        # Check shapes
        self.assertEqual(outputs1.shape[1], input_ids.shape[1])
        self.assertEqual(outputs2.shape[1], input_ids.shape[1] - 2)

    def test_long_prompt(self):
        """Test generation with a longer prompt."""
        long_prompt = " ".join([self.test_prompt] * 5)  # Repeat prompt 5 times
        generated = generate.generate_text(
            params=self.params,
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=long_prompt,
            max_length=self.max_length,
            temperature=self.temperature,
            seed=self.seed,
        )

        self.assertIsInstance(generated, str)
        self.assertTrue(long_prompt in generated)


if __name__ == "__main__":
    unittest.main()
