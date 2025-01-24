import os
import unittest

import jax
import jax.numpy as jnp
import numpy as np
import transformers
import torch

import llamax
from llamax import generate, model

MAX_LENGTH = 32

# Dict mapping models to prompts to known good text (generated from the API).
# We use a maximum length of 32 tokens.
GOLDEN_TEXT = {
    "deepseek-ai/deepseek-coder-1.3b-base": {
        "def fibonacci(n):": (
            "def fibonacci(n):\n    if n == 0:\n        return 0\n    elif n"
            " == 1:\n        return 1\n    else:\n        return fib"
        ),
    }
}


class TestTextGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up any necessary resources that will be shared across tests."""
        # Initialize tokenizer
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B", token=os.environ["HF_TOKEN"]
        )

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
            self.params,
            self.model,
            self.tokenizer,
            self.test_prompt,
            max_length=MAX_LENGTH,
        )

        self.assertIsInstance(generated, str)
        self.assertTrue(generated.startswith(self.test_prompt))

    @unittest.skip("currently failing, as we're using bad, random, weights.")
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

    @unittest.skip("currently failing, as we're using bad, random, weights.")
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

    @unittest.skip("currently failing, as we're using bad, random, weights.")
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

        # Generate batch with same sequence duplicated
        batch_input = jnp.tile(input_ids, (2, 1))
        output, _ = generate.generate_tokens(
            params=self.params,
            model=self.model,
            input_ids=batch_input,
            key=self.key,
            max_length=self.max_length,
            temperature=self.temperature,
        )
        np.testing.assert_array_equal(output[0], output[1])

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


# TODO(finbarr): Figure out how to parameterize the tests so we don't
# repeat as much code.'
class TestDeepSeekCoderGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up the official model and tokenizer for comparison."""
        # Initialize official model and tokenizer
        cls.official_model_name = "deepseek-ai/deepseek-coder-1.3b-base"

        cls.official_tokenizer = transformers.AutoTokenizer.from_pretrained(
            cls.official_model_name
        )
        cls.official_model = transformers.AutoModelForCausalLM.from_pretrained(
            cls.official_model_name
        )

        # Initialize your implementation
        cls.config = llamax.ModelArgs(
            dim=2048,  # Update with correct model dimensions
            n_layers=24,  # Update with correct number of layers
            n_heads=16,  # Update with correct number of heads
            vocab_size=cls.official_tokenizer.vocab_size,
            max_batch_size=1,
            max_seq_len=2048,
        )

        cls.model = model.Transformer(cls.config)
        key = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, 1), dtype=jnp.int32)
        cls.params = cls.model.init(key, dummy_input, 0)

        # Load the pretrained weights into your implementation
        # You'll need to implement this conversion
        # cls.params = convert_deepseek_weights_to_llamax(
        #     cls.official_model.state_dict())

    def test_simple_completion(self):
        """Test a simple code completion task."""
        prompt = "def fibonacci(n):"

        # Generate with official model
        inputs = self.official_tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            official_output = self.official_model.generate(
                inputs["input_ids"],
                max_new_tokens=32,
                do_sample=False,
            )
        official_text = self.official_tokenizer.decode(
            official_output[0], skip_special_tokens=True
        )
        golden_text = GOLDEN_TEXT[self.official_model_name][prompt]
        golden_text = golden_text[: len(official_text)]
        self.assertEqual(len(official_text), len(golden_text))
        self.assertEqual(official_text, golden_text)

        # # Generate with your implementation
        # llamax_text = generate.generate_text(
        #     self.params,
        #     self.model,
        #     self.official_tokenizer,
        #     prompt,
        #     max_length=100,
        #     temperature=0.7,
        #     top_p=0.95,
        # )

        # self.assertEqual(official_text, llamax_text)


class TestSamplingFunctions(unittest.TestCase):
    def setUp(self):
        # Common test data that will be used across multiple tests
        self.basic_logits = jnp.array([[1.0, 2.0, 3.0, 4.0]])
        self.batch_logits = jnp.array([[1.0, 2.0, 3.0, 4.0], [0.5, 2.5, 1.5, 3.5]])

    def test_apply_top_k_basic(self):
        """Test basic case with k=2"""
        result = generate.apply_top_k(self.basic_logits, top_k=2)
        expected = jnp.array([[float("-inf"), float("-inf"), 3.0, 4.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_top_k_batch(self):
        """Test batch case with k=2"""
        result = generate.apply_top_k(self.batch_logits, top_k=2)
        expected = jnp.array(
            [
                [float("-inf"), float("-inf"), 3.0, 4.0],
                [float("-inf"), 2.5, float("-inf"), 3.5],
            ]
        )
        np.testing.assert_array_almost_equal(result, expected)

    def test_apply_top_k_none(self):
        """Test when top_k is None"""
        result = generate.apply_top_k(self.basic_logits, top_k=None)
        np.testing.assert_array_equal(result, self.basic_logits)

    def test_apply_top_k_equal_values(self):
        """Test behavior with equal values"""
        logits = jnp.array([[2.0, 2.0, 1.0, 2.0]])
        result = generate.apply_top_k(logits, top_k=2)
        # Should keep two of the equal values
        mask = result != float("-inf")
        self.assertEqual(jnp.sum(mask), 2)

    def test_apply_top_p_basic(self):
        """Test basic case with p=0.5"""
        result = generate.apply_top_p(self.basic_logits, p=0.5)
        # After softmax, only the largest value(s) should remain
        expected_mask = result != float("-inf")
        self.assertGreaterEqual(jnp.sum(expected_mask), 1)

    def test_apply_top_p_batch(self):
        """Test batch case with p=0.7"""
        result = generate.apply_top_p(self.batch_logits, p=0.7)
        # Check that some values are kept and some are masked
        mask = result != float("-inf")
        self.assertTrue(jnp.all(jnp.sum(mask, axis=1) >= 1))

    def test_apply_top_p_extreme_values(self):
        """Test with p=0.0 and p=1.0"""
        # With p=0.0, should keep only the highest value
        result_0 = generate.apply_top_p(self.basic_logits, p=0.0)
        self.assertEqual(jnp.sum(result_0 != float("-inf")), 1)

        # With p=1.0, should keep all values
        result_1 = generate.apply_top_p(self.basic_logits, p=1.0)
        np.testing.assert_array_equal(result_1, self.basic_logits)

    def test_numerical_stability(self):
        """Test with very large and very small numbers"""
        logits = jnp.array([[-1e10, 0.0, 1e10]])

        # Test top-k
        result_k = generate.apply_top_k(logits, top_k=1)
        self.assertEqual(jnp.sum(jnp.where(jnp.isfinite(result_k), 1, 0)), 1)

        # Test top-p
        result_p = generate.apply_top_p(logits, p=0.5)
        np.testing.assert_array_equal(result_p, [[float("-inf"), float("-inf"), 1e10]])

    def test_top_k_shape_preservation(self):
        """Test that output shapes match input shapes"""
        # Test 1D case
        result_1d = generate.apply_top_k(self.basic_logits, top_k=2)
        self.assertEqual(result_1d.shape, self.basic_logits.shape)

        # Test 2D case
        result_2d = generate.apply_top_k(self.batch_logits, top_k=2)
        self.assertEqual(result_2d.shape, self.batch_logits.shape)

    def test_top_p_shape_preservation(self):
        """Test that output shapes match input shapes"""
        # Test 1D case
        result_1d = generate.apply_top_p(self.basic_logits, p=0.5)
        self.assertEqual(result_1d.shape, self.basic_logits.shape)

        # Test 2D case
        result_2d = generate.apply_top_p(self.batch_logits, p=0.5)
        self.assertEqual(result_2d.shape, self.batch_logits.shape)


if __name__ == "__main__":
    unittest.main()
