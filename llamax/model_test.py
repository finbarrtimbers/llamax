import unittest
import jax
import jax.numpy as jnp
import torch
import numpy as np

import llamax
from llamax import model
from llamax import reference_model_torch

import flax.linen as nn

SMALL_MODEL_CONFIG = llamax.ModelArgs(
    dim=128,
    n_layers=1,
)

# We use numbers that are mutually prime and >1 to ensure no weirdness.
BATCH_SIZE = 2
SEQ_LEN = 3
MODEL_DIM = 5



def assert_modules_output_same_code(inputs: np.ndarray, flax_module: nn.Module,
                                    torch_module: torch.nn.Module):
    # Convert input to appropriate types
    jax_inputs = jnp.array(inputs, dtype=jnp.float64)
    torch_inputs = torch.tensor(inputs, dtype=torch.float64)

    # Get outputs from both models
    jax_output = flax_module(jax_inputs)
    torch_output = torch_module(torch_inputs)

    # Convert outputs to numpy arrays
    jax_output_np = np.array(jax_output)
    torch_output_np = torch_output.detach().numpy()

    # Check if outputs are equal within a small tolerance
    np.testing.assert_array_almost_equal(jax_output_np, torch_output_np)



class TestModelEquivalence(unittest.TestCase):

    def setUp(self):
        # Set both JAX and PyTorch to use CPU
        jax.config.update('jax_platform_name', 'cpu')

        # Set random seed for reproducibility
        np.random.seed(42)
        torch.manual_seed(42)
        self.model = model
        self.config = SMALL_MODEL_CONFIG

    def test_rmsnorm_matches(self):
        inputs = np.random.randn(BATCH_SIZE, SEQ_LEN, MODEL_DIM)
        flax_rmsnorm = model.RMSNorm(dim=MODEL_DIM, eps=self.config.norm_eps)
        params = flax_rmsnorm.init(jax.random.key(0), inputs)
        apply_fn = jax.jit(flax_rmsnorm.apply)
        flax_module = lambda x: apply_fn(params, x)
        torch_rmsnorm = reference_model_torch.RMSNorm(dim=MODEL_DIM, eps=self.config.norm_eps).to(torch.float64)

        assert_modules_output_same_code(inputs, flax_module, torch_rmsnorm)

    def test_feedforward_matches(self):
        inputs = np.random.randn(BATCH_SIZE, SEQ_LEN, self.config.dim)
        flax_ffn = model.FeedForward(dim=self.config.dim,
                                     hidden_dim=4*self.config.dim,
                                     multiple_of=self.config.multiple_of,
                                     ffn_dim_multiplier=self.config.ffn_dim_multiplier)
        torch_ffn = reference_model_torch.FeedForward(dim=self.config.dim,
                                                      hidden_dim=4*self.config.dim,
                                                      multiple_of=self.config.multiple_of,
                                                      ffn_dim_multiplier=self.config.ffn_dim_multiplier).to(torch.float64)
        #flax_params = flax_ffn.init(jax.random.PRNGKey(0), inputs)
        params = model.feedforward_params_from_torch(torch_ffn)
        #flax_params_shapes = jax.tree.map(lambda x: x.shape, flax_params)
        params_shapes = jax.tree.map(lambda x: x.shape, params)
        #print(f'{flax_params_shapes=}, {params_shapes=}')
        #jax.tree.map(lambda x, y: np.testing.assert_array_equal(x.shape, y.shape),
        #             flax_params, params)
        #jax_w3_params = params_shapes['params']['w3']['kernel']
        #torch_params = torch_ffn.w3.weight.detach().numpy()
        #np.testing.assert_array_almost_equal(jax_w3_params, torch_params.T)
        flax_module = lambda x: jax.jit(flax_ffn.apply)(params, x)
        assert_modules_output_same_code(
            inputs, flax_module, torch_ffn)

    @unittest.skip("This isn't actually implemented, and is just a stub from Claude.")
    def test_forward_pass(self):
        # Create random input data
        input_data = np.random.randn(1, 3, 224, 224)

        # Convert input to appropriate types
        jax_input = jnp.array(input_data, dtype=jnp.float64)
        torch_input = torch.tensor(input_data, dtype=torch.float64)

        # Get outputs from both models
        jax_output = self.flax_model.apply({'params': self.flax_model.params}, jax_input)
        torch_output = self.torch_model(torch_input)

        # Convert outputs to numpy arrays
        jax_output_np = np.array(jax_output)
        torch_output_np = torch_output.detach().numpy()

        # Check if outputs are equal within a small tolerance
        np.testing.assert_allclose(jax_output_np, torch_output_np, rtol=1e-5, atol=1e-5)

    @unittest.skip("This isn't actually implemented, and is just a stub from Claude.")
    def test_gradient_computation(self):
        # Create random input data
        input_data = np.random.randn(1, 3, 224, 224)

        # JAX gradient computation
        def jax_loss_fn(params):
            output = self.flax_model.apply({'params': params}, jnp.array(input_data, dtype=jnp.float64))
            return jnp.mean(output)

        jax_grad = jax.grad(jax_loss_fn)(self.flax_model.params)

        # PyTorch gradient computation
        torch_input = torch.tensor(input_data, dtype=torch.float64, requires_grad=True)
        torch_output = self.torch_model(torch_input)
        torch_loss = torch.mean(torch_output)
        torch_loss.backward()

        # Compare gradients for a specific layer (e.g., first layer weights)
        jax_grad_np = np.array(jax_grad['layer1']['kernel'])
        torch_grad_np = self.torch_model.layer1.weight.grad.numpy()

        np.testing.assert_allclose(jax_grad_np, torch_grad_np, rtol=1e-5, atol=1e-5)

if __name__ == '__main__':
    unittest.main()