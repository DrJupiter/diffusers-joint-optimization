from sde_redefined import SDE
import torch
import numpy as np
import math
import jax.numpy as jnp

# TODO (Klaus) : Add mxin and config so we can save properly
class TorchSDE(SDE):

    def __init__(self, variable, drift, diffusion, diffusion_matrix, initial_variable_value=0, max_variable_value = math.inf, min_sample_value=1e-4, module='jax', model_target="epsilon", drift_integral_form=False, diffusion_integral_form=False, diffusion_integral_decomposition='cholesky', drift_diagonal_form=True, diffusion_diagonal_form=True, diffusion_matrix_diagonal_form=True):
        super().__init__(variable, drift, diffusion, diffusion_matrix, initial_variable_value, max_variable_value, module, model_target, drift_integral_form, diffusion_integral_form, diffusion_integral_decomposition, drift_diagonal_form, diffusion_diagonal_form, diffusion_matrix_diagonal_form)
        self.min_sample_value = min_sample_value
    
    def sample(self, timestep, initial_data, key, device='cuda'):

        # TODO (TECHNICALLY) there is no need to not take in a jax numpy array and then at the final step convert it to a tensor.
        batch_size = timestep.shape[0]
        original_shape = initial_data.shape
        jax_timestep = jnp.array(timestep.reshape(-1).numpy(force=True))
        jax_initial_data = jnp.array(initial_data.reshape(batch_size, -1).numpy(force=True))

        noisy_data, noise = super().sample(jax_timestep, jax_initial_data, key)
        return torch.from_numpy(np.array(noisy_data.reshape(original_shape))).to(device), torch.from_numpy(np.array(noise.reshape(original_shape))).to(device) 
    
    def set_timesteps(self,num_inference_steps, device):
        self.timesteps = torch.from_numpy(np.linspace(self.min_sample_value, self.max_variable_value, num_inference_steps)[::-1].copy()).to(device)

    def step(self, model_output, timestep, data, key, dt, device='cuda'):

        batch_size = data.shape[0]
        original_shape = data.shape
        
        jax_model_output = jnp.array(model_output.reshape(batch_size, -1).numpy(force=True))
        jax_data = jnp.array(data.reshape(batch_size, -1).numpy(force=True))
        jax_timestep = timestep.reshape(-1).numpy(force=True)
        if len(timestep.shape) in [0,1]:
            jax_timestep = jnp.array(jax_timestep.repeat(batch_size)).reshape(-1)
        else:
            jax_timestep = jnp.array(jax_timestep)

        next_sample, sample_derivative, key = super().step(jax_model_output, jax_timestep, jax_data, key, dt)
        return torch.from_numpy(np.array(next_sample.reshape(original_shape))).to(device), torch.from_numpy(np.array(sample_derivative.reshape(original_shape))).to(device), key
