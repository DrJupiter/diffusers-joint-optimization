from sde_redefined_param import SDE_PARAM, SDEDimension
from typing import Any, Tuple
#import torch
import numpy as np
import math
import jax.numpy as jnp
import jax
# TODO (Klaus) : Add mxin and config so we can save properly
import torch

class TorchSDE_PARAM(SDE_PARAM, torch.autograd.Function):
    def __init__(
        self,
    data_dimension: int,
    variable: Any,
    drift_parameters: Any,
    diffusion_parameters: Any,
    drift: Any,
    diffusion: Any,
    diffusion_matrix: Any,
    initial_variable_value: float = 0,
    max_variable_value: float = math.inf,
    module: str = 'jax',
    model_target: str = "epsilon",
    drift_integral_form: bool = False,
    diffusion_integral_form: bool = False,
    diffusion_integral_decomposition: str = 'cholesky',
    drift_dimension: SDEDimension = SDEDimension.DIAGONAL,
    diffusion_dimension: SDEDimension = SDEDimension.DIAGONAL,
    diffusion_matrix_dimension: SDEDimension = SDEDimension.SCALAR):
        super().__init__(variable, drift_parameters, diffusion_parameters, drift, diffusion, diffusion_matrix, initial_variable_value, max_variable_value, module, model_target, drift_integral_form, diffusion_integral_form, diffusion_integral_decomposition, drift_dimension, diffusion_dimension, diffusion_matrix_dimension)
        self.lambdify_symbolic_functions(data_dimension)
    
    def sample(self, *args, **kwargs):
        return self.get_gradient_function().apply(*args, **kwargs)

    def get_gradient_function(self):

        class TorchSDE_PARAM_F(torch.autograd.Function):

            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                return self.v_lambdified_sample(*args, **kwargs)

            @staticmethod 
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
                #t, data, noise, drift, diffusion = inputs
                ctx.save_for_backward(*inputs)
                #return super().setup_context(ctx, inputs, output) 
            
            @staticmethod
            def backward(ctx: Any, *grad_outputs: Any) -> Any:
                return None, None, None, grad_outputs * self.v_lambdified_drift_derivative(ctx.saved_tensors), grad_outputs * self.v_lambdified_diffusion_derivative(ctx.saved_tensors) 
                #return super().backward(ctx, *grad_outputs)

        return TorchSDE_PARAM_F

if __name__ == "__main__":

    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    import sympy
    from sympy import Symbol, Matrix

    sympy.init_printing(use_unicode=True, use_latex=True)
    t = Symbol('t', nonnegative=True, real=True)
    x1,x2,x3,x4,x5 = sympy.symbols("x1 x2 x3 x4 x5", real=True)
    drift_param = Matrix([x1,x2,x3])
    diffusion_param = Matrix([x4,x5])

    drift_param = Matrix([x1,x2,x3])
    diffusion_param = Matrix([x4,x5])
    #drift_param = diffusion_param = sympy.symbols("∅", real=True)

    v_drift_param =  jnp.array([1.,1.,1.])
    v_diffusion_param =  jnp.array([4.,4.])
    #v_drift_param = v_diffusion_param = jnp.array([None])

    n = 2 # dims of problem
    timesteps = jnp.array([0.1,0.4])
    x0 = jnp.ones((len(timesteps), n))*1/2

    model_output = jnp.ones_like(x0)

    z = jax.random.normal(jax.random.PRNGKey(0), x0.shape)

    F = Matrix.diag([sympy.cos(t*drift_param[1])*drift_param[0] + drift_param[2]]*n)
    L = Matrix.diag([sympy.sin(t)*diffusion_param[0]+diffusion_param[1]]*n) # n x n, but to use as diagonal then do 1 x n, 1 x 1 
    


    Q = Matrix.eye(n)
    # Normal test

    z = torch.from_numpy(np.array(z))
    x0 = torch.from_numpy(np.array(x0))
    v_drift_param =  jnp.array([1.,1.,1.])
    v_diffusion_param =  jnp.array([4.,4.])

    sde = TorchSDE_PARAM(2, t, drift_param, diffusion_param, F, L, Q, drift_dimension=SDEDimension.FULL, diffusion_dimension=SDEDimension.FULL, diffusion_matrix_dimension=SDEDimension.FULL)
    sde.lambdify_symbolic_functions(n)
    samples = sde.sample(timesteps, x0, z, v_drift_param, v_diffusion_param)
    print(samples, type(samples))
    
     

""" 
class TorchSDE(SDE_PARAM):

    def __init__(self, variable, drift_parameters, diffusion_parameters, drift, diffusion, diffusion_matrix, data_dimension, initial_variable_value = 0., max_variable_value = math.inf, min_sample_value=1e-4, module='jax', model_target="epsilon", drift_integral_form = False, diffusion_integral_form = False, diffusion_integral_decomposition = 'cholesky', drift_diagonal_form = True, diffusion_diagonal_form = True, diffusion_matrix_diagonal_form = True):
        super().__init__(variable, drift_parameters, diffusion_parameters, drift, diffusion, diffusion_matrix, initial_variable_value, max_variable_value, module, model_target, drift_integral_form, diffusion_integral_form, diffusion_integral_decomposition, drift_diagonal_form, diffusion_diagonal_form, diffusion_matrix_diagonal_form)
        
        self.min_sample_value = min_sample_value

        # Initialize the sde for some generation task
        self.lambdify_symbolic_functions(data_dimension)

    def sample(self, timestep, initial_data, noise, drift_parameters, diffusion_parameters, device="cuda"):
        batch_size = timestep.shape[0]
        original_shape = initial_data.shape
        jax_timestep = jnp.array(timestep.reshape(-1).numpy(force=True))
        jax_initial_data = jnp.array(initial_data.reshape(batch_size, -1).numpy(force=True))
        jax_noise = jnp.array(noise.reshape(batch_size, -1).numpy(force=True))

        jax_drift_parameters = jnp.array(drift_parameters.numpy(force=True))
        jax_diffusion_parameters = jnp.array(diffusion_parameters.numpy(force=True))
        sample = self.v_lambdified_sample(jax_timestep, jax_initial_data, jax_noise, jax_drift_parameters, jax_diffusion_parameters)
        return torch.from_numpy(np.array(sample.reshape(original_shape))).to(device)

    # TODO (KLAUS): REMOVE LEGACY SAMPLE AFTER REFACTOR
    #def sample(self, timestep, initial_data, key, device='cuda'):

    #    # TODO (TECHNICALLY) there is no need to not take in a jax numpy array and then at the final step convert it to a tensor.
    #    batch_size = timestep.shape[0]
    #    original_shape = initial_data.shape
    #    jax_timestep = jnp.array(timestep.reshape(-1).numpy(force=True))
    #    jax_initial_data = jnp.array(initial_data.reshape(batch_size, -1).numpy(force=True))

    #    noisy_data, noise = super().sample(jax_timestep, jax_initial_data, key)
    #    return torch.from_numpy(np.array(noisy_data.reshape(original_shape))).to(device), torch.from_numpy(np.array(noise.reshape(original_shape))).to(device) 
    
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
"""