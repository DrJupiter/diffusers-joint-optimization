from sde_redefined_param import SDE_PARAM, SDEDimension
from typing import Any, Tuple
#import torch
import numpy as np
import math
import jax.numpy as jnp
import jax
# TODO (Klaus) : Add mxin and config so we can save properly
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin

batch_matmul = torch.vmap(torch.matmul)

def jax_torch(array, requires_grad=False):
    array = torch.from_numpy(np.array(array))
    array.requires_grad_(requires_grad)
    return array

def torch_jax(tensor):
    return jnp.array(tensor.numpy(force=True))

class TorchSDE_PARAM(SchedulerMixin, ConfigMixin, SDE_PARAM):
    
    #@register_to_config
    def __init__(
        self,
    device: str,
    min_sample_value: float,
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

        self.initialize_parameters(device=device) # TODO (KLAUS): CONSIDER IF IT IS SMARTEST TO LEAVE INITIALIZING TO THE USER ALWAYS
        self.device=device
        self.min_sample_value = min_sample_value
        
    def initialize_parameters(self, drift_parameters=None, diffusion_parameters=None, device="cuda"):

        match self.drift_parameters.shape:
            case (1, x) | (x, 1):
                drift_shape = (x,)        
            case _shape:
                drift_shape = _shape
        
        tensor_drift_parameters = drift_parameters if drift_parameters is not None else torch.nn.init.xavier_normal_(torch.empty(self.drift_parameters.shape)).reshape(drift_shape).to(device) # TODO (KLAUS): Initialize smartly

        match self.diffusion_parameters.shape:
            case (1, x) | (x, 1):
                diffusion_shape = (x,)
            case _shape:
                diffusion_shape = _shape

        tensor_diffusion_parameters = diffusion_parameters if diffusion_parameters is not None else torch.nn.init.xavier_normal_(torch.empty(self.diffusion_parameters.shape)).reshape(diffusion_shape).to(device) # TODO (KLAUS): Initialize smartly

        self.tensor_drift_parameters = torch.nn.parameter.Parameter(tensor_drift_parameters, requires_grad=True)
        self.tensor_diffusion_parameters = torch.nn.parameter.Parameter(tensor_diffusion_parameters, requires_grad=True)

    def parameters(self):
        return [self.tensor_drift_parameters, self.tensor_diffusion_parameters]

    def sample(self, *args, device="cuda", **kwargs):
        return self.get_sample_gradient_function().apply(*args, **kwargs).to(device)

    def get_sample_gradient_function(self):

        class TorchSDE_PARAM_F(torch.autograd.Function):

            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                return jax_torch(self.v_lambdified_sample(*[torch_jax(arg) for arg in args], **kwargs), requires_grad=True)

            @staticmethod 
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
                #t, data, noise, drift, diffusion = inputs
                ctx.save_for_backward(*inputs)
                #return super().setup_context(ctx, inputs, output) 
            
            @staticmethod
            def backward(ctx: Any, grad_output) -> Any:
                tensors = [torch_jax(tensor) for tensor in ctx.saved_tensors]
                return None, None, None, batch_matmul(grad_output , jax_torch(self.v_lambdified_drift_derivative(*tensors))).to(self.device), batch_matmul(grad_output , jax_torch(self.v_lambdified_diffusion_derivative(*tensors))).to(self.device)
                #return super().backward(ctx, *grad_outputs)

        return TorchSDE_PARAM_F
    def set_timesteps(self, num_inference_steps, batch_size, device):
        self.timesteps = torch.from_numpy(np.linspace(self.min_sample_value, self.max_variable_value, num_inference_steps)[::-1].copy()).to(device).repeat(batch_size)
    
    def reverse_time_derivative(self, timesteps, data, noise, model_output, drift_param, diffusion_param, device="cuda"):
        
        args = (timesteps, data, noise, model_output, drift_param, diffusion_param)
        return jax_torch(self.v_lambdified_reverse_time_derivative(*(torch_jax(arg) for arg in args))).to(device)

    def step(self, data, reverse_time_derivative, dt):
        return data + dt * reverse_time_derivative

if __name__ == "__main__":
    from jax import config
    config.update("jax_enable_x64", True)
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
    import sympy
    from sympy import Symbol, Matrix
    from config.config import Config

    config = Config()
    noise_scheduler = TorchSDE_PARAM(
    device="cuda",
    min_sample_value=config.sde.min_sample_value,
    data_dimension=config.sde.data_dim,
    variable=config.sde.variable,
    drift_parameters=config.sde.drift_parameters,
    diffusion_parameters=config.sde.diffusion_parameters,
    drift=config.sde.drift,
    diffusion=config.sde.diffusion,
    diffusion_matrix=config.sde.diffusion_matrix,
    initial_variable_value=config.sde.initial_variable_value,
    max_variable_value=config.sde.max_variable_value,
    module=config.sde.module,
    model_target=config.sde.target,
    drift_integral_form=config.sde.drift_integral_form,
    diffusion_integral_form=config.sde.diffusion_integral_form,
    diffusion_integral_decomposition=config.sde.diffusion_integral_decomposition,
    drift_dimension=config.sde.drift_dimension,
    diffusion_dimension=config.sde.diffusion_dimension,
    diffusion_matrix_dimension=config.sde.diffusion_matrix_dimension
    )
    import sys 
    sys.exit(0)
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
    print(v_drift_param.shape)
    #v_drift_param = v_diffusion_param = jnp.array([None])

    n = 4 # dims of problem
    timesteps = jnp.array([0.1, 0.4])
    x0 = jnp.ones((len(timesteps), n))*1/2


    z = jax.random.normal(jax.random.PRNGKey(0), x0.shape)
    #z = jnp.array([[ 0.08086788, -0.38624702, -0.37565558,  1.66897423]])
    print(z)
    F = Matrix.diag([sympy.cos(t*drift_param[1])*drift_param[0] + drift_param[2]]*n)
    L = Matrix.diag([sympy.sin(t)*diffusion_param[0]+diffusion_param[1]]*n) # n x n, but to use as diagonal then do 1 x n, 1 x 1 


    Q = Matrix.eye(n)
    # Normal test

    timesteps = jax_torch(timesteps)
    z = jax_torch(z)
    x0 = jax_torch(x0)
    v_drift_param =  jax_torch(v_drift_param, requires_grad=True)
    v_diffusion_param =  jax_torch(v_diffusion_param, requires_grad=True)
    model_output = torch.ones_like(x0) 
    torch.set_printoptions(precision=8)
    sde = TorchSDE_PARAM(1e-3, n, t, drift_param, diffusion_param, F, L, Q, drift_dimension=SDEDimension.FULL, diffusion_dimension=SDEDimension.FULL, diffusion_matrix_dimension=SDEDimension.FULL)
    #sde.initialize_parameters(v_drift_param, v_diffusion_param)
    sde.initialize_parameters()
    print(sde.parameters())
    optimizer = torch.optim.SGD(sde.parameters(), lr=1, momentum=0.99)

    #samples = sde.sample(timesteps, x0, z, v_drift_param , v_diffusion_param)
    #print(samples)

    samples = sde.sample(timesteps, x0, z, *sde.parameters(), device="cuda")
    print(samples)
    samples.sum().backward()
    optimizer.step()
    optimizer.zero_grad()
    print(sde.parameters())

    #print(samples.sum().backward(), type(samples))
    torch.autograd.gradcheck(lambda param, param2: sde.sample(timesteps, x0, z, param, param2), (v_drift_param, v_diffusion_param)) 
    print(sde.reverse_time_derivative(timesteps, x0, z, model_output, v_drift_param, v_diffusion_param, device="cuda") )
    

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