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
from transformers import set_seed

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

        class TorchSDE_PARAM_SAMPLE(torch.autograd.Function):

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

        return TorchSDE_PARAM_SAMPLE

    def scaled_loss(self, *args, device="cuda", **kwargs):
        return self.get_scaled_loss_gradient_function().apply(*args, **kwargs).to(device)

    def get_scaled_loss_gradient_function(self):

        class TorchSDE_PARAM_SCALED_LOSS(torch.autograd.Function):

            @staticmethod
            def forward(*args: Any, **kwargs: Any) -> Any:
                return jax_torch(self.v_lambdified_scaled_loss(*[torch_jax(arg) for arg in args], **kwargs), requires_grad=True)

            @staticmethod 
            def setup_context(ctx: Any, inputs: Tuple[Any, ...], output: Any) -> Any:
                #t, data, drift= inputs
                ctx.save_for_backward(*inputs)
                #return super().setup_context(ctx, inputs, output) 
            
            @staticmethod
            def backward(ctx: Any, grad_output) -> Any:
                tensors = [torch_jax(tensor) for tensor in ctx.saved_tensors]
                return None, None, batch_matmul(grad_output, jax_torch(self.v_lambdified_scaled_loss_derivative_model(*tensors))).to(self.device), batch_matmul(grad_output, jax_torch(self.v_lambdified_scaled_loss_derivative_drift(*tensors))).to(self.device), batch_matmul(grad_output, jax_torch(self.v_lambdified_scaled_loss_derivative_diffusion(*tensors))).to(self.device) 
                #return super().backward(ctx, *grad_outputs)

        return TorchSDE_PARAM_SCALED_LOSS

    def set_timesteps(self, num_inference_steps, batch_size, device):
        self.timesteps = torch.from_numpy(np.linspace(self.min_sample_value, self.max_variable_value, num_inference_steps)[::-1].copy()).to(device).repeat(batch_size)
    
    def reverse_time_derivative(self, timesteps, data, noise, model_output, drift_param, diffusion_param, device="cuda"):
        
        args = (timesteps, data, noise, model_output, drift_param, diffusion_param)
        return jax_torch(self.v_lambdified_reverse_time_derivative(*(torch_jax(arg) for arg in args))).to(device)

    def step(self, data, reverse_time_derivative, dt):
        return data + dt * reverse_time_derivative

if __name__ == "__main__":
    import os
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

    from jax import config
    config.update("jax_enable_x64", True)
    torch.set_default_dtype(torch.float64)
    import sympy
    from sympy import Symbol, Matrix
    from config.config import Config

    config = Config()
    set_seed(config.training.seed)

    data_dimension = 4

    noise_scheduler = TorchSDE_PARAM(
    device="cuda",
    min_sample_value=config.sde.min_sample_value,
    data_dimension=data_dimension, #
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
    
    timesteps = jnp.array([0.1, 0.4])
    x0 = jnp.ones((len(timesteps), data_dimension))*1/2


    z = jax.random.normal(jax.random.PRNGKey(0), x0.shape)
    #z = jnp.array([[ 0.08086788, -0.38624702, -0.37565558,  1.66897423]])
    print(z)

    # Normal test

    timesteps = jax_torch(timesteps)
    z = jax_torch(z)
    x0 = jax_torch(x0)
    model_output = torch.ones_like(x0, device='cuda', requires_grad=True)

    torch.set_printoptions(precision=8)
    #sde.initialize_parameters(v_drift_param, v_diffusion_param)

    #samples = sde.sample(timesteps, x0, z, v_drift_param , v_diffusion_param)
    #print(samples)

    # TEST GRADIENTS
    torch.autograd.gradcheck(lambda param, param2: noise_scheduler.sample(timesteps, x0, z, param, param2), noise_scheduler.parameters()) 
    torch.autograd.gradcheck(lambda model, drift, diffusion: noise_scheduler.scaled_loss(timesteps, x0, model, drift, diffusion), (model_output,*noise_scheduler.parameters(),))


    import sys
    sys.exit(0)
    # TEST UPDATE
    model_output = torch.ones_like(x0, device='cuda') 
    print(noise_scheduler.parameters())
    optimizer = torch.optim.SGD(noise_scheduler.parameters(), lr = 0.1, momentum=0.9)

    samples = noise_scheduler.sample(timesteps, x0, z, *noise_scheduler.parameters())
    diff = model_output-samples
    mean_matrix = noise_scheduler.mean(timesteps, x0, noise_scheduler.parameters()[0])
    print(batch_matmul(diff, diff))
    I = torch.stack([torch.eye(data_dimension)]*2).cuda()+1
    print(batch_matmul(diff,batch_matmul(I, diff)))
    I2 = torch.eye(data_dimension).cuda()+1
    print(batch_matmul(diff, diff @ I2))
    
    #print(batch_matmul(diff, diff @ I).sum())
    print(torch.nn.functional.mse_loss(model_output, samples, reduction='sum'))
    loss = torch.mean(batch_matmul(diff, diff * mean_matrix))
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(noise_scheduler.parameters())