from os import PathLike
import os
from sde_redefined_param import SDE_PARAM, SDEDimension
from typing import Any, Dict, Tuple, Union, Optional, ClassVar
from torch import TensorType
#import torch
import numpy as np
import math
import jax.numpy as jnp
import jax
# TODO (Klaus) : Add mxin and config so we can save properly
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config, http_user_agent, hf_hub_download
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from transformers import set_seed

from safetensors import safe_open
from safetensors.torch import save_file
from config.utils import save_class_to_file, load_class_from_file
from config.config import Config

from huggingface_hub.utils import validate_hf_hub_args

batch_matmul = torch.vmap(torch.matmul)
batch_mul    = torch.vmap(torch.mul)

def jax_torch(array, requires_grad=False):
    array = torch.from_numpy(np.array(array))
    array.requires_grad_(requires_grad)
    return array

def torch_jax(tensor):
    return jnp.array(tensor.numpy(force=True))

class TorchSDE_PARAM(SchedulerMixin, ConfigMixin, SDE_PARAM):
    
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
    diffusion_matrix_dimension: SDEDimension = SDEDimension.SCALAR,
    config_class = Config.sde.__class__,
    non_symbolic_parameters: Optional[Dict[TensorType, TensorType]] = None,
    ):
        super().__init__(variable, drift_parameters, diffusion_parameters, drift, diffusion, diffusion_matrix, initial_variable_value, max_variable_value, module, model_target, drift_integral_form, diffusion_integral_form, diffusion_integral_decomposition, drift_dimension, diffusion_dimension, diffusion_matrix_dimension)

        self.lambdify_symbolic_functions(data_dimension)

        if non_symbolic_parameters is not None:
            self.initialize_parameters(non_symbolic_parameters.pop('drift', None), non_symbolic_parameters.pop('diffusion', None), device=device) 
        else:
            self.initialize_parameters(device=device) 

        self.device=device
        self.min_sample_value = min_sample_value

        self.config_class = config_class
        self._internal_dict = {'data_dimension': data_dimension, '_config_class_name': self.config_class.__name__}
        
    def initialize_parameters(self, drift_parameters=None, diffusion_parameters=None, device="cuda"):
        # TODO (KLAUS): ADD CHECK FOR PARAMETER SHAPES, WHEN THEY AREN'T NONE
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
                return None, None, batch_mul(grad_output, jax_torch(self.v_lambdified_scaled_loss_derivative_model(*tensors))).to(self.device), batch_mul(grad_output, jax_torch(self.v_lambdified_scaled_loss_derivative_drift(*tensors))).to(self.device), batch_mul(grad_output, jax_torch(self.v_lambdified_scaled_loss_derivative_diffusion(*tensors))).to(self.device) 
                #return super().backward(ctx, *grad_outputs)

        return TorchSDE_PARAM_SCALED_LOSS

    def set_timesteps(self, num_inference_steps, batch_size, device):
        self.timesteps = torch.from_numpy(np.linspace(self.min_sample_value, self.max_variable_value, num_inference_steps)[::-1].copy()).to(device).repeat(batch_size)
    
    def reverse_time_derivative(self, timesteps, data, noise, model_output, drift_param, diffusion_param, device="cuda"):
        
        args = (timesteps, data, noise, model_output, drift_param, diffusion_param)
        return jax_torch(self.v_lambdified_reverse_time_derivative(*(torch_jax(arg) for arg in args))).to(device)

    def step(self, data, reverse_time_derivative, dt):
        return data + dt * reverse_time_derivative

    def save_pretrained(self, save_directory: Union[str, PathLike], push_to_hub: bool = False, **kwargs):
        """
        Save a scheduler configuration object to a directory so that it can be reloaded using the
        [`~SchedulerMixin.from_pretrained`] class method.

        As the scheduler is parameterized we save:
            - the drift parameters
            - the diffusion parameters
            - the config file for the sde
            - the other arguments passed to the scheduler

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face Hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        #config.sde.__class__
        #<class 'config.config.SDEConfig'>
        #inspect.getsource(config.sde.__class__)

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        parameters = dict(zip(['drift', 'diffusion'] , self.parameters()))
        save_file(parameters, os.path.join(save_directory, "sdeparameters.pt"))

        save_class_to_file(self.config_class, os.path.join(save_directory, "scheduler_config.py"), getattr(self, "class_path", None))

        self.save_config(save_directory=save_directory, push_to_hub=push_to_hub, **kwargs)

    @classmethod
    @validate_hf_hub_args
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, PathLike]] = None,
        subfolder: Optional[str] = None,
        return_unused_kwargs=False,
        **kwargs,
    ):
        r"""
        Instantiate a scheduler from a pre-defined JSON configuration file in a local directory or Hub repository.

        Parameters:
            pretrained_model_name_or_path (`str` or `os.PathLike`, *optional*):
                Can be either:

                    - A string, the *model id* (for example `google/ddpm-celebahq-256`) of a pretrained model hosted on
                      the Hub.
                    - A path to a *directory* (for example `./my_model_directory`) containing the scheduler
                      configuration saved with [`~SchedulerMixin.save_pretrained`].
            subfolder (`str`, *optional*):
                The subfolder location of a model file within a larger model repository on the Hub or locally.
            return_unused_kwargs (`bool`, *optional*, defaults to `False`):
                Whether kwargs that are not consumed by the Python class should be returned or not.
            cache_dir (`Union[str, os.PathLike]`, *optional*):
                Path to a directory where a downloaded pretrained model configuration is cached if the standard cache
                is not used.
            force_download (`bool`, *optional*, defaults to `False`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            resume_download (`bool`, *optional*, defaults to `False`):
                Whether or not to resume downloading the model weights and configuration files. If set to `False`, any
                incompletely downloaded files are deleted.
            proxies (`Dict[str, str]`, *optional*):
                A dictionary of proxy servers to use by protocol or endpoint, for example, `{'http': 'foo.bar:3128',
                'http://hostname': 'foo.bar:4012'}`. The proxies are used on each request.
            output_loading_info(`bool`, *optional*, defaults to `False`):
                Whether or not to also return a dictionary containing missing keys, unexpected keys and error messages.
            local_files_only(`bool`, *optional*, defaults to `False`):
                Whether to only load local model weights and configuration files or not. If set to `True`, the model
                won't be downloaded from the Hub.
            token (`str` or *bool*, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, the token generated from
                `diffusers-cli login` (stored in `~/.huggingface`) is used.
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, a commit id, or any identifier
                allowed by Git.

        <Tip>

        To use private or [gated models](https://huggingface.co/docs/hub/models-gated#gated-models), log-in with
        `huggingface-cli login`. You can also activate the special
        ["offline-mode"](https://huggingface.co/diffusers/installation.html#offline-mode) to use this method in a
        firewalled environment.

        </Tip>

        """

        device = kwargs.pop("device", "cpu")

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        resume_download = kwargs.pop("resume_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        _ = kwargs.pop("mirror", None)
        subfolder = kwargs.pop("subfolder", "scheduler")
        user_agent = kwargs.pop("user_agent", {})

        user_agent = {**user_agent, "file_type": "config"}
        user_agent = http_user_agent(user_agent)

        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        if cls.config_name is None:
            raise ValueError(
                "`self.config_name` is not defined. Note that one should not load a config from "
                "`ConfigMixin`. Please make sure to define `config_name` in a class inheriting from `ConfigMixin`"
            )

        parameter_path = hf_hub_download(pretrained_model_name_or_path, filename="sdeparameters.pt", revision=revision, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, use_auth_token=token, user_agent=user_agent, subfolder=subfolder)

        config_json_path = hf_hub_download(pretrained_model_name_or_path, filename=cls.config_name, revision=revision, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, use_auth_token=token, user_agent=user_agent, subfolder=subfolder)

        config_py_path = hf_hub_download(pretrained_model_name_or_path, filename="scheduler_config.py", revision=revision, cache_dir=cache_dir, force_download=force_download, resume_download=resume_download, proxies=proxies, local_files_only=local_files_only, use_auth_token=token, user_agent=user_agent, subfolder=subfolder)

        parameters = {}
        with safe_open(parameter_path, framework="pt") as f:
            for key in f.keys():
                parameters[key] = f.get_tensor(key).to(device)
        config_json_dict = cls._dict_from_json_file(config_json_path) 
        
        config_class = load_class_from_file(config_json_dict["_config_class_name"], config_py_path) 
        noise_scheduler = cls(
            device=device,
            min_sample_value=config_class.min_sample_value,
            data_dimension=config_json_dict["data_dimension"],
            variable=config_class.variable,
            drift_parameters=config_class.drift_parameters,
            diffusion_parameters=config_class.diffusion_parameters,
            drift=config_class.drift,
            diffusion=config_class.diffusion,
            diffusion_matrix=config_class.diffusion_matrix,
            initial_variable_value=config_class.initial_variable_value,
            max_variable_value=config_class.max_variable_value,
            module=config_class.module,
            model_target=config_class.target,
            drift_integral_form=config_class.drift_integral_form,
            diffusion_integral_form=config_class.diffusion_integral_form,
            diffusion_integral_decomposition=config_class.diffusion_integral_decomposition,
            drift_dimension=config_class.drift_dimension,
            diffusion_dimension=config_class.diffusion_dimension,
            diffusion_matrix_dimension=config_class.diffusion_matrix_dimension,
            config_class=config_class.__class__,
            non_symbolic_parameters=parameters,
            )

        # Set class path, so we can save the correct class later
        noise_scheduler.class_path = config_py_path 

        return noise_scheduler


        

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
    # Loading and saving
    #noise_scheduler = TorchSDE_PARAM.from_pretrained('AltLuv/pokemon-testing', subfolder="scheduler", revision="main", cache_dir="/media/extra/diffusers-joint-optimization/expansion/config/cache")
    #noise_scheduler.save_pretrained("/media/extra/diffusers-joint-optimization/bird/scheduler")
    timesteps = jnp.array([0.1, 0.4])
    x0 = jnp.ones((len(timesteps), data_dimension))*1/2


    z = jax.random.normal(jax.random.PRNGKey(0), x0.shape)
    #z = jnp.array([[ 0.08086788, -0.38624702, -0.37565558,  1.66897423]])
    print(z)

    # Normal test

    timesteps = jax_torch(timesteps)
    z = jax_torch(z)
    x0 = jax_torch(x0)
    model_output = torch.randn_like(x0, device='cuda', requires_grad=True)


    torch.set_printoptions(precision=8)
    #sde.initialize_parameters(v_drift_param, v_diffusion_param)

    #samples = sde.sample(timesteps, x0, z, v_drift_param , v_diffusion_param)
    #print(samples)

    # TEST GRADIENTS
    torch.autograd.gradcheck(lambda param, param2: noise_scheduler.sample(timesteps, x0, z, param, param2), noise_scheduler.parameters()) 
    torch.autograd.gradcheck(lambda model, drift, diffusion: noise_scheduler.scaled_loss(timesteps, z, model, drift, diffusion), (model_output,*noise_scheduler.parameters(),))


    # TEST UPDATE
    print(noise_scheduler.parameters())
    optimizer = torch.optim.SGD(noise_scheduler.parameters(), lr = 0.1, momentum=0.9)

    samples = noise_scheduler.sample(timesteps, x0, z, *noise_scheduler.parameters())
    prediction = samples * model_output

    
    #print(batch_matmul(diff, diff @ I).sum())
    #loss = torch.nn.functional.mse_loss(prediction, z.cuda()) 
    loss = noise_scheduler.scaled_loss(timesteps, z, prediction, *noise_scheduler.parameters()).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(noise_scheduler.parameters())