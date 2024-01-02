import jax.numpy as jnp
import jax
import torch
from dataclasses import dataclass
import sympy
import sympy as sp
from sympy import Matrix, Symbol 
import math

@dataclass
class WandbConfig:
    """
    Config for Weights and Biases
    """

    entity = "ai-dtu"
    project = "Special Course"
    image_amount = 4

    experiment = "TTI-NO-INIT-NO-DIFFUSION-PARAM"  # TODO: Write Code which derives this


@dataclass
class TrainingConfig:

    # JAX
    weight_dtype = jnp.float32

    # TORCH
    mixed_precision = ("fp16", torch.float16) # `no` for float32, `fp16` for automatic mixed precision

    target = "epsilon"

    dataset_name = "lambdalabs/pokemon-blip-captions" #"mnist" 
    dataset_config_name = None # The config of the Dataset, leave as None if there's only one config.

    train_data_dir = None # A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file

    image_column = "image"
    caption_column = "text" #"label" 
    try_convert_label_string = False # True

    cache_dir = "/work3/s204123/cache" # The directory where the downloaded models and datasets will be stored.
    #cache_dir = "./cache" # The directory where the downloaded models and datasets will be stored.
    
# IMAGE CONFIGURATION

    center_crop = False
    random_flip = True
    resolution = 32 # 32

# HYPER PARAMETERS
    seed = 0

    batch_size = 32

    total_batch_size = batch_size * jax.local_device_count() # TODO (KLAUS) : CHANGE TO BE NON JAX DEPENDENT

    max_steps = None
    epochs = 10000


    repo_name = "pokemon-base-line-kerasVe"

    save_dir = f"/work3/s204123/{repo_name}"

    push_to_hub = True
    pretrained_model_or_path = "AltLuv/pokemon-test" # "runwayml/stable-diffusion-v1-5" # "stabilityai/stable-diffusion-xl-base-1.0" #"duongna/stable-diffusion-v1-4-flax" "CompVis/stable-diffusion-v1-4"
    revision = None # LEGITEMATALY DON'T KNOW WHAT THIS DOES

    load_pretrained_model = False # True -> load pretrained unet, False -> Train unet from scratch.
    load_optimizer = False # True -> load optimizer, False -> Train optimizer from scratch.

@dataclass
class OptimizerConfig:

    learning_rate = 5e-6 # Initial learning rate (after the potential warmup period) to use.

    scale_lr = False # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.

    # TODO (KLAUS) : CODE IN THE OTHER OPTIMIZERS
    lr_scheduler = "cosine" # 'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'

    # cosine parameters
    warmup_steps = 500
    init_value = 4e-7

    # adam parameters
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8

    max_grad_norm = 1.0


from sde_redefined_param import SDEDimension

@dataclass
class SDEConfig:
    name = "Custom"
    variable = Symbol('t', nonnegative=True, real=True)

    drift_dimension = SDEDimension.SCALAR 
    diffusion_dimension = SDEDimension.SCALAR
    diffusion_matrix_dimension = SDEDimension.SCALAR 

    # TODO (KLAUS): HANDLE THE PARAMETERS BEING Ø
    drift_parameters = Matrix([sympy.symbols("f1")])
    diffusion_parameters = Matrix([sympy.symbols("l1")])
    
    drift =-variable**2 * drift_parameters[0]**2
    k = 1 #* diffusion_parameters[0]**2 
    diffusion = sympy.Piecewise((k * sympy.sin(variable/2 * sympy.pi), variable < 1), (k*1, variable >= 1))
    # TODO (KLAUS) : in the SDE SAMPLING CHANGING Q impacts how we sample z ~ N(0, Q*(delta t))
    diffusion_matrix = 1 

    initial_variable_value = 0
    max_variable_value = 1 # math.inf
    min_sample_value = 1e-6

    module = 'jax'

    drift_integral_form=True
    diffusion_integral_form=True
    diffusion_integral_decomposition = 'cholesky' # ldl



    target = "epsilon" # x0

@dataclass
class SDEBaseLineConfig:
    name = "Custom"
    variable = Symbol('t', nonnegative=True, real=True)

    drift_dimension = SDEDimension.SCALAR 
    diffusion_dimension = SDEDimension.SCALAR
    diffusion_matrix_dimension = SDEDimension.SCALAR 

    # TODO (KLAUS): HANDLE THE PARAMETERS BEING Ø
    drift_parameters = Matrix([sympy.symbols("f1")])
    diffusion_parameters = Matrix([sympy.symbols("l1")])
    
    drift = 0

    sigma_min = 0.002 
    sigma_max = 80
    diffusion = sigma_min * (sigma_max/sigma_min)**variable * sympy.sqrt(2 * sympy.log(sigma_max/sigma_min)) 
    # TODO (KLAUS) : in the SDE SAMPLING CHANGING Q impacts how we sample z ~ N(0, Q*(delta t))
    diffusion_matrix = 1 

    initial_variable_value = 0
    max_variable_value = 1 # math.inf
    min_sample_value = 1e-6

    module = 'jax'

    drift_integral_form=False
    diffusion_integral_form=False
    diffusion_integral_decomposition = 'cholesky' # ldl



    target = "epsilon" # x0


def from_01_to_m_inf_inf(x):
    """applies a functino that maps from ]0;1[ to ]-\infty;\infty["""
    return -sp.ln(1/x-1)

def from_m_inf_inf_to_0_m_inf(x):
    """ applies a functino that maps from ]-\infty;\infty[ to ]0;-inf[ """
    return -sp.exp(x)

def polynomial(x,params):
    """Crates a polynomial with degrees equal to param count"""
    summ = 0
    for i,param in enumerate(params):
        summ += param*x**i
    return summ

def create_drift_param_func(var,degree,subname,func):
    """Creates a parameterized function that maps t for ]0;1[ to ]0, -inf[ while letting the params be in unconstrained space"""
    # Map from 01 to -inf inf
    f1 = from_01_to_m_inf_inf(var)

    # define params with biggest factor = 1 (must be postitive to ensure that the func goes to inf)
    params = sp.symbols(" ".join([f"p_{subname}{i}" for i in range(degree-1)])+",1",real=True)

    # map parameteierzed function to 0 -inf
    f2 = from_m_inf_inf_to_0_m_inf(func(f1,params))

    return f2

def create_diffusion_param_func(var,degree,subname,func):
    """Creates a parameterized function that is always positive and with a positive derivative"""
    # define params with biggest factor = 1 (must be postitive to ensure that the func goes to inf)
    params = sp.symbols(" ".join([f"p_{subname}{i}" for i in range(degree)]),real=True)

    # square paramterers, to ensure that we have a positive function
    params = [param**2 for param in params]

    # map parameteierzed function to 0 -inf
    f2 = func(var,params)

    return f2

# !! TODO (KLAUS): FOR SOME REASON IT SAYS t isn't defined in the var = t statement. 
# !! I HAVE NO IDEA WHY, AND IT MIGHT JUST BE ON MY MACHINE. 
# !! THAT'S WHY I'VE COMMENTED IT OUT.
#@dataclass
#class SDEConfigParamerterized:
#    name = "Custom"
#    t = Symbol('t', nonnegative=True, real=True)
#
#    drift_dimension = SDEDimension.DIAGONAL
#    diffusion_dimension = SDEDimension.DIAGONAL
#    diffusion_matrix_dimension = SDEDimension.SCALAR
#    n = 2 # n = 1 -> a scalar matrix
#    poly_degree = 4
#    # Using functions abovae
#    drift = sp.Matrix([create_drift_param_func(var = t,degree = poly_degree,subname = i, func = polynomial) for i in range(n)]).T
#    diffusion = sp.Matrix([create_diffusion_param_func(var = t,degree = poly_degree,subname = i, func = polynomial) for i in range(n)]).T
#    # TODO (KLAUS) : in the SDE SAMPLING CHANGING Q impacts how we sample z ~ N(0, Q*(delta t))
#    diffusion_matrix = Matrix([1]) # because of dimension choice, this will be delt with as I
#
#    initial_variable_value = 0
#    max_variable_value = 1 # math.inf
#    min_sample_value = 1e-4
#    # TODO: Max sample value (as we cannot sample t=1 for drift)
#
#    module = 'jax'
#
#    drift_integral_form=True
#    diffusion_integral_form=True
#    diffusion_integral_decomposition = 'cholesky' # ldl
#
#    target = "epsilon" # x0


@dataclass
class Config:
    logging = WandbConfig()
    training = TrainingConfig()
    #sde = SDEConfig()
    sde = SDEBaseLineConfig()
    sde.data_dim = training.resolution ** 2 * 3

    optimizer = OptimizerConfig()
    if optimizer.scale_lr:
        optimizer.learning_rate *= training.total_batch_size


    # SANITY CHECKS for the SDE
    # TODO (KLAUS) : REFACTOR 
    #if sde.drift_type == SDEDimension.SCALAR:
    #    assert sde.drift.shape == (1,1), "A scalar drift must have dimensions (1,1)"
    #elif sde.drift_type == SDEDimension.DIAGONAL:
    #    assert sde.drift.shape == (1, training.resolution), "A diagonal drift must have dimensions (1, resolution)"
    #elif sde.drift_type == SDEDimension.FULL:
    #    assert sde.drift.shape == (training.resolution, training.resolution), "A full drift must have dimensions (resolution, resolution)"
    
    #if sde.diffusion_type == SDEDimension.SCALAR:
    #    assert sde.diffusion.shape == (1,1), "A scalar drift must have dimensions (1,1)"
    #elif sde.diffusion_type == SDEDimension.DIAGONAL:
    #    assert sde.diffusion.shape == (1, training.resolution), "A diagonal drift must have dimensions (1, resolution)"
    #elif sde.diffusion_type == SDEDimension.FULL:
    #    assert sde.diffusion.shape == (training.resolution, training.resolution), "A full drift must have dimensions (resolution, resolution)"