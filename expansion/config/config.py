import jax.numpy as jnp
import jax
import torch
from dataclasses import dataclass
import sympy
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

    experiment = "TTI"  # TODO: Write Code which derives this


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

    cache_dir = "./cache" # The directory where the downloaded models and datasets will be stored.

# IMAGE CONFIGURATION

    center_crop = False
    random_flip = True
    resolution = 32

# HYPER PARAMETERS
    seed = 0

    batch_size = 30

    total_batch_size = batch_size * jax.local_device_count() # TODO (KLAUS) : CHANGE TO BE NON JAX DEPENDENT

    max_steps = None
    epochs = 1000 

    save_dir = "text-to-image"
    push_to_hub = True
    pretrained_model_or_path ="CompVis/stable-diffusion-v1-4" #"duongna/stable-diffusion-v1-4-flax" "CompVis/stable-diffusion-v1-4"
    revision = None # LEGITEMATALY DON'T KNOW WHAT THIS DOES

@dataclass
class OptimizerConfig:

    learning_rate = 2e-5 # Initial learning rate (after the potential warmup period) to use.

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

    

@dataclass
class SDEConfig:
    name = "Custom"
    variable = Symbol('t', nonnegative=True, real=True)

    n = 1 # n = 1 -> a scalar matrix
    
    drift = Matrix.diag([-100*variable**2]*n).diagonal()
    diffusion = Matrix.diag([sympy.Piecewise((sympy.sin(variable/2 * sympy.pi), variable < 1), (1, variable >= 1))]*n).diagonal()
    # TODO (KLAUS) : in the SDE SAMPLING CHANGING Q impacts how we sample z ~ N(0, Q*(delta t))
    diffusion_matrix = Matrix.eye(n).diagonal()

    initial_variable_value = 0
    max_variable_value = 1 # math.inf
    min_sample_value = 1e-5

    module = 'jax'

    drift_integral_form=True
    diffusion_integral_form=True
    diffusion_integral_decomposition = 'cholesky' # ldl

    drift_diagonal_form=True
    diffusion_diagonal_form=True
    diffusion_matrix_diagonal_form=True

    target = "epsilon" # x0



@dataclass
class Config:
    logging = WandbConfig()
    training = TrainingConfig()
    sde = SDEConfig()

    optimizer = OptimizerConfig()
    if optimizer.scale_lr:
        optimizer.learning_rate *= training.total_batch_size
