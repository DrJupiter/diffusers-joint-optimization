import jax.numpy as jnp
import jax

from dataclasses import dataclass


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
    weight_dtype = jnp.float32

    target = "epsilon"

    dataset_name = "lambdalabs/pokemon-blip-captions"
    dataset_config_name = None # The config of the Dataset, leave as None if there's only one config.

    train_data_dir = None # A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file

    image_column = "image"
    caption_column = "text"

    cache_dir = "./cache" # The directory where the downloaded models and datasets will be stored.

# IMAGE CONFIGURATION

    center_crop = False
    random_flip = True
    resolution = 64

# HYPER PARAMETERS
    seed = 0

    batch_size = 16

    total_batch_size = batch_size * jax.local_device_count()

    max_steps = None
    epochs = 10000 

    save_dir = "text-to-image"
    push_to_hub = True
    pretrained_model_or_path = "duongna/stable-diffusion-v1-4-flax"
    revision = None # LEGITEMATALY DON'T KNOW WHAT THIS DOES

@dataclass
class OptimizerConfig:

    learning_rate = 1e-4 # Initial learning rate (after the potential warmup period) to use.

    scale_lr = False # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.

    # TODO (KLAUS) : CODE IN THE OTHER OPTIMIZERS
    lr_scheduler = "constant" # 'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'


    # adam parameters
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_weight_decay = 1e-2
    adam_epsilon = 1e-8

    max_grad_norm = 1

@dataclass
class SDEConfig:
    name = "Custom"

@dataclass
class Config:
    logging = WandbConfig()
    training = TrainingConfig()
    sde = SDEConfig()

    optimizer = OptimizerConfig()
    if optimizer.scale_lr:
        optimizer.learning_rate *= training.total_batch_size
