import jax.numpy as jnp

from dataclasses import dataclass


@dataclass
class WandbConfig:
    """
    Config for Weights and Biases
    """

    entity = "ai-dtu"
    project = "Special Course"
    image_amount = 4

    experiment = None  # TODO: Write Code which derives this


@dataclass
class TrainingConfig:
    weight_dtype = jnp.float32

    target = "epsilon"

    dataset_name = "huggan/pokemon"

    seed = 0

    save_dir = None
    push_to_hub = True



@dataclass
class SDEConfig:
    name = "Custom"


@dataclass
class Config:
    logging = WandbConfig()
    training = TrainingConfig()
    sde = SDEConfig()
