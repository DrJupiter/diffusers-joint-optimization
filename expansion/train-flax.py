
from config.config import Config
from config.utils import get_wandb_input, save_local_cloud

from transformers import set_seed, FlaxCLIPTextModel

import wandb


from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)

def main():

# GET CONFIG & LOGGING
    config = Config()

    wandb.init(**get_wandb_input(config)) 

# SET SEED
    if config.training.seed is not None:
        set_seed(config.training.seed)

# MODELS
# TODO (KLAUS): CREATE MODELS
    text_encoder, vae, unet,  = get_models(config)
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.revision, subfolder="text_encoder", dtype=config.training.weight_dtype
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.revision, subfolder="vae", dtype=config.training.weight_dtype
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.revision, subfolder="unet", dtype=config.training.weight_dtype
    )    

# SAVE PARAMETERS
    params = NotImplementedError("MODELS, OPTIMIZER")
    save_local_cloud(config, params)
