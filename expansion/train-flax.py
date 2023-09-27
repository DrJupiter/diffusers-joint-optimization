
from config.config import Config
from config.utils import get_wandb_input, save_local_cloud

from transformers import set_seed

import wandb

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
        args.pretrained_model_name_or_path, revision=args.revision, subfolder="text_encoder", dtype=weight_dtype
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, subfolder="vae", dtype=weight_dtype
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, revision=args.revision, subfolder="unet", dtype=weight_dtype
    )    

# SAVE PARAMETERS
    params = NotImplementedError("MODELS, OPTIMIZER")
    save_local_cloud(config, params)
