import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import sys

import random
import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


from config.config import Config
from config.utils import get_wandb_input, initialize_sde_parameter_plot, update_sde_parameter_plot 
from data.dataload import get_dataset 

from transformers import set_seed, CLIPTextModel, CLIPTokenizer

from diffusers.utils import check_min_version, make_image_grid

from pipelines.pipeline_tti_torch import UTTIPipeline, Noise, SDESolver

check_min_version("0.22.0.dev0")
import wandb

from diffusers import (UNet2DConditionModel)

#from sde_torch import TorchSDE 
from sde_torch_param import TorchSDE_PARAM


def main():

    sys.setrecursionlimit(15000)
    config = Config()

    if config.debug:
        print("WARNING: DEBUG MODE IS ON")

# SET SEED
    if config.training.seed is not None:
        set_seed(config.training.seed)

# ACCELERATOR

    logging_dir = os.path.join(config.training.save_dir, "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=config.training.save_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision[0],
        project_config=accelerator_project_config,
        log_with="wandb"
    ) 
    print(f"Infering on device: {accelerator.device}")

    if accelerator.is_main_process:
        log_kwargs = {"wandb": get_wandb_input(config)}
        project_name = log_kwargs["wandb"].pop("project")
        accelerator.init_trackers(project_name, init_kwargs=log_kwargs)
        sde_param_plots = initialize_sde_parameter_plot(config)
        noise_types = list(Noise)
# LOAD DATA


    tokenizer = CLIPTokenizer.from_pretrained(
        config.training.pretrained_model_or_path, cache_dir=config.training.cache_dir, revision=config.training.revision, subfolder="tokenizer"
    )
    #tokenizer = torch.compile(tokenizer)

    train_dataset, train_dataloader = get_dataset(config, tokenizer, interface="torch", accelerator=accelerator, tokenize_captions=False)



# LOAD IN MODELS 
    text_encoder = CLIPTextModel.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="text_encoder", 
        cache_dir=config.training.cache_dir,
    )
    text_encoder.requires_grad_(False)
    #text_encoder = torch.compile(text_encoder)

    # TODO Make this a class containing the SDE and the UNET

    unet = UNet2DConditionModel.from_pretrained(config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="unet", cache_dir=config.training.cache_dir)
    print("Loaded pretrained UNET")

    #unet = torch.compile(unet)
    unet.eval()    
    unet.requires_grad_(False)
# NOISE SCHEDULAR

    #noise_scheduler = DDIMScheduler()

    
    noise_scheduler = TorchSDE_PARAM.from_pretrained(config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="scheduler", cache_dir=config.training.cache_dir, device=accelerator.device)
    print("Loaded pretrained noise scheduler")
    unet, train_dataloader, noise_scheduler = accelerator.prepare(unet, train_dataloader, noise_scheduler)

    text_encoder.to(accelerator.device,dtype=config.training.mixed_precision[1])
    global_step = 0
    
    if accelerator.is_main_process:
            _log_drift_param, _log_diffusion_param = noise_scheduler.parameters() 
            update_sde_parameter_plot(sde_param_plots[0], global_step, *_log_drift_param.detach())
            update_sde_parameter_plot(sde_param_plots[1], global_step, *_log_diffusion_param.detach())
            accelerator.log({"Drift Parameters": sde_param_plots[0], "Diffusion Parameters": sde_param_plots[1]}, step=global_step) 

    unet = accelerator.unwrap_model(unet) 
    unet.eval()

    pipeline = UTTIPipeline(unet, accelerator.unwrap_model(noise_scheduler), tokenizer, accelerator.unwrap_model(text_encoder))

    all_generated_images = []
    for batch in train_dataloader:
        if accelerator.is_main_process: 

            _log_drift_param, _log_diffusion_param = noise_scheduler.parameters() 
            update_sde_parameter_plot(sde_param_plots[0], global_step, *_log_drift_param.detach())
            update_sde_parameter_plot(sde_param_plots[1], global_step, *_log_diffusion_param.detach())
            accelerator.log({"Drift Parameters": sde_param_plots[0], "Diffusion Parameters": sde_param_plots[1]}, step=global_step) 


            prompts = batch["input_ids"]
            print(prompts)
            noise_type = random.choice(noise_types)
            images = pipeline(prompts, accelerator.device, generator=torch.manual_seed(config.training.seed), num_inference_steps=1000, noise=noise_type, method=SDESolver.EULER, debug=config.debug).images

            image_grid = make_image_grid(images, rows=3,cols=4)
            accelerator.log({f"image-{noise_type}": wandb.Image(image_grid)}, step=global_step)

            all_generated_images += images
            all_images += batch["pixel_values"]

if __name__ == "__main__":
    main()