import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

# LOAD JAX FOR SDE LIB IN JAX
import jax
import jax.numpy as jnp
# load DNN library 
#jax.random.PRNGKey(0)

import math
from tqdm.auto import tqdm

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


from config.config import Config
from config.utils import get_wandb_input, save_local_cloud 
from data.dataload import get_dataset 

from transformers import set_seed, CLIPTextModel, CLIPTokenizer

from diffusers.utils import check_min_version, make_image_grid
from diffusers.utils.pil_utils import numpy_to_pil

from pipelines.pipeline_tti_torch import UTTIPipeline

check_min_version("0.22.0.dev0")
import wandb

from diffusers import (DDPMScheduler,DDIMScheduler, UNet2DConditionModel)
from diffusers.optimization import get_cosine_schedule_with_warmup

from sde_torch import TorchSDE 

def main():

    config = Config()


# SET SEED
    if config.training.seed is not None:
        set_seed(config.training.seed)

    # jax rng
    key = jax.random.PRNGKey(config.training.seed)

# ACCELERATOR

    logging_dir = os.path.join(config.training.save_dir, "logs")

    accelerator_project_config = ProjectConfiguration(project_dir=config.training.save_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=config.training.mixed_precision[0],
        project_config=accelerator_project_config,
        log_with="wandb"
    ) 
    print(f"Training on device: {accelerator.device}")

    if accelerator.is_main_process:
        log_kwargs = {"wandb": get_wandb_input(config)}
        project_name = log_kwargs["wandb"].pop("project")
        accelerator.init_trackers(project_name, init_kwargs=log_kwargs)
# LOAD DATA


    tokenizer = CLIPTokenizer.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="tokenizer"
    )

    train_dataset, train_dataloader = get_dataset(config, tokenizer, interface="torch", accelerator=accelerator)


# CALCULATE NUMBER OF TRAINING STEPS
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    if config.training.max_steps is None:
        config.training.max_steps = config.training.epochs * num_update_steps_per_epoch

# LOAD IN MODELS 
    text_encoder = CLIPTextModel.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="text_encoder", 
        cache_dir=config.training.cache_dir,
    )
    text_encoder.requires_grad_(False)

    # TODO Make this a class containing the SDE and the UNET
    unet = UNet2DConditionModel(sample_size=config.training.resolution,
                                in_channels=3,
                                out_channels=3,
                                cross_attention_dim=768 # TODO (KLAUS) : EXTRACT THIS NUMBER FROM CLIP MODEL
                                )
#    unet = UNet2DConditionModel(sample_size=config.training.resolution,
#                                in_channels=3,
#                                out_channels=3,
#                                block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
#                                down_block_types=(
#                                    "DownBlock2D",  # a regular ResNet downsampling block
#                                    "DownBlock2D",
#                                    "DownBlock2D",
#                                    "DownBlock2D",
#                                    "CrossAttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
#                                    "DownBlock2D",
#                                ),
#                                up_block_types=(
#                                    "UpBlock2D",  # a regular ResNet upsampling block
#                                    "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
#                                    "UpBlock2D",
#                                    "UpBlock2D",
#                                    "UpBlock2D",
#                                    "UpBlock2D",
#                                ),
#                                cross_attention_dim=768, # TODO (KLAUS) : EXTRACT THIS NUMBER FROM CLIP MODEL
#                                )
    unet.train()    

# Optimizer 
    optimizer = torch.optim.AdamW(unet.parameters(), lr=config.optimizer.learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.warmup_steps,
        num_training_steps=config.training.max_steps,
    )

# ACCELERATE
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler)
    text_encoder.to(accelerator.device,dtype=config.training.mixed_precision[1])


# NOISE SCHEDULAR

    #noise_scheduler = DDIMScheduler()

    # TODO (KLAUS): Add tunable paramter to SDE
    noise_scheduler = TorchSDE(config.sde.variable, config.sde.drift, config.sde.diffusion, config.sde.diffusion_matrix, config.sde.initial_variable_value, config.sde.max_variable_value, config.sde.min_sample_value, config.sde.module, config.sde.target, config.sde.drift_integral_form, config.sde.diffusion_integral_form, config.sde.diffusion_integral_decomposition, config.sde.drift_diagonal_form, config.sde.diffusion_diagonal_form, config.sde.diffusion_matrix_diagonal_form)

# TRAIN

    global_step = 0
    
    config.training.epochs = math.ceil(config.training.max_steps / num_update_steps_per_epoch)

    epochs = tqdm(range(config.training.epochs), desc="Epoch ... ", position=0)

    for _epoch in epochs:
        
        steps_per_epoch = len(train_dataset) // config.training.total_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        train_loss = 0.0
        for batch in train_dataloader:
            
      
            with accelerator.accumulate(unet):

                clean_images = batch["pixel_values"]


                #z = torch.randn(clean_images.shape).to(clean_images.device)

                batch_size_z = clean_images.shape[0]

                # TODO (KLAUS) : CONTINOUOS SDE --> CONTINOUOS TIME

                #timesteps = torch.randint(
                #    0, # TODO (KLAUS) : SHOULD BE SDE MIN
                #    noise_scheduler.config.num_train_timesteps,
                #    (batch_size_z,),
                #    device=clean_images.device
                #).long()

                timesteps = torch.rand((batch_size_z,), device=clean_images.device) *(noise_scheduler.max_variable_value-noise_scheduler.min_sample_value) + noise_scheduler.min_sample_value

                #timesteps = torch.randint(
                #    noise_scheduler.initial_variable_value, # TODO (KLAUS) : SHOULD BE SDE MIN
                #    noise_scheduler.max_variable_value,
                #    (batch_size_z,),
                #    device=clean_images.device
                #).long()
                key, subkey = jax.random.split(key)

                noisy_images, z = noise_scheduler.sample(timesteps, clean_images, subkey, device=accelerator.device)
#                log_image = (noisy_images / 2 + 0.5).clamp(0,1)
#                log_image = numpy_to_pil(log_image.cpu().permute(0,2,3,1).numpy())
#
#                image_grid = make_image_grid(log_image, rows=4,cols=4)
#                accelerator.log({"image": wandb.Image(image_grid)}, step=global_step)

                
                #noisy_images = noise_scheduler.add_noise(clean_images, z, timesteps)

                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample


                if config.sde.target == "epsilon":
                    target = z
                else:
                    raise ValueError(f"Unknown prediction type {config.sde.target}")
#                if noise_scheduler.config.prediction_type == "epsilon":
#                    target = z
#                else:
#                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
#
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") 

                avg_loss = accelerator.gather(loss.repeat(config.training.batch_size)).mean()
                train_loss += avg_loss.item()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.optimizer.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(logs, step=global_step)
            global_step+= 1

            if global_step >= config.training.max_steps:
                break


        if accelerator.is_main_process: 
             
            pipeline = UTTIPipeline(accelerator.unwrap_model(unet), noise_scheduler, tokenizer, accelerator.unwrap_model(text_encoder))

            # TODO (KLAUS): SAMPLE RANDOM PROMPTS FROM THE DATASET
            prompts=["a drawing of a green pokemon with red eyes", "a red and white ball with an angry look on its face", "a cartoon butterfly with a sad look on its face", "a cartoon character with a smile on his face", "a blue and white bird with a long tail", "a blue and black object with two eyes", "a drawing of a bird with its mouth open", "a green bird with a red tail and a black nose", "drawing of a sheep with a bell on its head", "a black and yellow pokemon type animal"]
            #prompts = ["0", "1", "2", "3", "4", "5"]

            images = pipeline(prompts, key, accelerator.device, generator=torch.manual_seed(config.training.seed), num_inference_steps=1000, gen_twice=bool(_epoch % 2)).images
            image_grid = make_image_grid(images, rows=3,cols=4)
            accelerator.log({"image": wandb.Image(image_grid)}, step=global_step)

            if _epoch % 1000:
                save_local_cloud(config, None, pipeline, interface="torch", accelerator=accelerator)

    if accelerator.is_main_process: 
        pipeline = UTTIPipeline(accelerator.unwrap_model(unet), noise_scheduler, tokenizer, accelerator.unwrap_model(text_encoder))

        save_local_cloud(config, None, pipeline, interface="torch", accelerator=accelerator)
    
    accelerator.end_training()
        
if __name__ == "__main__":
    main()