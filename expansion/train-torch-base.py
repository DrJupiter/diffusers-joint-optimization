import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import math
from tqdm.auto import tqdm
import sys

import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


from config.config import Config
from config.utils import get_wandb_input, save_local_cloud, initialize_sde_parameter_plot, update_sde_parameter_plot 
from data.dataload import get_dataset 

from transformers import set_seed, CLIPTextModel, CLIPTokenizer

from diffusers.utils import check_min_version, make_image_grid
from diffusers.utils.pil_utils import numpy_to_pil

from pipelines.pipeline_tti_torch import UTTIPipeline, Noise, SDESolver

check_min_version("0.22.0.dev0")
import wandb

from diffusers import (UNet2DConditionModel)
from diffusers.optimization import get_cosine_schedule_with_warmup

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
    print(f"Training on device: {accelerator.device}")

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
    #text_encoder = torch.compile(text_encoder)

    # TODO Make this a class containing the SDE and the UNET
    if config.training.load_pretrained_model:

        unet = UNet2DConditionModel.from_pretrained(config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="unet", cache_dir=config.training.cache_dir)
        print("Loaded pretrained UNET")
    else:

        #unet = UNet2DConditionModel(sample_size=config.training.resolution,
        #                        in_channels=3,
        #                        out_channels=3,
        #                        cross_attention_dim=768 # TODO (KLAUS) : EXTRACT THIS NUMBER FROM CLIP MODEL
        #                        )

        unet = UNet2DConditionModel(sample_size=config.training.resolution,
                                    in_channels=3,
                                    out_channels=3,
                                    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
                                    down_block_types=(
                                        "DownBlock2D",  # a regular ResNet downsampling block
                                        "DownBlock2D",
                                        "DownBlock2D",
                                        "DownBlock2D",
                                        "CrossAttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
                                        "DownBlock2D",
                                    ),
                                    up_block_types=(
                                        "UpBlock2D",  # a regular ResNet upsampling block
                                        "CrossAttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
                                        "UpBlock2D",
                                        "UpBlock2D",
                                        "UpBlock2D",
                                        "UpBlock2D",
                                    ),
                                    cross_attention_dim=768, # TODO (KLAUS) : EXTRACT THIS NUMBER FROM CLIP MODEL
                                    )
    #unet = torch.compile(unet)
    unet.train()    
# NOISE SCHEDULAR

    #noise_scheduler = DDIMScheduler()

    if config.training.load_noise_scheduler:
        noise_scheduler = TorchSDE_PARAM.from_pretrained(config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="scheduler", cache_dir=config.training.cache_dir, device=accelerator.device)
        print("Loaded pretrained noise scheduler")
    else:
        noise_scheduler = TorchSDE_PARAM(
        device=accelerator.device,
        min_sample_value=config.sde.min_sample_value,
        data_dimension=config.sde.data_dim,
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
        diffusion_matrix_dimension=config.sde.diffusion_matrix_dimension,
        non_symbolic_parameters=getattr(config.sde, "non_symbolic_parameters", None),
    )

# Optimizer 
    optimizer = torch.optim.AdamW([{"params": unet.parameters()}, {"params": noise_scheduler.parameters(), "lr": config.optimizer.sde_learning_rate}], lr=config.optimizer.learning_rate)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=config.optimizer.warmup_steps,
        num_training_steps=config.training.max_steps,
    )
    if config.training.load_optimizer:
        from huggingface_hub import hf_hub_download

        optimizer_state = torch.load(hf_hub_download(config.training.pretrained_model_or_path, filename="optimizer.pt", revision=config.training.revision, subfolder="optimizer", cache_dir=config.training.cache_dir))
        optimizer.load_state_dict(optimizer_state["optimizer"])

        lr_scheduler_state = torch.load(hf_hub_download(config.training.pretrained_model_or_path, filename="lr_scheduler.pt", revision=config.training.revision, subfolder="lr_scheduler", cache_dir=config.training.cache_dir))
        lr_scheduler.load_state_dict(lr_scheduler_state["lr_scheduler"])
        print("Loaded pretrained optimizer and lr scheduler")
# ACCELERATE
    unet, optimizer, train_dataloader, lr_scheduler, noise_scheduler = accelerator.prepare(unet, optimizer, train_dataloader, lr_scheduler, noise_scheduler)

    text_encoder.to(accelerator.device,dtype=config.training.mixed_precision[1])
    


# TRAIN

    global_step = 0
    
    if accelerator.is_main_process:
            _log_drift_param, _log_diffusion_param = noise_scheduler.parameters() 
            update_sde_parameter_plot(sde_param_plots[0], global_step, *_log_drift_param.detach())
            update_sde_parameter_plot(sde_param_plots[1], global_step, *_log_diffusion_param.detach())
            accelerator.log({"Drift Parameters": sde_param_plots[0], "Diffusion Parameters": sde_param_plots[1]}, step=global_step) 

    config.training.epochs = math.ceil(config.training.max_steps / num_update_steps_per_epoch)

    epochs = tqdm(range(config.training.epochs), desc="Epoch ... ", position=0)

    for _epoch in epochs:
        
        steps_per_epoch = len(train_dataset) // config.training.total_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        for batch in train_dataloader:
            with accelerator.accumulate(unet):

                clean_images = batch["pixel_values"]


                batch_size_z = clean_images.shape[0]



                timesteps = torch.rand((batch_size_z,), device=clean_images.device) *(noise_scheduler.max_variable_value-noise_scheduler.min_sample_value) + noise_scheduler.min_sample_value

                #timesteps = torch.randint(
                #    noise_scheduler.initial_variable_value, # TODO (KLAUS) : SHOULD BE SDE MIN
                #    noise_scheduler.max_variable_value,
                #    (batch_size_z,),
                #    device=clean_images.device
                #).long()
                #key, subkey = jax.random.split(key)
                #noisy_images, z = noise_scheduler.sample(timesteps, clean_images, subkey, device=accelerator.device)

                if accelerator.device.type == "mps":

                    noise = torch.randn_like(clean_images)
                    noise = noise.to(clean_images.device)
                else:
                    noise = torch.randn_like(clean_images, device=clean_images.device)
                
                noisy_images = noise_scheduler.sample(timesteps, clean_images.reshape(batch_size_z,-1), noise.reshape(batch_size_z,-1),*noise_scheduler.parameters(), device=accelerator.device).reshape(clean_images.shape)


                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                model_pred = unet(noisy_images, timesteps, encoder_hidden_states).sample


                if config.sde.target == "epsilon":
                    target = noise
                else:
                    raise ValueError(f"Unknown prediction type {config.sde.target}")

                # Regulizar term
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean") + (F.mse_loss(noisy_images.float(), clean_images.float(), reduction="none").reshape(batch_size_z, -1) * torch.exp(-((timesteps)**2 + 4 * timesteps**4 + 6*timesteps**5)).reshape(batch_size_z, 1)).mean()
                #loss = noise_scheduler.scaled_loss(timesteps, target.float().reshape(batch_size_z,-1), model_pred.float().reshape(batch_size_z,-1), *noise_scheduler.parameters(), device=accelerator.device).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config.optimizer.max_grad_norm)
                    torch.nn.utils.clip_grad_value_(noise_scheduler.parameters(), config.optimizer.max_grad_norm)
                    #accelerator.clip_grad_norm_(noise_scheduler.parameters(), config.optimizer.max_grad_norm) # TODO (KLAUS): POTENTIALLY UNDESERIED
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            accelerator.log(logs, step=global_step)
            global_step+= 1

            if global_step >= config.training.max_steps:
                break


        if accelerator.is_main_process: 

            _log_drift_param, _log_diffusion_param = noise_scheduler.parameters() 
            update_sde_parameter_plot(sde_param_plots[0], global_step, *_log_drift_param.detach())
            update_sde_parameter_plot(sde_param_plots[1], global_step, *_log_diffusion_param.detach())
            accelerator.log({"Drift Parameters": sde_param_plots[0], "Diffusion Parameters": sde_param_plots[1]}, step=global_step) 

            if (_epoch % 50) == 0 or (config.debug and (_epoch % 5) == 0):
                unwrapped_unet = accelerator.unwrap_model(unet)
                unwrapped_unet.eval()
                pipeline = UTTIPipeline(unwrapped_unet, accelerator.unwrap_model(noise_scheduler), tokenizer, accelerator.unwrap_model(text_encoder))

                # TODO (KLAUS): SAMPLE RANDOM PROMPTS FROM THE DATASET
                #prompts=["a drawing of a green pokemon with red eyes", "a red and white ball with an angry look on its face", "a cartoon butterfly with a sad look on its face", "a cartoon character with a smile on his face", "a blue and white bird with a long tail", "a blue and black object with two eyes", "a drawing of a bird with its mouth open", "a green bird with a red tail and a black nose", "drawing of a sheep with a bell on its head", "a black and yellow pokemon type animal","a drawing of a red and black dragon", "a purple and green animal with a blue nose"]
                #prompts = ["0", "1", "2", "3", "4", "5"]
                prompts=["Ghost attribute, with a yellow zipper on the mouth, bloodshot glasses, gray body", "Grass attribute, with a yellow zipper on the mouth, bloodshot glasses, gray body", "Grass attributes, the top of the head is a tuft of hair that looks like a tall and straight grass, fan-shaped big ears, white eyebrows, thick and long emerald green tail",  "Ghost attributes, the top of the head is a tuft of hair that looks like a tall and straight grass, fan-shaped big ears, white eyebrows, thick and long emerald green tail", "Ghost attribute, the whole body is gray and pink, two big eyes, no feet", "Steel attribute, huge sword body, hilt, sword tan, and golden-yellow sword spine, deep purple eyes and palm, black arms, jagged blade", "Insect attribute, huge sword body, hilt, sword tan, and golden-yellow sword spine, deep purple eyes and palm, black arms, jagged blade", "Steel attribute, huge ice body, and golden-yellow spine, deep purple eyes and palm, black arms", "Fairy attribute, head resembling an owl, purple feathers, pink belly feathers", "Water attribute, head resembling an owl, blue feathers, pink belly feathers", "Insect-like, with a dark blue-purple body, huge white wings with black veins, four pink-blue legs, and light red eyes", "Insect-like, with a bright blue-purple body, huge green wings with black veins, four pink-blue legs, and light red eyes"]

                noise_type = random.choice(noise_types)
                images = pipeline(prompts, accelerator.device, generator=torch.manual_seed(config.training.seed), num_inference_steps=1000, noise=noise_type, method=SDESolver.EULER, debug=config.debug).images
                image_grid = make_image_grid(images, rows=3,cols=4)
                accelerator.log({f"image-{noise_type}": wandb.Image(image_grid)}, step=global_step)

                if (_epoch % 100) == 0 and not config.debug:
                    save_local_cloud(config, {"optimizer": accelerator.unwrap_model(optimizer).state_dict(), "lr_scheduler": accelerator.unwrap_model(lr_scheduler).state_dict()}, pipeline, interface="torch", accelerator=accelerator)
                    #save_local_cloud(config, None, pipeline, interface="torch", accelerator=accelerator)

                unwrapped_unet.train()

    if accelerator.is_main_process: 
        pipeline = UTTIPipeline(accelerator.unwrap_model(unet), noise_scheduler, tokenizer, accelerator.unwrap_model(text_encoder))

        save_local_cloud(config, None, pipeline, interface="torch", accelerator=accelerator)
    
    accelerator.end_training()
        
if __name__ == "__main__":
    main()