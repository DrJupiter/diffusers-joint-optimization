import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import math
from tqdm.auto import tqdm

import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration


from config.config import Config
from config.utils import get_wandb_input
import plotly
from data.dataload import get_dataset 

from transformers import set_seed, CLIPTextModel, CLIPTokenizer

from diffusers.utils import check_min_version, make_image_grid, numpy_to_pil

from diffusers import DDIMScheduler, DDPMScheduler


from pipelines.pipeline_tti_torch import UTTIPipeline, Noise, SDESolver

check_min_version("0.22.0.dev0")
import wandb


#from sde_torch import TorchSDE 
from sde_torch_param import TorchSDE_PARAM

def main():

    config = Config()

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
    print(f"Running test on device: {accelerator.device}")

    if accelerator.is_main_process:
        log_kwargs = {"wandb": get_wandb_input(config)}
        project_name = log_kwargs["wandb"].pop("project")
        accelerator.init_trackers(project_name, init_kwargs=log_kwargs)
        noise_types = list(Noise)
# LOAD DATA


    tokenizer = CLIPTokenizer.from_pretrained(
        config.training.pretrained_model_or_path, cache_dir=config.training.cache_dir, revision=config.training.revision, subfolder="tokenizer"
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

# NOISE SCHEDULAR


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
    diffusion_matrix_dimension=config.sde.diffusion_matrix_dimension
)

    #noise_scheduler = DDIMScheduler()
    #noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    
    if isinstance(noise_scheduler, DDIMScheduler) or isinstance(noise_scheduler, DDPMScheduler):
        noise_scheduler.sample = lambda t, x, noise, *args, **kwargs: noise_scheduler.add_noise(x, noise, t).to(accelerator.device)
        noise_scheduler.parameters = lambda: [[0],[0]]

    text_encoder.to(accelerator.device,dtype=config.training.mixed_precision[1])




# TRAIN

    global_step = 0

    config.training.epochs = math.ceil(config.training.max_steps / num_update_steps_per_epoch)


    sample = next(iter(train_dataloader))
    sample_images = sample["pixel_values"]

    def get_noise(image):
        if accelerator.device.type == "mps":
            
            noise = torch.randn_like(image)
            noise = noise.to(image.device)
        else:
            noise = torch.randn_like(image, device=image.device)
        return noise
    def noisey_image_overtime(images, n=10):

        if isinstance(noise_scheduler, DDIMScheduler) or isinstance(noise_scheduler, DDPMScheduler):
            noise_scheduler.set_timesteps(n, device=accelerator.device)
        else:

            noise_scheduler.set_timesteps(n, (1), device=accelerator.device)
        timesteps = noise_scheduler.timesteps
        print(timesteps)
        log_images = []
        for image in images:
            copy_image = image.reshape(1,-1)
            copy_image = copy_image.repeat(n,1)
            noise = get_noise(copy_image)
            noisy_images = noisy_image(timesteps, copy_image, noise).reshape(n, *image.shape)
            noise_images = images_to_pil(noisy_images)
            log_images += noise_images[::-1] # reverse the reverse time

        image_grid = make_image_grid(log_images, rows=images.shape[0],cols=n)
        accelerator.log({"image-noise": wandb.Image(image_grid)}, step=global_step)

    def images_to_pil(images):
        images = (images / 2 + 0.5 ).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy(force=True)
        return numpy_to_pil(images)

    def noisy_image(time, image, noise):
        return noise_scheduler.sample(time, image.reshape(image.shape[0], -1), noise, *noise_scheduler.parameters(), device=accelerator.device).reshape(image.shape)
    noisey_image_overtime(sample_images, n=100)

    accelerator.end_training()

    """

    pipeline = UTTIPipeline(unwrapped_unet, accelerator.unwrap_model(noise_scheduler), tokenizer, accelerator.unwrap_model(text_encoder))

    # TODO (KLAUS): SAMPLE RANDOM PROMPTS FROM THE DATASET
    prompts=["a drawing of a green pokemon with red eyes", "a red and white ball with an angry look on its face", "a cartoon butterfly with a sad look on its face", "a cartoon character with a smile on his face", "a blue and white bird with a long tail", "a blue and black object with two eyes", "a drawing of a bird with its mouth open", "a green bird with a red tail and a black nose", "drawing of a sheep with a bell on its head", "a black and yellow pokemon type animal","a drawing of a red and black dragon", "a brown and white animal with a black nose"]
    #prompts = ["0", "1", "2", "3", "4", "5"]

    noise_type = random.choice(noise_types)
    images = pipeline(prompts, accelerator.device, generator=torch.manual_seed(config.training.seed), num_inference_steps=1000, noise=noise_type, method=SDESolver.EULER).images
    image_grid = make_image_grid(images, rows=3,cols=4)
    accelerator.log({f"image-{noise_type}": wandb.Image(image_grid)}, step=global_step)

    

    # Make plots with plotly over the initial time and end time for the sde and the decompostions and also make a perfect score which always predicts the noise correctly

    """






if __name__ == "__main__":
    main()