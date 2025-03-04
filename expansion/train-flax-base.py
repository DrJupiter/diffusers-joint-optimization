import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
import jax.numpy as jnp

# load DNN library 
jax.random.PRNGKey(0)

import math
from tqdm.auto import tqdm

from config.config import Config
from config.utils import get_wandb_input, save_local_cloud, get_params_to_save
from data.dataload import get_dataset

from transformers import set_seed, FlaxCLIPTextModel, CLIPImageProcessor, CLIPTokenizer

from diffusers.pipelines.stable_diffusion import FlaxStableDiffusionSafetyChecker

from diffusers.utils import check_min_version, make_image_grid
# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.22.0.dev0")

from PIL import Image

import wandb


from diffusers import (
    FlaxAutoencoderKL,
    FlaxDDPMScheduler,
    FlaxPNDMScheduler,
    FlaxStableDiffusionPipeline,
    FlaxUNet2DConditionModel,
)

import optax

from flax.training import train_state

from flax import jax_utils

from flax.training.common_utils import shard

import numpy as np

def main():

# GET CONFIG & LOGGING
    config = Config()

    wandb.init(**get_wandb_input(config)) 

# SET SEED
    if config.training.seed is not None:
        set_seed(config.training.seed)

# Initialize RNG
    rng = jax.random.PRNGKey(config.training.seed)



# DATASET & TOKENIZER
    tokenizer = CLIPTokenizer.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="tokenizer"
    )

    train_dataset, train_dataloader = get_dataset(config, tokenizer)  

# CALCULATE NUMBER OF TRAINING STEPS
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    if config.training.max_steps is None:
        config.training.max_steps = config.training.epochs * num_update_steps_per_epoch
# MODELS
# TODO (KLAUS): INITIALIZE SDE AND ITS PARAMETERS
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="text_encoder", dtype=config.training.weight_dtype,
        cache_dir=config.training.cache_dir,
    )
  
#    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
#        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="unet", dtype=config.training.weight_dtype,
#        cache_dir=config.training.cache_dir,
#    )    
    unet = FlaxUNet2DConditionModel(sample_size=32, 
                                        in_channels=3,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=3,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    cross_attention_dim=768,)
    unet_params = unet.init_weights(rng)

# OPTIMIZER
    if config.optimizer.lr_scheduler == "constant":
        optimizer_scheduler = optax.constant_schedule(config.optimizer.learning_rate)
    elif config.optimizer.lr_scheduler == "cosine":
        optimizer_scheduler = optax.warmup_cosine_decay_schedule(
            init_value=config.optimizer.init_value,
            peak_value=config.optimizer.learning_rate,
            warmup_steps=config.optimizer.warmup_steps,
            decay_steps=config.training.max_steps
            )

    adamw = optax.adamw(
        learning_rate=optimizer_scheduler,
        b1=config.optimizer.adam_beta1,
        b2=config.optimizer.adam_beta2,
        eps=config.optimizer.adam_epsilon,
        weight_decay=config.optimizer.adam_weight_decay,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(config.optimizer.max_grad_norm),
        adamw,
    )

# STATE

    # TODO (KLAUS): CHANGE STATE TO TAKE SDE PARAMETERS TOO
    state = train_state.TrainState.create(apply_fn=unet.__call__, params=unet_params, tx=optimizer)

# SDE/NOISE SCHEDULER

    noise_scheduler = FlaxDDPMScheduler()
    noise_scheduler_state = noise_scheduler.create_state()


# TRAIN STEP

    train_rngs = jax.random.split(rng, jax.local_device_count())

    def train_step(state, text_encoder_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        def compute_loss(params):
            # Convert images to latent space
            # sample latents 
            latents = batch["pixel_values"]
            #print(latents.shape)
            #latents = jnp.transpose(latents, (0, 3, 1, 2))
            #latents = latents

            # Sample noise that we'll add to the latents
            noise_rng, timestep_rng = jax.random.split(sample_rng)
            noise = jax.random.normal(noise_rng, latents.shape)
            # Sample a random timestep for each image
            bsz = latents.shape[0]
            timesteps = jax.random.randint(
                timestep_rng,
                (bsz,),
                0,
                noise_scheduler.config.num_train_timesteps,
            )

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

            # Get the text embedding for conditioning
            encoder_hidden_states = text_encoder(
                batch["input_ids"],
                params=text_encoder_params,
                train=False,
            )[0]
            # Predict the noise residual and compute loss
            model_pred = unet.apply(
                {"params": params}, noisy_latents, timesteps, encoder_hidden_states, train=True
            ).sample

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(noise_scheduler_state, latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

            loss = (target - model_pred) ** 2
            loss = loss.mean()

            return loss

        grad_fn = jax.value_and_grad(compute_loss)
        loss, grad = grad_fn(state.params)
        grad = jax.lax.pmean(grad, "batch")

        new_state = state.apply_gradients(grads=grad)

        metrics = {"loss": loss}
        metrics = jax.lax.pmean(metrics, axis_name="batch")

        return new_state, metrics, new_train_rng

    
    def setup_noise(text_encoder_params, batch, train_rng):
        sample_rng, _ = jax.random.split(train_rng, 2)

        latents = jnp.zeros_like(batch["pixel_values"])
        #print(latents.shape)
        #latents = jnp.transpose(latents, (0, 3, 1, 2))
        #latents = latents

        # Sample noise that we'll add to the latents
        noise_rng, _ = jax.random.split(sample_rng)
        noise = jax.random.normal(noise_rng, latents.shape)
        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = (jax.numpy.ones((bsz,)) * int(noise_scheduler.config.num_train_timesteps-1)).astype(int)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = noise_scheduler.add_noise(noise_scheduler_state, latents, noise, timesteps)

        # Get the text embedding for conditioning
        encoder_hidden_states = text_encoder(
            batch["input_ids"],
            params=text_encoder_params,
            train=False,
        )[0]

        return timesteps, encoder_hidden_states, noisy_latents

    p_setup_noise = jax.pmap(setup_noise, "batch") 

    def inference_step(state, time, noisy_image, timesteps, encoder_hidden_states):
        model_pred = unet.apply(
                {"params": state.params}, noisy_image, timesteps, encoder_hidden_states, train=False
            ).sample
        noisy_image = noise_scheduler.step(noise_scheduler_state, model_pred, time, noisy_image).prev_sample
        timesteps -= 1
        return state, noisy_image, timesteps, encoder_hidden_states 
    p_inference_step = jax.pmap(inference_step, donate_argnums=(0,1,))

    def generate_image(state,text_encoder_params, batch, train_rng):
        
        # Predict the noise residual and compute loss

#        for _ in reversed(range(noise_scheduler.config.num_train_timesteps)):
#            model_pred = unet.apply(
#                {"params": state.params}, noisy_latents, timesteps, encoder_hidden_states, train=False
#            ).sample
#            noisy_latents = noise_scheduler.step(noise_scheduler_state, model_pred, _, noisy_latents).prev_sample
#            timesteps -= 1

        timesteps, encoder_hidden_states, noisy_image = p_setup_noise(text_encoder_params, batch, train_rng)
        for t in noise_scheduler_state.timesteps:

            state, noisy_image, timesteps, encoder_hidden_states = p_inference_step(state, jnp.array([[t]]), noisy_image, timesteps, encoder_hidden_states) 

        noisy_image = jax_utils.unreplicate(noisy_image)
        image = jnp.clip(noisy_image/ 2 + 0.5, 0, 1)
        image = jnp.round(jnp.transpose(image, (0, 2, 3, 1))*255)
        return state, image
    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))
    p_generate_image = generate_image #jax.pmap(generate_image, "batch", donate_argnums=(0,))

# Replicate the train state on each device
    state = jax_utils.replicate(state)
    text_encoder_params = jax_utils.replicate(text_encoder.params)

# TRAIN!


    config.training.epochs = math.ceil(config.training.max_steps / num_update_steps_per_epoch)


    global_step = 0

    epochs = tqdm(range(config.training.epochs), desc="Epoch ... ", position=0)

    for epoch in epochs:
        # ======================== Training ================================

        train_metrics = []

        steps_per_epoch = len(train_dataset) // config.training.total_batch_size
        train_step_progress_bar = tqdm(total=steps_per_epoch, desc="Training...", position=1, leave=False)
        # train
        for batch in train_dataloader:
            batch = shard(batch)
            state, train_metric, train_rngs = p_train_step(state, text_encoder_params, batch, train_rngs)
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            global_step += 1
            if global_step >= config.training.max_steps:
                break

        train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{config.training.epochs} | Loss: {train_metric['loss']})")
        wandb.log({"loss": train_metric['loss']}, step = global_step)

# SAVE PARAMETERS
    # TODO (KLAUS): SAVE THE OPTIMIZER's AND SDE's PARAMETERS too
        
        if (jax.process_index() == 0) and (global_step % 10 == 0):
            state, images = p_generate_image(state, text_encoder_params, batch, train_rngs)
            wandb.log({"image": wandb.Image(make_image_grid([Image.fromarray(image) for image in np.array(images.astype(np.uint8))], 4,4))}, step = global_step)

if __name__ == "__main__":
    main()