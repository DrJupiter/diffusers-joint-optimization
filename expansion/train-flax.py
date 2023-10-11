import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'

import jax
import jax.numpy as jnp

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


# DATASET & TOKENIZER
    tokenizer = CLIPTokenizer.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="tokenizer"
    )

    train_dataset, train_dataloader = get_dataset(config, tokenizer)  

# MODELS
# TODO (KLAUS): INITIALIZE SDE AND ITS PARAMETERS
    text_encoder = FlaxCLIPTextModel.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="text_encoder", dtype=config.training.weight_dtype,
        cache_dir=config.training.cache_dir,
    )
    vae, vae_params = FlaxAutoencoderKL.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="vae", dtype=config.training.weight_dtype,
        cache_dir=config.training.cache_dir,
    )
    unet, unet_params = FlaxUNet2DConditionModel.from_pretrained(
        config.training.pretrained_model_or_path, revision=config.training.revision, subfolder="unet", dtype=config.training.weight_dtype,
        cache_dir=config.training.cache_dir,
    )    

# OPTIMIZER
    optimizer_scheduler = optax.constant_schedule(config.optimizer.learning_rate)

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

    noise_scheduler = FlaxDDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    noise_scheduler_state = noise_scheduler.create_state()


# Initialize our training
    rng = jax.random.PRNGKey(config.training.seed)
    train_rngs = jax.random.split(rng, jax.local_device_count())

# TRAIN STEP


    def train_step(state, text_encoder_params, vae_params, batch, train_rng):
        dropout_rng, sample_rng, new_train_rng = jax.random.split(train_rng, 3)

        def compute_loss(params):
            # Convert images to latent space
            vae_outputs = vae.apply(
                {"params": vae_params}, batch["pixel_values"], deterministic=True, method=vae.encode
            )
            latents = vae_outputs.latent_dist.sample(sample_rng)
            # (NHWC) -> (NCHW)
            latents = jnp.transpose(latents, (0, 3, 1, 2))
            latents = latents * vae.config.scaling_factor

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


    # Create parallel version of the train step
    p_train_step = jax.pmap(train_step, "batch", donate_argnums=(0,))

# Replicate the train state on each device
    state = jax_utils.replicate(state)
    text_encoder_params = jax_utils.replicate(text_encoder.params)
    vae_params = jax_utils.replicate(vae_params)

# TRAIN!
    num_update_steps_per_epoch = math.ceil(len(train_dataloader))

    if config.training.max_steps is None:
        config.training.max_steps = config.training.epochs * num_update_steps_per_epoch

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
            state, train_metric, train_rngs = p_train_step(state, text_encoder_params, vae_params, batch, train_rngs)
            train_metrics.append(train_metric)

            train_step_progress_bar.update(1)

            global_step += 1
            if global_step >= config.training.max_steps:
                break

        train_metric = jax_utils.unreplicate(train_metric)

        train_step_progress_bar.close()
        epochs.write(f"Epoch... ({epoch + 1}/{config.training.epochs} | Loss: {train_metric['loss']})")
        wandb.log({"loss": train_metric['loss']})

# SAVE PARAMETERS
    # TODO (KLAUS): SAVE THE OPTIMIZER's AND SDE's PARAMETERS too

        if jax.process_index() == 0:
            # TODO : WE NEED TO REPLACE THE PIPELINE WITH OUR OWN PIPELINE
            scheduler = FlaxPNDMScheduler(
                    beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", skip_prk_steps=True
                )

            safety_checker = FlaxStableDiffusionSafetyChecker.from_pretrained(
                    "CompVis/stable-diffusion-safety-checker", from_pt=True
                )
            pipeline = FlaxStableDiffusionPipeline(
                    text_encoder=text_encoder,
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                    scheduler=scheduler,
                    safety_checker=safety_checker,
                    feature_extractor=CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32"),
                )
            params = {
                        "text_encoder": get_params_to_save(text_encoder_params),
                        "vae": get_params_to_save(vae_params),
                        "unet": get_params_to_save(state.params),
                        "safety_checker": safety_checker.params,
                    }
            tokens = jnp.array(tokenizer(["a blue and black object with two eyes"], max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True).input_ids)
            image_grid = make_image_grid(pipeline(tokens, params, train_rngs[0])["images"])
            wandb.log({"image": wandb.Image(image_grid)}, step=global_step)
            save_local_cloud(config, params, pipeline)


if __name__ == "__main__":
    main()