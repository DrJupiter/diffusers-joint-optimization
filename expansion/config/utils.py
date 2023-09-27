
from .config import Config
import jax

from huggingface_hub import create_repo, upload_folder, HfFolder, whoami
import os

from diffusers.pipelines import FlaxDiffusionPipeline

# LOGGING

def get_wandb_input(config: Config):

    args = {}
    args["entity"] = config.logging.entity
    args["project"] = config.logging.project

    tags = [config.logging.experiment, config.training.target, config.sde.name, config.training.dataset_name]
   
    args["tags"] = tags
    return args 

# SAVING MODEL

def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def save_local_cloud(config: Config, params):
    # Handle the repository creation
    if jax.process_index() == 0:
        
        os.makedirs(config.training.save_dir, exist_ok=True)

        if config.training.push_to_hub:
            repo_id = create_repo(
                repo_id=get_full_repo_name(config.training.save_dir), exist_ok=True, token=HfFolder.get_token()
            ).repo_id

        # TODO (KLAUS): CHANGE THIS TO OUR PIPELINE 
        pipeline = FlaxDiffusionPipeline()

        pipeline.save_pretrained(
            config.training.save_dir,
            params=params
        )

        if config.training.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=config.training.save_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )