
from .config import Config
import jax

from huggingface_hub import create_repo, upload_folder, HfFolder, whoami
import os

from typing import List

import plotly.graph_objs as go
from plotly.subplots import make_subplots
import torch

# LOGGING

def get_wandb_input(config: Config):

    args = {}
    args["entity"] = config.logging.entity
    args["project"] = config.logging.project

    tags = [config.logging.experiment, config.training.target, config.sde.name, config.training.dataset_name]
   
    args["tags"] = tags
    return args 

# SAVING MODEL

def get_params_to_save(params):
    return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], params))


def get_full_repo_name(model_id: str, organization: str = None, token: str = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"

def save_local_cloud(config: Config, params, pipeline, interface="jax", accelerator=None):
    # Handle the repository creation

    if interface == "jax":

        check = jax.process_index() == 0
    elif interface == "torch":
        check = accelerator.is_main_process

    if check:
        
        os.makedirs(config.training.save_dir, exist_ok=True)

        if config.training.push_to_hub:
            repo_id = create_repo(
                repo_id=get_full_repo_name(config.training.repo_name), exist_ok=True, token=HfFolder.get_token()
            ).repo_id

         
        #pipeline = TODO (KLAUS): CHANGE THIS TO OUR PIPELINE 
        if interface == "jax":
            pipeline.save_pretrained(
                config.training.save_dir,
                params=params
            )
        elif interface == "torch":
            pipeline.save_pretrained(
                config.training.save_dir
            )
            if params is not None:
                
                optimizer_state = params.pop("optimizer", None)
                if optimizer_state is not None:
                    os.makedirs(os.path.join(config.training.save_dir, "optimizer"), exist_ok=True)
                    torch.save({"optimizer": optimizer_state}, os.path.join(config.training.save_dir, "optimizer/optimizer.pt"))
                lr_scheduler_state = params.pop("lr_scheduler", None)
                if lr_scheduler_state is not None:
                    os.makedirs(os.path.join(config.training.save_dir, "lr_scheduler"), exist_ok=True)
                    torch.save({"lr_scheduler": lr_scheduler_state}, os.path.join(config.training.save_dir, "lr_scheduler/lr_scheduler.pt"))

        if config.training.push_to_hub:
            upload_folder(
                repo_id=repo_id,
                folder_path=config.training.save_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*"],
            )


def initialize_sde_parameter_plot(config: Config):

    def make_plot(parameters, name):
        line_plot = make_subplots(1, 1, shared_xaxes=True, shared_yaxes=False)
        for p in parameters:
            line_plot.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name=f"{name}: {p}"))
        return line_plot
    
    drift_param_plot = make_plot(config.sde.drift_parameters, "Drift")
    diffusion_param_plot = make_plot(config.sde.diffusion_parameters, "Diffusion")
    return drift_param_plot, diffusion_param_plot

def update_sde_parameter_plot(plot, step, *args):

    for i, arg in enumerate(args):
        plot.data[i].x = list(plot.data[i].x) + [step]
        plot.data[i].y = list(plot.data[i].y) + [arg.item()]



