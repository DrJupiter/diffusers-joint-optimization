
import matplotlib.pyplot as plt

"""
from diffusers import DDPMPipeline
ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")
images = ddpm(num_inference_steps=1000).images
"""

from diffusers import DDPMScheduler, UNet2DModel, UNet2DConditionModel

name = "google/ddpm-cat-256"

scheduler = DDPMScheduler.from_pretrained(name) 
model = UNet2DModel.from_pretrained(name).to('cuda')
scheduler.set_timesteps(50)

print(scheduler.timesteps)

import torch 

sample_size = model.config.sample_size

noise = torch.randn((1,3, sample_size, sample_size)).to('cuda')

sample = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(sample, t).sample
    
    previous_sample = scheduler.step(noisy_residual, t, sample).prev_sample
    sample = previous_sample

import numpy as np

image = (sample / 2 + 0.5).clamp(0,1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()

from PIL import Image
plt.imshow(Image.fromarray(image))
plt.show()


def display_images(images, titles = [], rows = None, columns = 2, figsize= (7,7), pad=0.2, log_title=None):
    """
    Takes a list of images and plots them

    Takes the config, so we know how to plot the image in accordance with the dataset
    """
    
    if rows is None:
        rows = len(images)

    fig = plt.figure(figsize=figsize)

    # Title correction

    for idx, img in enumerate(images):
        fig.add_subplot(rows, columns, idx+1) 
   
        plt.imshow(img)
        plt.axis('off')
        if len(titles) == len(images):
            plt.title(titles[idx])
        else:
            plt.title(str(idx+1))
    plt.tight_layout(pad=pad) 

    plt.show()
    plt.close()
