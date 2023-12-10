# Copyright 2023 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from typing import List, Optional, Tuple, Union

import torch

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput

from enum import Enum, auto
class Noise(Enum):
    STANDARD_NORMAL = auto()
    DISTRIBUTION = auto()
    ZERO = auto() # zero is the same as sampling noise from the distribution right after the last time step in the reverse SDE.
    def __str__(self):
        return self.name

class SDESolver(Enum):
    EULER = auto()

    def __str__(self) -> str:
        return self.name



class UTTIPipeline(DiffusionPipeline):
    r"""
    Pipeline for image generation.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Parameters:
        unet ([`UNet2DModel`]):
            A `UNet2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image. Can be one of
            [`DDPMScheduler`], or [`DDIMScheduler`].
    """
    model_cpu_offload_seq = "unet"

    def __init__(self, unet, scheduler, tokenizer, text_encoder):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, tokenizer=tokenizer, text_encoder=text_encoder)

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        device: str = "cuda",
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        noise: Noise = Noise.STANDARD_NORMAL,
        method: SDESolver = SDESolver.EULER, # TODO (KLAUS) : CODE IN THE OTHER SOLVERS
        debug: bool = False,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """


        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        text_input = self.tokenizer(prompt, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids.to(self._execution_device)

        prompt_embeddings = self.text_encoder(text_input)[0]

        # Sample gaussian noise to begin loop
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)
        match noise:
            case Noise.STANDARD_NORMAL:

                if self.device.type == "mps":
                    # randn does not work reproducibly on mps
                    image = randn_tensor(image_shape, generator=generator)
                    image = image.to(self.device)
                else:
                    image = randn_tensor(image_shape, generator=generator, device=self.device)
            case Noise.DISTRIBUTION:
                if self.device.type == "mps":
                    image = torch.zeros(image_shape, device=self.device)
                    timesteps = torch.ones((batch_size,), dtype=torch.long, device=self.device) * self.scheduler.max_variable_value
                    # randn does not work reproducibly on mps
                    noise = randn_tensor(image_shape, generator=generator)
                    noise = noise.to(self.device)
                    image = self.scheduler.sample(timesteps, image, noise, *self.scheduler.parameters(), device=self.device)
                else:
                    image = torch.zeros(image_shape, device=self.device)
                    timesteps = torch.ones((batch_size,), dtype=torch.long, device=self.device) * self.scheduler.max_variable_value
                    # randn does not work reproducibly on mps
                    noise = randn_tensor(image_shape, generator=generator, device=self.device)
                    image = self.scheduler.sample(timesteps, image, noise, *self.scheduler.parameters(), device=self.device)
            case Noise.ZERO:
                if self.device.type == "mps":
                    image = torch.zeros(image_shape, device=self.device)
                else:
                    image = torch.zeros(image_shape, device=self.device)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps, (1), device)
        dt = torch.tensor([-(self.scheduler.max_variable_value-self.scheduler.min_sample_value)/(num_inference_steps-1)]).to(device)

        def denoise(image, prompt_embeddings):
            for t in self.progress_bar(self.scheduler.timesteps):
                # 1. predict noise model_output
                model_output = self.unet(image, t, encoder_hidden_states=prompt_embeddings).sample

                # 2. compute previous image: x_t -> x_t-1
                if debug:
                    if torch.isnan(model_output).sum() > 0 or torch.isinf(model_output).sum() > 0:
                        print(t)
                        print(image.max(), image.min())
                        print("nan value or inf from model output")
                        print(torch.isnan(model_output).sum(), torch.isinf(model_output).sum())
                        print(f"nan values in image {torch.isnan(image).sum()}")
                        break
                noise = randn_tensor(image_shape, device=self.device) 
                reverse_time_derivative =  self.scheduler.reverse_time_derivative(t.repeat(batch_size), image, noise, model_output, *self.scheduler.parameters(), device)

                image = self.scheduler.step(image, reverse_time_derivative, dt)
                
                if debug:
                    if torch.isnan(image).sum() > 0 or torch.isinf(image).sum() > 0:
                        print(t)
                        print("nan values in image")
                        break
            return image
        image = denoise(image, prompt_embeddings)
       


        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
