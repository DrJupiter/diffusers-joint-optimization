
from datasets import load_dataset

import os

import numpy as np

import random

from torchvision import transforms

import torch

import jax

from PIL import Image

dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

from config.utils import Config

# TODO (KLAUS): FILL OUT CONFIG ATTRIBUTES

def get_dataset(config: Config, tokenizer, interface: str = "torch", accelerator = None):

    assert interface in ["torch", "jax"], f"Iterface {interface} not found place choose torch/jax"
    if interface == "torch":
        assert accelerator is not None, f"For the {interface} you must supply an accelerator"

# LOAD DATASET
    if config.training.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            config.training.dataset_name,
            config.training.dataset_config_name,
            cache_dir=config.training.cache_dir,
        )
    else:
        data_files = {}
        if config.training.train_data_dir is not None:
            data_files["train"] = os.path.join(config.training.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=config.training.cache_dir,
        )
    
    column_names = dataset["train"].column_names


# MAKE SURE THE KEY FOR ACCESING IMAGES AND LABELS IS CORRECT

    dataset_columns = dataset_name_mapping.get(config.training.dataset_name, None)

    if config.training.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = config.training.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{config.training.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    if config.training.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = config.training.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{config.training.caption_column}' needs to be one of: {', '.join(column_names)}"
            )


# PREPROCESS THE DATASET

    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.training.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.training.resolution) if config.training.center_crop else transforms.RandomCrop(config.training.resolution),
            transforms.RandomHorizontalFlip() if config.training.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            elif config.training.try_convert_label_string:
                captions.append(str(caption))
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
            
        if interface == "jax":
            inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        elif interface == "torch":
            inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
            
        input_ids = inputs.input_ids
        return input_ids


    def image_preprocess(image, background_color=(255,255,255)):
        if image.mode == 'RGBA':
            # Create a blank background image
            background = Image.new('RGB', image.size, background_color)
            # Paste the image on the background
            background.paste(image, mask=image.split()[3])  # 3 is the alpha channel
            image = background 
        else:
            image = image.convert("RGB")
        
        return image

    # Transform function
    def preprocess_train(examples):
        images = [image_preprocess(image) for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    # In a function to avoid repeating code
    def apply_transform():
        # Resize the dataset if desired
        if config.training.max_steps is not None:
            dataset["train"] = dataset["train"].shuffle(seed=config.training.seed).select(range(config.training.max_steps))

        train_dataset = dataset["train"].with_transform(preprocess_train)
        return train_dataset
    
    if interface == "jax":
        train_dataset = apply_transform()
    elif interface == "torch":
        with accelerator.main_process_first():
            train_dataset = apply_transform()



    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]

        if interface == "jax":

            padded_tokens = tokenizer.pad(
                {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
            )
            batch = {
                "pixel_values": pixel_values,
                "input_ids": padded_tokens.input_ids,
            }
            batch = {k: v.numpy() for k, v in batch.items()}

        elif interface == "torch":
            batch = {"pixel_values": pixel_values, "input_ids": torch.stack(input_ids)}

        return batch

    
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=config.training.total_batch_size, drop_last=True
    )

    return train_dataset, train_dataloader
