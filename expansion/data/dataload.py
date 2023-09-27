
from datasets import load_dataset

import os

import numpy as np

import random

from torchvision import transforms

import torch


from transformers import CLIPTokenizer

dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

def get_dataset(config):


# LOAD DATASET
    if config.dataset_name is not None:
    # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config_name,
            cache_dir=config.cache_dir,
        )
    else:
        data_files = {}
        if config.train_data_dir is not None:
            data_files["train"] = os.path.join(config.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=config.cache_dir,
        )
    
    column_names = dataset["train"].column_names


# MAKE SURE THE KEY FOR ACCESING IMAGES AND LABELS IS CORRECT

    dataset_columns = dataset_name_mapping.get(config.dataset_name, None)

    if config.image_column is None:
        image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        image_column = config.image_column
        if image_column not in column_names:
            raise ValueError(
                f"--image_column' value '{config.image_column}' needs to be one of: {', '.join(column_names)}"
            )
    
    if config.caption_column is None:
        caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        caption_column = config.caption_column
        if caption_column not in column_names:
            raise ValueError(
                f"--caption_column' value '{config.caption_column}' needs to be one of: {', '.join(column_names)}"
            )

# LOAD TOKENIZER
    tokenizer = CLIPTokenizer.from_pretrained(
        config.pretrained_model_name_or_path, revision=config.revision, subfolder="tokenizer"
    )

# PREPROCESS THE DATASET

    train_transforms = transforms.Compose(
        [
            transforms.Resize(config.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(config.resolution) if config.center_crop else transforms.RandomCrop(config.resolution),
            transforms.RandomHorizontalFlip() if config.random_flip else transforms.Lambda(lambda x: x),
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
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
        input_ids = inputs.input_ids
        return input_ids
    
    # Transform function
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)

        return examples

    # Resize the dataset if desired
    if config.max_train_samples is not None:
        dataset["train"] = dataset["train"].shuffle(seed=config.seed).select(range(config.max_train_samples))

    train_dataset = dataset["train"].with_transform(preprocess_train)


    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]

        padded_tokens = tokenizer.pad(
            {"input_ids": input_ids}, padding="max_length", max_length=tokenizer.model_max_length, return_tensors="pt"
        )
        batch = {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
        }
        batch = {k: v.numpy() for k, v in batch.items()}

        return batch

