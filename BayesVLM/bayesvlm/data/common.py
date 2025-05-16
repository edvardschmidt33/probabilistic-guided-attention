import torch
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,
)

DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)

def default_collate_fn(batch: dict):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    image_ids = torch.tensor([item['image_id'] for item in batch]) if 'image_id' in batch[0] else None
    class_ids = torch.tensor([item['class_id'] for item in batch]) if 'class_id' in batch[0] else None

    # if images are tensors then stack them along the batch dimension
    # otherwise, return the list of images as is
    if len(images) > 0 and isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)

    d = dict(image=images, text=texts)

    if image_ids is not None:
        d['image_id'] = image_ids
    
    if class_ids is not None:
        d['class_id'] = class_ids
    
    return d

def _convert_image_to_rgb(image):
        return image.convert("RGB")

class AddGaussianNoise(object):
    def __init__(self, std: float, mean=0.0):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy_image = tensor + noise
        noisy_image = torch.clamp(noisy_image, 0.0, 1.0)
        return noisy_image
    
def revert_normalization(tensor: torch.Tensor):
    mean = torch.tensor(DEFAULT_MEAN).view(3, 1, 1)
    std = torch.tensor(DEFAULT_STD).view(3, 1, 1)

    if tensor.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        
    return tensor * std + mean

def revert_siglip_normalization(tensor: torch.Tensor):
    mean = torch.tensor(IMAGENET_STANDARD_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STANDARD_MEAN).view(3, 1, 1)

    if tensor.dim() == 4:
        mean = mean.view(1, 3, 1, 1)
        std = std.view(1, 3, 1, 1)
        
    return tensor * std + mean

def default_transform(image_size: int):
    "CLIP transform, but i'm afraid to change the name"
    return Compose([
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])

def corruption_transform(image_size: int, std: float):
    return Compose([
        Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        CenterCrop(image_size),
        _convert_image_to_rgb,
        ToTensor(),
        AddGaussianNoise(std=std),
        Normalize(DEFAULT_MEAN, DEFAULT_STD),
    ])

def siglip_transform(image_size: int):
    return Compose([
        Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize(IMAGENET_STANDARD_MEAN, IMAGENET_STANDARD_STD),
    ])
