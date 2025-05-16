from typing import Callable, Tuple, Literal
from bayesvlm.data.common import default_transform, siglip_transform
from bayesvlm.vlm import CLIPImageEncoder, CLIPTextEncoder, CLIP, SiglipImageEncoder, SiglipTextEncoder, SIGLIP
from bayesvlm.constants import MODEL_NAME_MAP

def get_model_type_and_size(model_str: str) -> Tuple[str, str]:
    name, size = model_str.split("-")
    return name, size

def get_image_size(model_str) -> int:
    _, _, transform_size = MODEL_NAME_MAP[model_str]
    return transform_size

def get_model_url(model_str: str) -> str:
    provider, model_id, _ = MODEL_NAME_MAP[model_str]
    return f"{provider}/{model_id}"

def get_transform(model_type: Literal['clip', 'siglip'], image_size: int) -> Callable:
    if model_type == "siglip":
        return siglip_transform(image_size)
    return default_transform(image_size)

def get_likelihood(model_type: Literal['clip', 'siglip']) -> str:
    if model_type == 'clip':
        return 'info_nce'
    return 'siglip'

def load_model(
    model_str: str, 
    device: str,
) -> Tuple[CLIPImageEncoder, CLIPTextEncoder, CLIP] | Tuple[SiglipImageEncoder, SiglipTextEncoder, SIGLIP]:
    model_type, _ = get_model_type_and_size(model_str)
    model_url = get_model_url(model_str)

    if model_type == "siglip":
        image_encoder = SiglipImageEncoder.from_huggingface(model_url, device=device).eval().to(device)
        text_encoder = SiglipTextEncoder.from_huggingface(model_url, device=device).eval().to(device)
        vlm = SIGLIP.from_huggingface(model_url, device=device).eval().to(device)
    elif model_type == "clip":
        image_encoder = CLIPImageEncoder.from_huggingface(model_url, device=device).eval().to(device)
        text_encoder = CLIPTextEncoder.from_huggingface(model_url, device=device).eval().to(device)
        vlm = CLIP.from_huggingface(model_url, device=device).eval().to(device)
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    
    return image_encoder, text_encoder, vlm
