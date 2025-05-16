from typing import Iterable, List, Tuple
from pathlib import Path
import torch
from tqdm import tqdm
import os
import sys

from bayesvlm.vlm import (
    CLIP,
    EncoderResult,
    ProbabilisticLogits,
    CLIPTextEncoder,
    CLIPImageEncoder,
    SiglipImageEncoder,
    SiglipTextEncoder,
)

def make_predictions(
    clip: CLIP,
    image_outputs: EncoderResult,
    text_outputs: EncoderResult,
    batch_size: int,
    device: str,
    save_predictions: bool = False,
    map_estimate: bool = False,
    cache_dir: Path = None,
) -> ProbabilisticLogits:
    if cache_dir is not None:
        mean_path = cache_dir / "logits_mean.pt"
        var_path = cache_dir / "logits_var.pt"

    if cache_dir is not None and mean_path.exists() and var_path.exists():
        return ProbabilisticLogits(
            mean=torch.load(mean_path, map_location='cpu'),
            var=torch.load(var_path, map_location='cpu'),
        )

    clip = clip.eval().to(device)
    text_outputs = text_outputs.to(device)

    loader = torch.utils.data.DataLoader(
        image_outputs,
        shuffle=False,
        batch_size=batch_size,
        num_workers=1,
    )

    means, vars = [], []
    for img_embeds, img_activations, img_residuals in tqdm(loader):
        img_outputs_batch = EncoderResult(
            embeds=img_embeds.to(device),
            activations=img_activations.to(device),
            residuals=img_residuals.to(device),
        )
        logits = clip(img_outputs_batch, text_outputs, map_estimate=map_estimate)
        means.append(logits.mean.detach().cpu())
        vars.append(logits.var.detach().cpu())
    means = torch.cat(means, dim=0)
    vars = torch.cat(vars, dim=0)

    if cache_dir is not None and save_predictions:
        torch.save(means, mean_path)
        torch.save(vars, var_path)

    return ProbabilisticLogits(mean=means, var=vars)


def precompute_image_features(
    image_encoder: CLIPImageEncoder | SiglipImageEncoder,
    loader: Iterable,
    save_predictions: bool = False,
    cache_dir: Path = None,
) -> Tuple[EncoderResult, torch.Tensor, torch.Tensor]:
    if save_predictions and cache_dir is None:
        raise ValueError("cache_dir must be provided if save_predictions is True")
    
    if cache_dir is not None:
        if not cache_dir.exists() and save_predictions:
            print(f"Creating cache directory {cache_dir}")
            cache_dir.mkdir(parents=True)

        embeds_path = cache_dir / "embeddings_img.pt"
        activations_path = cache_dir / "activations_img.pt"
        residuals_path = cache_dir / "residuals_img.pt"
        class_ids_path = cache_dir / "class_ids_img.pt"
        image_ids_path = cache_dir / "image_ids.pt"
        
        if embeds_path.exists() and class_ids_path.exists() and activations_path.exists() and image_ids_path.exists() and residuals_path.exists():
            result = EncoderResult(
                embeds=torch.load(embeds_path, map_location='cpu'),
                activations=torch.load(activations_path, map_location='cpu'),
                residuals=torch.load(residuals_path, map_location='cpu'),
            )
            class_ids = torch.load(class_ids_path, map_location='cpu')
            img_ids = torch.load(image_ids_path, map_location='cpu')
            return result, class_ids, img_ids
    
    img_embeds = []
    img_activations = []
    img_residuals = []
    img_ids = []
    labels = []
    for batch in tqdm(loader):
        result = image_encoder(batch, return_activations=True)
        img_embeds.append(result.embeds.detach().cpu())
        img_activations.append(result.activations.detach().cpu())
        img_residuals.append(result.residuals.detach().cpu())
        labels.append(batch["class_id"].cpu())
        img_ids.append(batch["image_id"].cpu())

    embeds = torch.cat(img_embeds, dim=0)
    activations = torch.cat(img_activations, dim=0)
    residuals = torch.cat(img_residuals, dim=0)
    class_ids = torch.cat(labels, dim=0)
    img_ids = torch.cat(img_ids, dim=0)

    if save_predictions:
        torch.save(embeds, embeds_path)
        torch.save(activations, activations_path)
        torch.save(residuals, residuals_path)
        torch.save(class_ids, class_ids_path)
        torch.save(img_ids, image_ids_path)

    return EncoderResult(embeds=embeds, activations=activations, residuals=residuals), class_ids, img_ids


def precompute_text_features(
    text_encoder: CLIPTextEncoder | SiglipTextEncoder,
    class_prompts: List[str],
    batch_size: int,
    save_predictions: bool = False,
    cache_dir: Path = None,
) -> EncoderResult:
    if cache_dir is None and save_predictions:
        raise ValueError("cache_dir must be provided if save_predictions is True")
    
    if cache_dir is not None:
        embeds_path = cache_dir / "embeddings_txt.pt"
        activations_path = cache_dir / "activations_txt.pt"
        if embeds_path.exists() and activations_path.exists():
            return EncoderResult(
                embeds=torch.load(embeds_path, map_location='cpu'), 
                activations=torch.load(activations_path, map_location='cpu'),
            )
    
    loader = torch.utils.data.DataLoader(
        class_prompts, 
        batch_size=batch_size, 
        num_workers=1, 
        shuffle=False,
        collate_fn=_label_collate_fn,
    )

    txt_embeds = []
    txt_activations = []
    for batch in tqdm(loader):
        result = text_encoder(batch, return_activations=True)
        txt_embeds.append(result.embeds.detach().cpu())
        txt_activations.append(result.activations.detach().cpu())
    
    embeds = torch.cat(txt_embeds, dim=0)
    activations = torch.cat(txt_activations, dim=0)

    if save_predictions and cache_dir is not None:
        torch.save(embeds, embeds_path)
        torch.save(activations, activations_path)

    return EncoderResult(embeds=embeds, activations=activations)

def _label_collate_fn(batch: List[str]):
    return dict(text=batch)

def compute_features(
    encoder: CLIPImageEncoder | CLIPTextEncoder,
    loader: str,
    tag: str = None,
    cache_dir: str = None,
    return_tensors: bool = False,
):
    if cache_dir is not None:
        path_activations = f"{cache_dir}/activations_{tag}.pt"
        path_embeddings = f"{cache_dir}/embeddings_{tag}.pt"

        if os.path.exists(path_activations) and os.path.exists(path_embeddings):
            return path_activations, path_embeddings

    activations = []
    embeddings = []

    for batch in tqdm(loader, file=sys.stdout):
        result = encoder(batch, return_activations=True)
        activations.append(result.activations.detach().cpu())
        embeddings.append(result.embeds.detach().cpu())
        
    activations = torch.cat(activations)
    embeddings = torch.cat(embeddings)

    if cache_dir is not None:
        torch.save(activations, path_activations)
        torch.save(embeddings, path_embeddings)

    if return_tensors:
        return activations, embeddings
    
    return path_activations, path_embeddings
