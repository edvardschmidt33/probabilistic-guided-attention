from typing import Tuple, Literal
import argparse
import torch
from tqdm import tqdm
import sys
import os
import json
import math

from bayesvlm.vlm import CLIP, SIGLIP
from bayesvlm.hessians import (
    compute_hessian_analytic_InfoNCE, 
    compute_hessian_analytic_SigLIP, 
    optimize_prior_precision,
)
from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.utils import (
    get_model_type_and_size, 
    load_model, 
    get_transform,
    get_likelihood,
    get_image_size,
)
from bayesvlm.precompute import compute_features

@torch.no_grad()
def kfac_ggn(
    vlm: CLIP | SIGLIP, 
    num_classes: int,
    batch_size: int,
    source_embeds: torch.Tensor,
    source_activations: torch.Tensor,
    target_embeds: torch.Tensor,
    device: str,
    likelihood: Literal["info_nce", "siglip"],
    siglip_chunk_size_j: int = 8000,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the K-FAC approximation (only for the last layer) of the GGN for the cross-entropy loss function modeling "-log p(to|from)".

    Args:
        vlm (CLIP): Similarity function.
        from_encoder (Union[ImageEncoder, TextEncoder]): Source encoder.
        to_encoder (Union[ImageEncoder, TextEncoder]): Target encoder.
        loader (torch.utils.data.DataLoader): Data loader.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A and B matrices.
    """

    vlm = vlm.eval()
    vlm.logit_scale.requires_grad = False
    vlm.logit_bias.requires_grad = False

    num_class_batches = len(target_embeds) // num_classes
    if num_class_batches == 0:
        raise ValueError(f"To few datapoints for K-FAC approximation. Need at least {num_classes} datapoints.")
    
    print(f"Computing K-FAC approximation for {num_class_batches} batches of size {num_classes}...")

    A, B = 0, 0
    for i in tqdm(range(num_class_batches), total=num_class_batches, file=sys.stdout):
        print(f"Batch {i + 1}/{num_class_batches}...", flush=True)

        start = i * num_classes
        end = (i + 1) * num_classes
        target_embeds_class_batch = target_embeds[start:end].to(device)
        source_embeds_class_batch = source_embeds[start:end].to(device)
        source_activations_class_batch = source_activations[start:end].to(device)
        
        num_data_batches = len(source_embeds_class_batch) // batch_size

        for j in range(num_data_batches):
            print(f"\t- Data batch {j + 1}/{num_data_batches}...", end='\r', flush=True)
            batch_start = j * batch_size
            batch_end = (j + 1) * batch_size
            source_embeds_data_batch = source_embeds_class_batch[batch_start:batch_end]

            if likelihood == "info_nce":
                H_nll = compute_hessian_analytic_InfoNCE(
                    source_embeds_data_batch,
                    target_embeds_class_batch,
                    vlm.logit_scale.data,
                ).cpu()
            elif likelihood == "siglip":
                indices_data_batch = torch.arange(batch_start, batch_end, dtype=torch.long).to(source_embeds_data_batch.device)
                H_nll = compute_hessian_analytic_SigLIP(
                    x_batch=source_embeds_data_batch,
                    indices_batch=indices_data_batch,
                    y=target_embeds_class_batch,
                    logit_scale=vlm.logit_scale.data,
                    logit_bias=vlm.logit_bias.data,
                    chunk_size_j=siglip_chunk_size_j,
                ).cpu()
            else:
                raise ValueError(f"Invalid likelihood: {likelihood}, must be one of ['info_nce', 'siglip'].")
            B = B + H_nll

        if likelihood == "info_nce":
            A = A + source_activations_class_batch.T @ source_activations_class_batch
        elif likelihood == "siglip":
            # append the bias term to the activations
            source_activations_class_batch_bias = torch.cat([source_activations_class_batch, torch.ones_like(source_activations_class_batch[:, :1])], dim=1)
            A = A + source_activations_class_batch_bias.T @ source_activations_class_batch_bias

    n = num_class_batches * num_classes
    A = A / math.sqrt(n)
    B = B / math.sqrt(n)
    return A, B

                        
def main(
    device: str,
    dataset: str,
    model_str: str,
    precompute_batch_size: int,
    la_num_classes: int,
    la_batch_size: int,
    num_workers: int,
    hessian_dir: str,
    num_files: int = None,
    max_datapoints: int = None,
    siglip_chunk_size_j: int = 8000,

    # prior precision optimization
    lambda_init_txt: float = 400,
    lambda_init_img: float = 400,
    n_init_txt: float = 1.0,
    n_init_img: float = 1.0,
    lr: float = 1e-2,
    num_steps: int = 300,
):
    if not os.path.exists(hessian_dir):
        os.makedirs(hessian_dir)
    
    model_type, model_size = get_model_type_and_size(model_str)
    likelihood = get_likelihood(model_type)

    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)

    image_encoder, text_encoder, vlm = load_model(model_str, device)

    dm_factory = DataModuleFactory(
        batch_size=precompute_batch_size,
        num_workers=num_workers,
        shuffle_train=False,
        train_transform=transform,
        test_transform=transform,
    )
    dm = dm_factory.create(dataset)
    if num_files is not None and dataset == "laion400m":
        print(f'Reducing number of files from {len(dm.data_path)} to {num_files}')
        dm.data_path = dm.data_path[:num_files]
    dm.setup()
    loader = dm.test_dataloader()

    # precompute image embeddings and activations
    path_activations_img, path_embeddings_img = compute_features(
        encoder=image_encoder,
        loader=loader,
        tag="img",
        cache_dir=hessian_dir,
    )

    # precompute text embeddings and activations
    path_activations_txt, path_embeddings_txt = compute_features(
        encoder=text_encoder,
        loader=loader,
        tag="txt",
        cache_dir=hessian_dir,
    )

    # load embeddings and activations
    print("Loading embeddings and activations...")
    print(path_activations_img, path_embeddings_img, path_activations_txt, path_embeddings_txt)
    activations_img = torch.load(path_activations_img, map_location='cpu')
    embeddings_img = torch.load(path_embeddings_img, map_location='cpu')
    activations_txt = torch.load(path_activations_txt, map_location='cpu')
    embeddings_txt = torch.load(path_embeddings_txt, map_location='cpu')
    print("Done loading embeddings and activations.")

    if max_datapoints is not None:
        activations_img = activations_img[:max_datapoints]
        embeddings_img = embeddings_img[:max_datapoints]
        activations_txt = activations_txt[:max_datapoints]
        embeddings_txt = embeddings_txt[:max_datapoints]

    # compute K-FAC approximation for image encoder
    print("Computing Hessian for image encoder...")
    A_img_path = f"{hessian_dir}/A_img_analytic.pt"
    B_img_path = f"{hessian_dir}/B_img_analytic.pt"

    if os.path.exists(A_img_path) and os.path.exists(B_img_path):
        A_img = torch.load(A_img_path, map_location='cpu')
        B_img = torch.load(B_img_path, map_location='cpu')
    else:
        A_img, B_img = kfac_ggn(
            vlm=vlm,
            num_classes=la_num_classes,
            batch_size=la_batch_size,
            source_embeds=embeddings_img,
            source_activations=activations_img,
            target_embeds=embeddings_txt,
            device=device,
            likelihood=likelihood,
            siglip_chunk_size_j=siglip_chunk_size_j,
        )
        torch.save(A_img, f"{hessian_dir}/A_img_analytic.pt")
        torch.save(B_img, f"{hessian_dir}/B_img_analytic.pt")

    # compute K-FAC approximation for text encoder
    print("Computing Hessian for text encoder...")
    A_txt_path = f"{hessian_dir}/A_txt_analytic.pt"
    B_txt_path = f"{hessian_dir}/B_txt_analytic.pt"

    if os.path.exists(A_txt_path) and os.path.exists(B_txt_path):
        A_txt = torch.load(A_txt_path, map_location='cpu')
        B_txt = torch.load(B_txt_path, map_location='cpu')
    else:
        A_txt, B_txt = kfac_ggn(
            vlm=vlm,
            num_classes=la_num_classes,
            batch_size=la_batch_size,
            source_embeds=embeddings_txt,
            source_activations=activations_txt,
            target_embeds=embeddings_img,
            device=device,
            likelihood=likelihood,
            siglip_chunk_size_j=siglip_chunk_size_j,
        )
        torch.save(A_txt, f"{hessian_dir}/A_txt_analytic.pt")  
        torch.save(B_txt, f"{hessian_dir}/B_txt_analytic.pt")

    print("Optimizing prior precision for image encoder...")
    lambda_img = optimize_prior_precision(
        projection=image_encoder.vision_projection,
        A=A_img,
        B=B_img,
        lmbda_init=lambda_init_img,
        n=n_init_img,
        lr=lr,
        num_steps=num_steps,
        device=device,
    ).item()

    print("Optimizing prior precision for text encoder...")
    lambda_txt = optimize_prior_precision(
        projection=text_encoder.text_projection,
        A=A_txt,
        B=B_txt,
        lmbda_init=lambda_init_txt,
        n=n_init_txt,
        lr=lr,
        num_steps=num_steps,
        device=device,
    ).item()

    result = {
        "lambda_img": lambda_img,
        "n_img": n_init_img,
        "lambda_txt": lambda_txt,
        "n_txt": n_init_txt,
    }
    with open(f"{hessian_dir}/prior_precision_analytic.json", "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dataset", type=str, default="laion400m")
    parser.add_argument("--model", type=str, default="clip-base")
    parser.add_argument("--precompute_batch_size", type=int, default=10)
    parser.add_argument("--la_num_classes", type=int, default=32768)
    parser.add_argument("--la_batch_size", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--hessian_dir", type=str, default="hessians/custom-hessian-clip-base")
    parser.add_argument("--num_files", type=int, default=60)
    parser.add_argument("--max_datapoints", type=int, default=327680)
    parser.add_argument("--siglip_chunk_size", type=int, default=8000)

    # prior precision optimization
    parser.add_argument("--lambda_init_txt", type=float, default=400)
    parser.add_argument("--lambda_init_img", type=float, default=800)
    parser.add_argument("--n_init_txt", type=float, default=1.0)
    parser.add_argument("--n_init_img", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--num_steps", type=int, default=300)
    
    args = parser.parse_args()

    main(
        device=args.device,
        dataset=args.dataset,
        model_str=args.model,
        precompute_batch_size=args.precompute_batch_size,
        la_num_classes=args.la_num_classes,
        la_batch_size=args.la_batch_size,
        num_workers=args.num_workers,
        hessian_dir=args.hessian_dir,
        num_files=args.num_files,
        max_datapoints=args.max_datapoints,
        siglip_chunk_size_j=args.siglip_chunk_size,

        # prior precision optimization
        lambda_init_txt=args.lambda_init_txt,
        lambda_init_img=args.lambda_init_img,
        n_init_txt=args.n_init_txt,
        n_init_img=args.n_init_img,
        lr=args.lr,
        num_steps=args.num_steps,
    )
