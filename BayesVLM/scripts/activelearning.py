import argparse
import copy
import json
from typing import Literal
from collections import OrderedDict
from pathlib import Path

import torch
import torch.utils.data
import wandb
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_calibration_error,
)
from tqdm import tqdm

from bayesvlm.vlm import CLIP, EncoderResult

from bayesvlm.data.factory import DataModuleFactory
from bayesvlm.precompute import precompute_image_features, precompute_text_features, make_predictions

from bayesvlm.epig import select_epig_online
from bayesvlm.knn import (
    extract_test_train_indices,
    find_similar_samples_cosine,
    find_similar_samples_wasserstein,
)
from bayesvlm.selection import select_random, select_topk
from bayesvlm.hessians import compute_covariances, load_hessians, optimize_prior_precision
from bayesvlm.utils import get_model_type_and_size, get_image_size, get_transform, load_model


def evaluate(
    projection: torch.nn.Module,
    text_outputs: EncoderResult,
    clip: CLIP,
    loader: torch.utils.data.DataLoader,
    num_classes: int,
    device: str,
):
    clip = clip.eval().to(device)
    projection = projection.eval().to(device)
    text_outputs = text_outputs.to(device)

    all_logits_mean = []
    all_logits_var = []
    all_labels = []
    loss = 0.0
    with torch.no_grad():
        for activations, residuals, lbls in loader:
            image_embeds = projection(activations.to(device)) + residuals.to(device)
            image_outputs = EncoderResult(embeds=image_embeds, activations=activations.to(device), residuals=residuals.to(device))

            logits = clip(image_outputs, text_outputs)
            all_logits_mean.append(logits.mean.cpu())
            all_logits_var.append(logits.var.cpu())
            all_labels.append(lbls.cpu())
            loss += torch.nn.functional.cross_entropy(logits.mean, lbls.to(device), reduction='sum').item()
    
    all_logits_mean = torch.cat(all_logits_mean, dim=0)
    all_logits_var = torch.cat(all_logits_var, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    acc = (all_logits_mean.argmax(dim=1) == all_labels).float().mean().item()
    acc_weighted = multiclass_accuracy(all_logits_mean, all_labels, num_classes=num_classes, average='weighted')
    ece = multiclass_calibration_error(all_logits_mean, all_labels, num_classes=num_classes)
    
    return dict(
        accuracy=acc,
        accuracy_weighted=acc_weighted,
        ece=ece,
        loss=loss / len(loader.dataset),
    )


def finetune(
    img_projection: torch.nn.Module,
    txt_projection: torch.nn.Module,
    clip: CLIP,
    image_features_train: EncoderResult,
    labels_train: torch.Tensor,
    image_features_val: EncoderResult,
    labels_val: torch.Tensor,
    image_features_test: EncoderResult,
    labels_test: torch.Tensor,
    text_features: EncoderResult,
    lr: float,
    wd: float,
    epochs: int,
    batch_size: int,
    device: str,
    finetune_dir: Path,
    selection: str,
    num_classes: int,
    k_nearest: int,
    subset_size: int,
    dataset: str,
    hessian_scale: float,
    project_name: str,
    epig_lr: float,
    epig_hessian_update_scale: float,
    epig_mc_samples: int = 100,
    knn_method: str = 'wasserstein',
):
    wandb.init(project=project_name, dir=str(finetune_dir), reinit=True)
    wandb.config.update({
        'lr': lr,
        'wd': wd,
        'epochs': epochs,
        'batch_size': batch_size,
        'selection': selection,
        'subset_size': subset_size,
        'k_nearest': k_nearest,
        'dataset': dataset,
        'hessian_scale': hessian_scale,

        'epig_lr': epig_lr,
        'epig_hessian_update_scale': epig_hessian_update_scale,
        'epig_mc_samples': epig_mc_samples,
        'knn_method': knn_method,
    })

    wandb.run.name = finetune_dir.parent.name + '/' + finetune_dir.name

    clip = clip.eval().to(device)
    clip.logit_scale.data.requires_grad = False
    clip.logit_bias.data.requires_grad = False

    # freeze projection layers for finetuning
    txt_projection = txt_projection.eval().to(device)
    for param in txt_projection.parameters():
        param.requires_grad = False

    # unfreeze projection layers for finetuning
    img_projection = img_projection.train().to(device)
    for param in img_projection.parameters():
        param.requires_grad = True

    text_features = text_features.to(device)

    train_ds = torch.utils.data.TensorDataset(image_features_train.activations.cpu(), image_features_train.residuals.cpu(), labels_train.cpu())
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
    )

    val_ds = torch.utils.data.TensorDataset(image_features_val.activations.cpu(), image_features_val.residuals.cpu(), labels_val.cpu())
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )
    
    test_ds = torch.utils.data.TensorDataset(image_features_test.activations.cpu(), image_features_test.residuals.cpu(), labels_test.cpu())
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
    )

    params = list(img_projection.parameters())
    # only params that require grad
    params = [p for p in params if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params,
        lr=lr,
        weight_decay=wd,
    )

    train_metrics = evaluate(img_projection, text_features, clip, train_loader, num_classes=num_classes, device=device)
    val_metrics = evaluate(img_projection, text_features, clip, val_loader, num_classes=num_classes, device=device)
    test_metrics = evaluate(img_projection, text_features, clip, test_loader, num_classes=num_classes, device=device)

    wandb.log({f'train_{k}': v for k, v in train_metrics.items()}, step=0)
    wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, step=0)
    wandb.log({f'test_{k}': v for k, v in test_metrics.items()}, step=0)

    best_val_loss = float('inf')
    best_test_metrics = None
    best_val_metrics = None
    best_projection = None

    for epoch in range(epochs):
        losses = []
        for activations, residuals, lbls in tqdm(train_loader):
            optimizer.zero_grad()

            image_embeds = img_projection(activations.to(device)) + residuals.to(device)
            text_embeds = txt_projection(text_features.activations.to(device))

            logits = clip(image_embeds, text_embeds)
            
            loss = torch.nn.functional.cross_entropy(logits, lbls.to(device))

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"Epoch {epoch + 1}/{epochs}, loss: {sum(losses) / len(losses)}")

        train_metrics = evaluate(img_projection, text_features, clip, train_loader, num_classes=num_classes, device=device)
        val_metrics = evaluate(img_projection, text_features, clip, val_loader, num_classes=num_classes, device=device)
        test_metrics = evaluate(img_projection, text_features, clip, test_loader, num_classes=num_classes, device=device)

        if val_metrics['loss'] <= best_val_loss:
            best_val_loss = val_metrics['loss']
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics
            best_projection = copy.deepcopy(img_projection)
        
        if best_test_metrics is not None:
            wandb.log({f'best_test_{k}': v for k, v in best_test_metrics.items()}, step=epoch + 1)
            wandb.log({f'best_val_{k}': v for k, v in best_val_metrics.items()}, step=epoch + 1)

        wandb.log({f'train_{k}': v for k, v in train_metrics.items()}, step=epoch + 1)
        wandb.log({f'val_{k}': v for k, v in val_metrics.items()}, step=epoch + 1)
        wandb.log({f'test_{k}': v for k, v in test_metrics.items()}, step=epoch + 1)
    
    return best_projection


def run_knn(
    embeds_train: EncoderResult, 
    embeds_test: EncoderResult,
    indices_test: torch.Tensor,
    values_test: torch.Tensor,
    k_nearest: int,
    device: str,
    source_covariance,
    method: str,
    proj_has_bias=False,
    ):
    if proj_has_bias:
        embeds_train = embeds_train.clone()
        embeds_test = embeds_test.clone()
        embeds_train.activations = torch.cat([embeds_train.activations, torch.ones_like(embeds_train.activations[:, :1])], dim=1)
        embeds_test.activations = torch.cat([embeds_test.activations, torch.ones_like(embeds_test.activations[:, :1])], dim=1)

    if method == 'cosine':
        return find_similar_samples_cosine(embeds_train, embeds_test, indices_test, values_test, k_nearest, source_covariance, device)
    elif method == 'wasserstein':
        return find_similar_samples_wasserstein(embeds_train, embeds_test, indices_test, values_test, k_nearest, source_covariance, device)
    else:
        raise ValueError(f"Unknown method {method}")


def main(
    model_str: str,
    dataset: str,
    hessian_dir: str,

    experiment_dir: str,
    project_name: str, 

    # experiment parameters
    hessian_scale: float,
    subset_size: int,

    # precompute parameters
    predictions_batch_size: int = 256,
    precompute_batch_size: int = 256,
    precompute_num_workers: int = 8,

    # fine-tuning parameters
    finetune_lr: float = 1e-5,
    finetune_wd: float = 5e-2,
    finetune_epochs: int = 100,
    finetune_batch_size: int = 30,

    # selection strategies to run
    only_deterministic_strategies: bool = False,
    only_random_strategies: bool = False,
    only_epig: bool = False,
    without_epig: bool = False,

    # epig parameters
    epig_lr: float = 1e-4,
    epig_hessian_update_scale: float = 10.0,
    epig_num_samples: int = 100,

    # knn parameters
    k_nearest: int = 1,
    knn_method: Literal['cosine', 'wasserstein'] = 'wasserstein',

    device: str = 'cuda',
):
    run_dir = Path(experiment_dir) / dataset
    if not run_dir.exists():
        print(f"Creating run directory {run_dir}")
        run_dir.mkdir(parents=True)
    
    model_type, model_size = get_model_type_and_size(model_str)
    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)

    factory = DataModuleFactory(
        batch_size=precompute_batch_size,
        num_workers=precompute_num_workers,
        shuffle_train=False,
        train_transform=transform,
        test_transform=transform,
    )
    dm = factory.create(dataset)
    dm.setup()

    # load / compute features
    image_encoder, text_encoder, clip = load_model(model_str, device=device)
    image_encoder.freeze_all_layers()
    text_encoder.freeze_all_layers()
    clip.logit_scale.data.requires_grad = False
    clip.logit_bias.data.requires_grad = False
    
    print("[1] Precomputing features ...")
    image_outputs_train, image_class_ids_train, image_ids_train = precompute_image_features(
        image_encoder=image_encoder,
        loader=dm.train_dataloader(),
        cache_dir=run_dir / 'base' / 'train',
        save_predictions=True,
    )

    image_outputs_val, image_class_ids_val, image_ids_val = precompute_image_features(
        image_encoder=image_encoder,
        loader=dm.val_dataloader(),
        cache_dir=run_dir / 'base' / 'val',
        save_predictions=True,
    )

    image_outputs_test, image_class_ids_test, image_ids_test = precompute_image_features(
        image_encoder=image_encoder,
        loader=dm.test_dataloader(),
        cache_dir=run_dir / 'base' / 'test',
        save_predictions=True,
    )

    label_outputs = precompute_text_features(
        text_encoder=text_encoder,
        class_prompts=dm.class_prompts,
        batch_size=precompute_batch_size,
        cache_dir=run_dir / 'base',
        save_predictions=True,
    )

    A_img, B_img = load_hessians(la_dir=hessian_dir, tag='img', return_info=False)
    A_txt, B_txt, info = load_hessians(la_dir=hessian_dir, tag='txt', return_info=True)
    
    lambda_img = optimize_prior_precision(
        projection=image_encoder.vision_projection,
        A=A_img,
        B=B_img,
        lmbda_init=info['lambda_img'],
        n=hessian_scale,
        lr=1e-2,
        num_steps=500,
        device=device,
        retain_graph=True,
    ).item()

    lambda_txt = optimize_prior_precision(
        projection=text_encoder.text_projection,
        A=A_txt,
        B=B_txt,
        lmbda_init=info['lambda_txt'],
        n=hessian_scale,
        lr=1e-2,
        num_steps=500,
        device=device,
        retain_graph=True,
    ).item()

    covar_info = dict(
        lambda_img=lambda_img,
        lambda_txt=lambda_txt,
        n_img=hessian_scale,
        n_txt=hessian_scale,
    )

    cov_img, cov_txt = compute_covariances(A_img, B_img, A_txt, B_txt, covar_info)

    clip.set_covariances(
        source_covariance=cov_img, 
        target_covariance=cov_txt,
    )

    print("[2] Making predictions ...")
    prob_logits_train = make_predictions(
        clip=clip,
        image_outputs=image_outputs_train,
        text_outputs=label_outputs,
        batch_size=predictions_batch_size,
        device=device,
        save_predictions=False,
        map_estimate=False,
    )

    prob_logits_train_map = make_predictions(
        clip=clip,
        image_outputs=image_outputs_train,
        text_outputs=label_outputs,
        batch_size=predictions_batch_size,
        device=device,
        save_predictions=False,
        map_estimate=True,
    )

    prob_logits_val = make_predictions(
        clip=clip,
        image_outputs=image_outputs_val,
        text_outputs=label_outputs,
        batch_size=predictions_batch_size,
        device=device,
        save_predictions=False,
        map_estimate=False,
    )

    prob_logits_test = make_predictions(
        clip=clip,
        image_outputs=image_outputs_test,
        text_outputs=label_outputs,
        batch_size=predictions_batch_size,
        device=device,
        save_predictions=False,
        map_estimate=False,
    )

    prob_logits_test_map = make_predictions(
        clip=clip,
        image_outputs=image_outputs_test,
        text_outputs=label_outputs,
        batch_size=predictions_batch_size,
        device=device,
        save_predictions=False,
        map_estimate=True,
    )

    path = f'subset_{subset_size}_k_{k_nearest}_n_{hessian_scale}_epig_lr_{epig_lr}_epig_update_{epig_hessian_update_scale}_knn_{knn_method}'
    subset_dir = run_dir / path

    if not subset_dir.exists():
        subset_dir.mkdir(parents=True)

    print("[3] Creating training subsets ...")
    #todo: move in separate function
    json_path = subset_dir / 'subset_indices_train.json'
    if json_path.exists():
        with open(json_path) as f:
            subset_indices_train = json.load(f)
    else:
        subset_indices_train = OrderedDict()

    
    if not only_random_strategies and not only_epig:
        print("    - Aleatoric entropy ...", flush=True)
        if 'entropy_map' not in subset_indices_train:
            indices_entropy_alea_test_map, values_entropy_alea_test_map = select_topk(
                prob_logits_test_map, 
                k=subset_size, 
                variant='entropy', 
                entropy_variant='map_alea',
                return_values=True,
            )
            indices_entropy_alea_support = run_knn(
                embeds_train=image_outputs_train,
                embeds_test=image_outputs_test,
                indices_test = indices_entropy_alea_test_map,
                values_test = values_entropy_alea_test_map,
                k_nearest=k_nearest,
                source_covariance=clip.source_covariance,
                device=device,
                method=knn_method,
                proj_has_bias=clip.source_projection_has_bias,
            )
            subset_indices_train['entropy_map'] = indices_entropy_alea_support
        
        print("    - Aleatoric entropy on train ...", flush=True)
        if 'entropy_map_test' not in subset_indices_train:
            indices_entropy_alea_train_map, values_entropy_alea_train_map = select_topk(
                prob_logits_train_map, 
                k=subset_size, 
                variant='entropy', 
                entropy_variant='map_alea',
                return_values=True,
            )
            subset_indices_train['entropy_map_train'] = {
                0: dict(
                    score=0.0,
                    indices=indices_entropy_alea_train_map.tolist(),
                    similarities=values_entropy_alea_train_map.tolist(),
                )
            }

        # BALD on test 
        print(f"    - BALD (on test) ...", flush=True)
        if f'bald_test' not in subset_indices_train:
            indices_entropy_exp_mutual_info_test, values_entropy_exp_mutual_info_test = select_topk(
                prob_logits_test,
                k=subset_size,
                variant='exp_mutual_info',
                return_values=True,
                seed=0,
            )
            indices_entropy_exp_mutual_info_support = run_knn(
                embeds_train=image_outputs_train,
                embeds_test=image_outputs_test,
                indices_test = indices_entropy_exp_mutual_info_test,
                values_test = values_entropy_exp_mutual_info_test,
                k_nearest=k_nearest,
                source_covariance=clip.source_covariance,
                device=device,
                method=knn_method,
                proj_has_bias=clip.source_projection_has_bias,
            )
            subset_indices_train[f'bald_test'] = indices_entropy_exp_mutual_info_support


    if not only_random_strategies and not without_epig:
        # EPIG with KNN subsampling       
        print(f"    - EPIG KNN...", flush=True)
        if f'epig_knn' not in subset_indices_train:
            pooling_subsampling = 'knn_cosine' if knn_method == 'cosine' else 'knn_wasserstein'
            
            indices_epig, epig_scores = select_epig_online(
                label_features=label_outputs,
                pool_features=image_outputs_train,
                target_features=image_outputs_test,
                pool_class_ids=image_class_ids_train,
                image_projection=image_encoder.vision_projection,
                clip=clip,
                A_img=A_img,
                B_img=B_img,
                A_txt=A_txt,
                B_txt=B_txt,
                cov_info=covar_info,
                budget=subset_size,
                lr=epig_lr,
                hessian_update_scale=epig_hessian_update_scale,
                device=device,
                num_samples=epig_num_samples,
                seed=0,
                pool_max_size=40_000,
                target_max_size=20_000,
                pool_subsampling = pooling_subsampling,
                proj_has_bias=clip.source_projection_has_bias,
            )

            subset_indices_train[f'epig_knn'] = {
                0: dict(
                    score=0.0,
                    indices=indices_epig,
                    similarities=epig_scores,
                )
            }
    

    if not only_deterministic_strategies and not only_epig:
        for i in range(5):
            if f'random_on_test_{i}' not in subset_indices_train:
                indices_random_test = select_random(
                    prob_logits_test, 
                    k=subset_size, 
                    seed=i,
                )
                indices_random_support = run_knn(
                    embeds_train=image_outputs_train,
                    embeds_test=image_outputs_test,
                    indices_test=indices_random_test,
                    values_test=torch.ones_like(indices_random_test),
                    k_nearest=k_nearest,
                    source_covariance=clip.source_covariance,
                    device=device, 
                    method=knn_method,
                    proj_has_bias=clip.source_projection_has_bias,
                )
                subset_indices_train[f'random_on_test_{i}'] = indices_random_support

        for i in range(5):
            if f'random_on_train_{i}' not in subset_indices_train:
                indices_random_trivial_support = select_random(
                    prob_logits_train, 
                    k=k_nearest * subset_size, 
                    seed=i,
                )
                # hacky way to fit into the existing structure
                subset_indices_train[f'random_on_train_{i}'] = {
                    0: dict(
                        score=0.0,
                        indices=indices_random_trivial_support.tolist(), 
                        similarities=[1.0] * len(indices_random_trivial_support)
                    )
                }

    # save dictionary
    with open(subset_dir / 'subset_indices_train.json', 'w') as f:
        json.dump(subset_indices_train, f)

    print("[4] Fine-tuning based on training subsets ...")
    for subset, indices_dict in subset_indices_train.items():
        print(f"    - Fine-tuning on subset {subset} ...")
        indices = extract_test_train_indices(indices_dict)['train']
        masked_image_features = image_outputs_train[indices]
        masked_class_ids = image_class_ids_train[indices]
        masked_image_ids = image_ids_train[indices]

        finetune_dir = subset_dir / subset
        checkpoint_path = finetune_dir / 'img_projection.pt'

        finetune_dir.mkdir(parents=True, exist_ok=True)

        img_projection = copy.deepcopy(image_encoder.vision_projection)
        txt_projection = copy.deepcopy(text_encoder.text_projection)

        def _selection_from_key(key: str):
            elements = key.split('_')

            # if last element is a number, it is a seed
            if elements[-1].isdigit():
                return '_'.join(elements[:-1])
            
            return key
        

        img_projection = finetune(
            img_projection=img_projection,
            txt_projection=txt_projection,
            clip=clip,
            image_features_train=masked_image_features,
            labels_train=masked_class_ids,
            image_features_val=image_outputs_val,
            labels_val=image_class_ids_val,
            image_features_test=image_outputs_test,
            labels_test=image_class_ids_test,
            text_features=label_outputs,
            lr=finetune_lr,
            wd=finetune_wd,
            epochs=finetune_epochs,
            batch_size=finetune_batch_size,
            device=device,
            finetune_dir=finetune_dir,
            selection=_selection_from_key(subset),
            num_classes=len(dm.class_prompts),
            k_nearest=k_nearest,
            subset_size=subset_size,
            project_name=project_name,
            dataset=dataset,
            hessian_scale=hessian_scale,
            epig_lr=epig_lr,
            epig_hessian_update_scale=epig_hessian_update_scale,
            epig_mc_samples=epig_num_samples,
            knn_method=knn_method,
        )

        torch.save(
            img_projection.state_dict(),
            checkpoint_path,
        )

if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='clip-base')
    parser.add_argument('--dataset', type=str, default='homeoffice-da-clipart')
    parser.add_argument('--hessian_dir', type=str, default='hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K')

    parser.add_argument('--experiment_dir', type=str, default='experiments/active-finetuning')
    parser.add_argument('--project_name', type=str, default='active-finetuning')
    
    # experiment parameters
    parser.add_argument('--subset_size', type=int, default=50)
    parser.add_argument('--hessian_scale', type=float, default=10)

    # precompute parameters
    parser.add_argument('--predictions_batch_size', type=int, default=256)
    parser.add_argument('--precompute_batch_size', type=int, default=256)
    parser.add_argument('--precompute_num_workers', type=int, default=8)

    # fine-tuning parameters
    parser.add_argument('--finetune_lr', type=float, default=1e-5)
    parser.add_argument('--finetune_wd', type=float, default=5e-2)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--finetune_batch_size', type=int, default=30)

    # which selection strategies to run
    parser.add_argument('--only_deterministic_strategies', action='store_true', default=False)
    parser.add_argument('--only_random_strategies', action='store_true', default=False)
    parser.add_argument('--without_epig', action='store_true', default=False)
    parser.add_argument('--only_epig', action='store_true', default=False)

    # epig parameters
    parser.add_argument('--epig_lr', type=float, default=1e-4)
    parser.add_argument('--epig_hessian_update_scale', type=float, default=10.0)

    # knn parameters
    parser.add_argument('--k_nearest', type=int, default=1)
    parser.add_argument('--knn_method', type=str, default='wasserstein')

    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    main(
        model_str=args.model,
        dataset=args.dataset,
        hessian_dir=args.hessian_dir,

        experiment_dir=args.experiment_dir,
        project_name=args.project_name,

        # experiment parameters
        hessian_scale=args.hessian_scale,
        subset_size=args.subset_size,

        # precompute parameters
        predictions_batch_size=args.predictions_batch_size,
        precompute_batch_size=args.precompute_batch_size,
        precompute_num_workers=args.precompute_num_workers,

        # finetuning parameters
        finetune_lr=args.finetune_lr,
        finetune_wd=args.finetune_wd,
        finetune_epochs=args.finetune_epochs,   
        finetune_batch_size=args.finetune_batch_size,

        # selection strategies to run
        only_deterministic_strategies=args.only_deterministic_strategies,
        only_random_strategies=args.only_random_strategies,
        without_epig=args.without_epig,
        only_epig=args.only_epig,

        # epig parameters
        epig_lr=args.epig_lr,
        epig_hessian_update_scale=args.epig_hessian_update_scale,

        # knn parameters
        k_nearest=args.k_nearest,
        knn_method=args.knn_method,

        device=args.device,
    )
