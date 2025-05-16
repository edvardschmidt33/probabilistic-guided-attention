import torch
from tqdm import tqdm
from typing import Optional, Literal

import copy
from bayesvlm.vlm import EncoderResult, ProbabilisticLogits, CLIP
from bayesvlm.hessians import (
    optimize_prior_precision,
    compute_covariances,
    compute_hessian_analytic_InfoNCE, 
)
from bayesvlm.knn import wdist2


def update_embeddings(projection: torch.nn.Module, outputs: EncoderResult, device: str = 'cuda'):
    outputs = outputs.to(device)

    ds = torch.utils.data.TensorDataset(outputs.activations.cpu(), outputs.residuals.cpu())
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=256,
        shuffle=False,
        num_workers=1,
    )

    all_embeds = []
    with torch.no_grad():
        for batch in loader:
            activations = batch[0]
            residuals = batch[1]
            embeds = projection(activations.to(device)) + residuals.to(device)
            all_embeds.append(embeds.cpu())
    
    all_embeds = torch.cat(all_embeds, dim=0)

    outputs = EncoderResult(
        embeds=all_embeds,
        activations=outputs.activations,
        residuals=outputs.residuals,
    )

    return outputs.to('cpu')

def select_epig_online(
    label_features: EncoderResult,
    pool_features: EncoderResult,
    target_features: EncoderResult,
    pool_class_ids: torch.Tensor,
    image_projection: torch.nn.Linear,
    clip: CLIP,
    A_img: torch.Tensor,
    A_txt: torch.Tensor,
    B_img: torch.Tensor,
    B_txt: torch.Tensor,
    cov_info: dict,
    budget: int,
    lr: float,
    hessian_update_scale: float,
    device: torch.device,
    num_samples: int,
    seed: int,
    pool_max_size: Optional[int] = None,
    target_max_size: Optional[int] = None,
    chunk_size: int = 4096,
    pool_subsampling: Literal['random', 'knn'] = 'random',
    k_nearest_neighbors: int = 1,
    proj_has_bias=False,
):   
    torch.manual_seed(seed)

    if pool_max_size is not None:
        pool_max_size = min(pool_max_size, len(pool_features.embeds))
    if target_max_size is not None:
        target_max_size = min(target_max_size, len(target_features.embeds))

    image_projection = copy.deepcopy(image_projection)
    image_projection = image_projection.to(device).train()

    pool_features = pool_features.to(device)
    target_features = target_features.to(device)
    label_features = label_features.to(device)
    pool_class_ids = pool_class_ids.to(device)

    A_img = A_img.to(device)
    B_img = B_img.to(device)
    A_txt = A_txt.to(device)
    B_txt = B_txt.to(device)

    clip = clip.to(device).eval()
    for param in clip.parameters():
        param.requires_grad = False
    
    cov_img, cov_txt= compute_covariances(A_img, B_img, A_txt, B_txt, cov_info)
    clip.set_covariances(cov_img, cov_txt)
    
    selected_indices = []
    epig_scores = []

    if target_max_size is not None and target_max_size < len(target_features.embeds):
        indices_samples_target = torch.randperm(len(target_features.embeds))[:target_max_size]
    else:
        indices_samples_target = torch.arange(len(target_features.embeds))

    if pool_subsampling == 'random':
        if pool_max_size is not None and pool_max_size < len(pool_features.embeds):
            indices_samples_pool = torch.randperm(len(pool_features.embeds))[:pool_max_size]
        else:
            indices_samples_pool = torch.arange(len(pool_features.embeds))
    elif pool_subsampling == 'knn_cosine':
        embeds_train = pool_features.embeds 
        train_activations = pool_features.activations
        embeds_test_subset = target_features.embeds[indices_samples_target] 
        test_activations = target_features.activations[indices_samples_target]
        
        source_B_factor = cov_img.B_inv.diagonal()

        if proj_has_bias:
            train_activations = torch.cat([train_activations, torch.ones_like(train_activations[:, :1])], dim=1)
            test_activations = torch.cat([test_activations, torch.ones_like(test_activations[:, :1])], dim=1)

        train_diag_cov = torch.einsum('ij,jk,ik->i', train_activations, cov_img.A_inv, train_activations)[:,None] * source_B_factor
        test_diag_cov = torch.einsum('ij,jk,ik->i', test_activations, cov_img.A_inv, test_activations)[:,None] * source_B_factor
            
        norm_train = embeds_train**2 + train_diag_cov
        expect_norm_train = norm_train.sum(dim=-1, keepdim=True)
        norm_test = embeds_test_subset**2 + test_diag_cov
        expect_norm_test = norm_test.sum(dim=-1, keepdim=True)

        # compute expected value
        embeds_train = embeds_train / torch.sqrt(expect_norm_train)
        embeds_test_subset  = embeds_test_subset / torch.sqrt(expect_norm_test)
        
        expected_similarity = embeds_test_subset @ embeds_train.t()
        
        nearest_neighbors = torch.argsort(expected_similarity, descending=True, dim=1)
        indices_samples_pool = nearest_neighbors[:, :k_nearest_neighbors].flatten().unique().cpu()
        
        if len(indices_samples_pool) < budget:
            raise ValueError(f"Could not find enough samples in the pool. Found {len(indices_samples_pool)}, expected at least {budget}.")
        
    elif pool_subsampling == 'knn_wasserstein':
        embeds_train = pool_features.embeds 
        train_activations = pool_features.activations
        embeds_test_subset = target_features.embeds[indices_samples_target] 
        test_activations = target_features.activations[indices_samples_target]

        if proj_has_bias:
            train_activations = torch.cat([train_activations, torch.ones_like(train_activations[:, :1])], dim=1)
            test_activations = torch.cat([test_activations, torch.ones_like(test_activations[:, :1])], dim=1)
        
        source_B_factor = cov_img.B_inv.diagonal()

        train_diag_cov = torch.einsum('ij,jk,ik->i', train_activations, cov_img.A_inv, train_activations)[:,None] * source_B_factor
        test_diag_cov = torch.einsum('ij,jk,ik->i', test_activations, cov_img.A_inv, test_activations)[:,None] * source_B_factor
        
        similarities = wdist2(embeds_test_subset, embeds_train, test_diag_cov, train_diag_cov) * -1
        nearest_neighbors = torch.argsort(similarities, descending=True, dim=1)
        indices_samples_pool = nearest_neighbors[:, :k_nearest_neighbors].flatten().unique().cpu()
        
        if len(indices_samples_pool) < budget:
            raise ValueError(f"Could not find enough samples in the pool. Found {len(indices_samples_pool)}, expected at least {budget}.")
        
    else:
        raise ValueError(f"Unknown subsampling method: {pool_subsampling}")

    for i in tqdm(range(budget), total=budget):
        # compute logits for pool and target
        
        
        pool_features_samples = EncoderResult(
            embeds=pool_features.embeds[indices_samples_pool],
            activations=pool_features.activations[indices_samples_pool],
        )
        pool_class_ids_samples = pool_class_ids[indices_samples_pool]

        target_features_samples = EncoderResult(
            embeds=target_features.embeds[indices_samples_target],
            activations=target_features.activations[indices_samples_target],
        )

        logits_pool = clip(pool_features_samples.to(device), label_features.to(device)).detach()
        logits_target = clip(target_features_samples.to(device), label_features.to(device)).detach()

        # compute EPIG
        epig = epig_from_logits_using_matmul(
            logits_pool, 
            logits_target, 
            num_samples=num_samples,
            chunk_size=chunk_size, 
            seed=seed + i,
        )

        # select the best sample that has not been selected yet
        for idx in torch.argsort(epig, descending=True):
            if indices_samples_pool[idx].item() in selected_indices:
                print(f"Skipping {idx} as it has already been selected.")
                continue
            best_sample_index = idx
            break

        best_activation = pool_features_samples.activations[best_sample_index].unsqueeze(0)
        best_residual = pool_features_samples.residuals[best_sample_index].unsqueeze(0)
        best_class_id = pool_class_ids_samples[best_sample_index].unsqueeze(0)

        selected_indices.append(indices_samples_pool[best_sample_index].item())
        epig_scores.append(epig[best_sample_index].item())

        # gradient step
        for param in image_projection.parameters():
            param.requires_grad = True

        image_projection.zero_grad()

        best_embed = image_projection(best_activation) + best_residual
        best_feature = EncoderResult(
            embeds=best_embed,
            activations=best_activation,
            residuals=best_residual,
        )

        best_logits = clip(best_feature, label_features)

        loss = torch.nn.functional.cross_entropy(
            input=best_logits.mean, 
            target=best_class_id,
        )
        loss.backward()

        with torch.no_grad():
            image_projection.weight.data -= lr * image_projection.weight.grad
            image_projection.weight.grad.zero_()

        # update the pool and target features
        pool_features = update_embeddings(image_projection, pool_features, device)
        target_features = update_embeddings(image_projection, target_features, device)

        best_pool_embed = pool_features_samples.embeds[best_sample_index]
        best_pool_activation = pool_features_samples.activations[best_sample_index]

        A_new = best_pool_activation @ best_pool_activation.T
        
        B_new = compute_hessian_analytic_InfoNCE(
            source_embeds=best_pool_embed.unsqueeze(0).to(device),
            target_embeds=label_features.embeds.to(device),
            logit_scale=clip.logit_scale.data.to(device),
        )

        # hard-coded number of samples for the initial hessian estimate
        # todo: fix this
        n = 327_680 + i
        scale0 = torch.sqrt(torch.tensor(n))
        scale1 = torch.sqrt(torch.tensor(n+1))

        A_img = (scale0 * A_img + A_new * hessian_update_scale) / scale1
        B_img = (scale0 * B_img + B_new * hessian_update_scale) / scale1

        lmda_img = optimize_prior_precision(
            projection=image_projection,
            A=A_img,
            B=B_img,
            lmbda_init=cov_info['lambda_img'],
            n=cov_info['n_img'],
            lr=1e-3,
            num_steps=20,
            device=device,
            retain_graph=True,
        )
        cov_info['lambda_img'] = lmda_img.item()

        cov_img, cov_txt = compute_covariances(A_img, B_img, A_txt, B_txt, cov_info)
        clip.set_covariances(cov_img, cov_txt)

    return selected_indices, epig_scores

def entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    H[p(y|x)] = - ∑_{y} p(y|x) log p(y|x)

    Using torch.distributions.Categorical().entropy() would be cleaner but more memory-intensive.

    If p(y_i|x) is 0, we make sure p(y_i|x) log p(y_i|x) evaluates to 0, not NaN.

    References:
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs: Tensor[float]

    Returns:
        Tensor[float]
    """
    return -torch.sum(torch.xlogy(probs, probs), dim=-1)

def marginal_entropy_from_probs(probs: torch.Tensor) -> torch.Tensor:
    """
    H[E_{p(θ)}[p(y|x,θ)]]

    Arguments:
        probs: Tensor[float], [N, K, Cl]

    Returns:
        Tensor[float], [N,]
    """
    assert probs.ndim == 3

    probs = torch.mean(probs, dim=1)  # [N, Cl]

    scores = entropy_from_probs(probs)  # [N,]
    # scores = check(scores, max_value=math.log(probs.shape[-1]), score_type="ME")  # [N,]

    return scores  # [N,]

@torch.no_grad()
def epig_from_logits_using_matmul(
    logits_pool: ProbabilisticLogits,
    logits_targ: ProbabilisticLogits,
    seed: int,
    num_samples: int,
    chunk_size: int = 4096,
) -> torch.Tensor:
    all_scores = []

    for i in tqdm(range(0, logits_pool.mean.shape[0], chunk_size), total=logits_pool.mean.shape[0] // chunk_size, desc="Computing EPIG scores"):
        probs_targ = logits_targ.sample_probas(num_samples, seed=seed + i).to(torch.float16)

        logits_mean_pool_chunk = logits_pool.mean[i:i + chunk_size]
        logits_var_pool_chunk = logits_pool.var[i:i + chunk_size]

        logits_pool_chunk = ProbabilisticLogits(
            mean=logits_mean_pool_chunk,
            var=logits_var_pool_chunk,
        )

        probs_pool_chunk = logits_pool_chunk.sample_probas(num_samples, seed=seed + i).to(torch.float16)
    
        epig_chunk = epig_from_probs_using_matmul(probs_pool_chunk, probs_targ, chunk_size=chunk_size).to(torch.float32)
        all_scores.append(epig_chunk)

    scores = torch.cat(all_scores, dim=0)
    return scores
    
@torch.no_grad()
def epig_from_probs_using_matmul(
    probs_pool: torch.Tensor, 
    probs_targ: torch.Tensor,
    chunk_size: int = 8192,
) -> torch.Tensor:
    # https://github.com/fbickfordsmith/epig/blob/b11124d2dd48381a5756e14d920d401f1fd3120d/src/uncertainty/epig_probs.py#L13

    """
    EPIG(x) = E_{p_*(x_*)}[I(y;y_*|x,x_*)]
            = H[p(y|x)] + E_{p_*(x_*)}[H[p(y_*|x_*)]] - E_{p_*(x_*)}[H[p(y,y_*|x,x_*)]]

    This uses the fact that I(A;B) = H(A) + H(B) - H(A,B).

    References:
        https://en.wikipedia.org/wiki/Mutual_information#Relation_to_conditional_and_joint_entropy
        https://github.com/baal-org/baal/pull/270#discussion_r1271487205

    Arguments:
        probs_pool: Tensor[float], [N_p, K, Cl]
        probs_targ: Tensor[float], [N_t, K, Cl]

    Returns:
        Tensor[float], [N_p,]
    """
    assert probs_pool.ndim == probs_targ.ndim == 3

    N_t, K, Cl = probs_targ.shape

    entropy_pool = marginal_entropy_from_probs(probs_pool)  # [N_p,]
    entropy_targ_mean = torch.mean(marginal_entropy_from_probs(probs_targ)) # [,]

    probs_pool = probs_pool.permute(0, 2, 1)  # [N_p, Cl, K]
    probs_targ = probs_targ.permute(1, 0, 2)  # [K, N_t, Cl]
    probs_targ = probs_targ.reshape(K, N_t * Cl)  # [K, N_t * Cl]

    # chunk matrix multiplication over last dimension
    _, N_t_Cl = probs_targ.shape

    entropy_joint_accum = torch.zeros(probs_pool.shape[0], device=probs_pool.device)  # [N_p,]

    for start in range(0, N_t_Cl, chunk_size):
        end = min(start + chunk_size, N_t_Cl)
        probs_targ_chunk = probs_targ[:, start:end]  # [K, chunk_size]

        probs_joint_chunk = probs_pool @ probs_targ_chunk  # [N_p, Cl, chunk_size]
        probs_joint_chunk = probs_joint_chunk / K

        xlogy_chunk = torch.xlogy(probs_joint_chunk, probs_joint_chunk)
        entropy_joint_chunk = -torch.sum(xlogy_chunk, dim=(-2, -1)) / N_t  # [N_p,]

        entropy_joint_accum += entropy_joint_chunk  # Accumulate the entropy contributions

    scores = entropy_pool + entropy_targ_mean - entropy_joint_accum  # [N_p,]

    return scores  # [N_p,]
