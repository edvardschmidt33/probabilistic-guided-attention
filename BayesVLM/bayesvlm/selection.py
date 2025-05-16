import torch
from typing import Literal

from bayesvlm.vlm import ProbabilisticLogits


def _entropy(
        logits_mean: torch.Tensor, 
        logits_var: torch.Tensor, 
        variant: Literal['map_alea', 'exp_alea', 'comb', 'comb_covar'],
        num_samples: int = 1000,
        seed: int = None
    ) -> torch.Tensor:
    prob_logits = ProbabilisticLogits(
        mean=logits_mean,
        var=logits_var,
    )
    if variant == 'exp_alea':
        return prob_logits.expected_aleatoric_entropy(num_samples=num_samples)
    if variant == 'map_alea':
        probas = torch.nn.functional.softmax(prob_logits.mean, dim=1)
    elif variant == 'comb':
        probas = prob_logits.softmax(num_samples=0, seed=seed)
    elif variant == 'comb_covar':
        probas = prob_logits.softmax(num_samples=num_samples, seed=seed)
    return -(probas * probas.log()).sum(dim=1)

def complexity_score(
    prob_logits: ProbabilisticLogits,
    variant: Literal['var', 'logdet', 'entropy', 'map_mutual_info', 'exp_mutual_info'],
    entropy_variant: Literal['map_alea', 'exp_alea', 'comb', 'comb_covar'] = None,
    seed: int = None,
) -> torch.Tensor:
    if variant == 'var':
        return prob_logits.var.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    if variant == 'logdet':
        return prob_logits.var.logdet()
    elif variant == 'entropy':
        entropy = _entropy(prob_logits.mean, prob_logits.var, entropy_variant, seed=seed)
        return entropy
    elif variant == 'exp_mutual_info':
        entropy_total = _entropy(prob_logits.mean, prob_logits.var, 'comb_covar', seed=seed)
        entropy_alea = _entropy(prob_logits.mean, prob_logits.var, 'exp_alea', seed=seed)
        entropy_epi = entropy_total - entropy_alea
        return entropy_epi
    elif variant == 'map_mutual_info':
        entropy_total = _entropy(prob_logits.mean, prob_logits.var, 'comb_covar', seed=seed)
        entropy_alea = _entropy(prob_logits.mean, prob_logits.var, 'map_alea', seed=seed)
        entropy_epi = entropy_total - entropy_alea
        return entropy_epi

def select_topk(
        prob_logits: ProbabilisticLogits,
        k: int,
        variant: Literal['var', 'logdet', 'entropy', 'map_mutual_info', 'exp_mutual_info'],
        entropy_variant: Literal['map_alea', 'exp_alea', 'comb', 'comb_covar'] = None,
        ignore_percentage: float = 0.0,
        return_values=False,
        seed: int = None,
    ) -> torch.Tensor:

    if ignore_percentage > 0.0:
        offset = int(prob_logits.mean.shape[0] * ignore_percentage)
    else:
        offset = 0
    
    n = k + offset
    n = min(n, prob_logits.mean.shape[0])

    complexity = complexity_score(prob_logits, variant, entropy_variant, seed=seed)

    if return_values:
        return complexity.topk(n).indices[offset:], complexity.topk(n).values[offset:]
    
    return complexity.topk(n).indices[offset:]

def select_topk_classbalanced(
        prob_logits: ProbabilisticLogits,
        class_ids: torch.Tensor, 
        k: int,
        variant: Literal['var', 'entropy'],
        entropy_variant: Literal['alea', 'comb', 'comb_covar'] = None,
    ) -> torch.Tensor:
    classes = class_ids.unique(sorted=True)
    elements_per_class = k // len(classes)
    residuals = k % len(classes)

    indices = []
    for i, c in enumerate(classes):
        n = elements_per_class
        if i < residuals:
            n = elements_per_class + 1
        
        mask = class_ids == c

        if variant == 'var':
            indices.append(prob_logits.var[mask].sum(dim=1).topk(n).indices)
        elif variant == 'entropy':
            entropy = _entropy(prob_logits.mean[mask], prob_logits.var[mask], entropy_variant)
            indices.append(entropy.topk(n).indices)
    
    return torch.cat(indices)

def select_topk_randomized(
    prob_logits: ProbabilisticLogits,
    k: int,
    temp: float,
    variant: Literal['var', 'logdet', 'entropy', 'map_mutual_info', 'exp_mutual_info'],
    entropy_variant: Literal['map_alea', 'exp_alea', 'comb', 'comb_covar'] = None,
    seed: int = 0,
):
    complexity = complexity_score(prob_logits, variant, entropy_variant)
    
    torch.manual_seed(seed)

    complexity = (complexity - complexity.mean()) / complexity.std()
    probs = torch.softmax(complexity * temp, dim=0)

    dist = torch.distributions.Categorical(probs=probs)
    return dist.sample((k,))
    


def select_random_classbalanced(logits_var: torch.Tensor, class_ids: torch.Tensor, k: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)

    classes = class_ids.unique(sorted=True)
    elements_per_class = k // len(classes)
    residuals = k % len(classes)

    indices = []
    for i, c in enumerate(classes):
        n = elements_per_class
        if i < residuals:
            n = elements_per_class + 1
        
        class_indices = torch.where(class_ids == c)[0]
        indices.append(class_indices[torch.randperm(len(class_indices))[:n]])
    
    return torch.cat(indices)


def select_random(prob_logits: ProbabilisticLogits, k: int, seed: int) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    N, *_ = prob_logits.var.shape
    return torch.randperm(N)[:k]


def create_subset_json(
        prob_logits: ProbabilisticLogits, 
        class_ids: torch.Tensor, 
        k: int,
    ) -> dict:
    d = {
        'topk_var': select_topk(prob_logits, k, variant='var').tolist(),
        'topk_entropy_alea': select_topk(prob_logits, k, variant='entropy', entropy_variant='alea').tolist(),
        'topk_entropy_comb': select_topk(prob_logits, k, variant='entropy', entropy_variant='comb').tolist(),
        'topk_entropy_comb_covar': select_topk(prob_logits, k, variant='entropy', entropy_variant='comb_covar').tolist(),
        # 'topk_cb_var': select_topk_classbalanced(prob_logits, class_ids, k, variant='var').tolist(),
        # 'topk_cb_entropy_alea': select_topk_classbalanced(prob_logits, class_ids, k, variant='entropy', entropy_variant='alea').tolist(),
        # 'topk_cb_entropy_comb': select_topk_classbalanced(prob_logits, class_ids, k, variant='entropy', entropy_variant='comb').tolist(),
    }

    for i in range(3):
        d[f'topk_randomized_var_{i}'] = select_topk_randomized(prob_logits, k, temp=1.0, variant='var', seed=i).tolist()
        d[f'topk_randomized_entropy_alea_{i}'] = select_topk_randomized(prob_logits, k, temp=1.0, variant='entropy', entropy_variant='alea', seed=i).tolist()
        d[f'topk_randomized_entropy_comb_{i}'] = select_topk_randomized(prob_logits, k, temp=1.0, variant='entropy', entropy_variant='comb', seed=i).tolist()
        d[f'topk_randomized_entropy_comb_covar_{i}'] = select_topk_randomized(prob_logits, k, temp=1.0, variant='entropy', entropy_variant='comb_covar', seed=i).tolist()

        # d[f'random_cb_{i}'] = select_random_classbalanced(prob_logits, class_ids, k, seed=i).tolist()
        d[f'random_{i}'] = select_random(prob_logits.var, k, seed=i).tolist()

    return d