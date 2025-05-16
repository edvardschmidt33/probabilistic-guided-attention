import torch
from tqdm import tqdm
from collections import OrderedDict
from bayesvlm.vlm import EncoderResult

def diagonal_wasserstein_distance(mu1, mu2, cov1, cov2):
    # Compute L2 squared distance between means
    l2_squared = torch.cdist(mu1, mu2)**2

    # Compute sum of variances (diagonal elements of covariance matrices)
    var_prod = 2*torch.einsum('ak,bk->ab', torch.sqrt(cov1), torch.sqrt(cov2))

    # Compute the diagonal Wasserstein distance
    diagonal_wasserstein = l2_squared + cov1.sum(dim=-1)[:,None] + cov2.sum(dim=-1)[None, :] - var_prod

    return diagonal_wasserstein

def wdist2(mu1, mu2, cov1, cov2):
    # Compute the Sliced Wasserstein distance
    return diagonal_wasserstein_distance(mu1, mu2, cov1, cov2)

def _remove_last_elements_to_keep_n_unique(indices: torch.Tensor, n):
    while len(torch.unique(indices)) > n:
        indices = indices[:-1]
    return indices
  
def extract_test_train_indices(text_idx_to_train_data):
    test_indices = []
    train_indices = []
    for test_idx, data in text_idx_to_train_data.items():
        test_indices.append(int(test_idx))
        train_indices.extend([int(x) for x in data['indices']])

    # remove duplicates
    train_indices = list(set(train_indices))

    return dict(test=test_indices, train=train_indices)
  
def find_similar_samples_cosine(
    train: EncoderResult,
    test: EncoderResult,
    indices_test: torch.Tensor,
    values_test: torch.Tensor,
    k_nearest: int,
    source_covariance,
    device: str,
    buffersize=150,
):
    """
    Based on the embeddings of the test set, find the k_nearest neighbors in the training set.

    Args:
        embeds_train: Embeddings of the training set
        embeds_test: Embeddings of the test set
        k_nearest: Number of nearest neighbors to find
        device: Device to run the computations on
    """
    
    train_activations = train.activations.to(device)
    test_activations = test.activations[indices_test].to(device)
    
    embeds_train = train.embeds.to(device)
    embeds_test_subset = test.embeds[indices_test].to(device)
    
    # embeds_train = embeds_train.to(device)
    # embeds_test_subset = embeds_test[indices_test].to(device)
    
    source_B_factor = source_covariance.B_inv.diagonal()

    train_diag_cov = torch.einsum('ij,jk,ik->i', train_activations, source_covariance.A_inv, train_activations)[:,None] * source_B_factor
    test_diag_cov = torch.einsum('ij,jk,ik->i', test_activations, source_covariance.A_inv, test_activations)[:,None] * source_B_factor
        
    norm_train = embeds_train**2 + train_diag_cov
    expect_norm_train = norm_train.sum(dim=-1, keepdim=True)
    norm_test = embeds_test_subset**2 + test_diag_cov
    expect_norm_test = norm_test.sum(dim=-1, keepdim=True)

    # compute expected value
    embeds_train = embeds_train / torch.sqrt(expect_norm_train)
    embeds_test_subset  = embeds_test_subset / torch.sqrt(expect_norm_test)
    
    expected_similarity = embeds_test_subset @ embeds_train.t()

    # embeds_train = embeds_train / embeds_train.norm(dim=-1, keepdim=True)
    # embeds_test_subset = embeds_test_subset / embeds_test_subset.norm(dim=-1, keepdim=True)

    # [N_test, N_train]
    # similarities = embeds_test_subset @ embeds_train.t()

    # [N_test, k_nearest + buffersize]
    topk = expected_similarity.topk(min(k_nearest + buffersize, len(embeds_train)), dim=1)

    done = False
    k_ = k_nearest
    while not done:
        # flatten the indices after transposing so that we do still have the topk indices for each test sample
        indices = topk.indices[:, :k_].T.flatten()
        unique_indices = torch.unique(indices, sorted=False)

        print(f"Unique indices:", unique_indices.size(0))
        print("Goal size:", k_nearest * embeds_test_subset.size(0))
        print(f"K: {k_}")

        if unique_indices.size(0) >= k_nearest * embeds_test_subset.size(0):
            # only keep the first k_nearest * N_test indices
            first_unique_indices = _remove_last_elements_to_keep_n_unique(indices, k_nearest * embeds_test_subset.size(0))
            done = True
            break

        k_ += 1

    unique_indices = torch.unique(first_unique_indices, sorted=False)

    indices = topk.indices[:, :k_]
    values = topk.values[:, :k_]

    text_idx_to_train_data = OrderedDict()
    for i, (topk_idx, topk_val) in enumerate(zip(topk.indices, topk.values)):
        test_idx = indices_test[i].item()
        test_value = values_test[i].item()

        topk_idx = topk_idx[:k_]
        topk_val = topk_val[:k_]

        keep_ids, keep_val = [], []
        for idx, val in zip(topk_idx, topk_val):
            if idx in unique_indices:
                keep_ids.append(idx.item())
                keep_val.append(val.item())

        text_idx_to_train_data[test_idx] = dict(
            score=test_value,
            indices=keep_ids,
            similarities=keep_val,
        )
        
    return text_idx_to_train_data
  
def find_similar_samples_wasserstein(
    train: EncoderResult,
    test: EncoderResult,
    indices_test: torch.Tensor,
    values_test: torch.Tensor,
    k_nearest: int,
    source_covariance,
    device: str,
    buffersize=150,
):
    """
    Based on the embeddings of the test set, find the k_nearest neighbors in the training set.

    Args:
        embeds_train: Embeddings of the training set
        embeds_test: Embeddings of the test set
        k_nearest: Number of nearest neighbors to find
        device: Device to run the computations on
    """
       
    train_activations = train.activations.to(device)
    test_activations = test.activations[indices_test].to(device)
    
    train_embeds = train.embeds.to(device)
    test_embeds = test.embeds[indices_test].to(device)
    
    source_B_factor = source_covariance.B_inv.diagonal()

    train_diag_cov = torch.einsum('ij,jk,ik->i', train_activations, source_covariance.A_inv, train_activations)[:,None] * source_B_factor
    test_diag_cov = torch.einsum('ij,jk,ik->i', test_activations, source_covariance.A_inv, test_activations)[:,None] * source_B_factor

    done = False
    k_ = k_nearest
    
    similarities = wdist2(test_embeds, train_embeds, test_diag_cov, train_diag_cov) * -1
    topk = similarities.topk(min(k_nearest + buffersize, len(train_embeds)), dim=1)

    done = False
    k_ = k_nearest
    while not done:
        # flatten the indices after transposing so that we do still have the topk indices for each test sample
        indices = topk.indices[:, :k_].T.flatten()
        unique_indices = torch.unique(indices, sorted=False)

        print(f"Unique indices:", unique_indices.size(0))
        print("Goal size:", k_nearest * test_embeds.size(0))
        print(f"K: {k_}")

        if unique_indices.size(0) >= k_nearest * test_embeds.size(0):
            # only keep the first k_nearest * N_test indices
            first_unique_indices = _remove_last_elements_to_keep_n_unique(indices, k_nearest * test_embeds.size(0))
            done = True
            break

        k_ += 1

    unique_indices = torch.unique(first_unique_indices, sorted=False)

    indices = topk.indices[:, :k_]
    values = topk.values[:, :k_]

    text_idx_to_train_data = OrderedDict()
    for i, (topk_idx, topk_val) in enumerate(zip(topk.indices, topk.values)):
        test_idx = indices_test[i].item()
        test_value = values_test[i].item()

        topk_idx = topk_idx[:k_]
        topk_val = topk_val[:k_]

        keep_ids, keep_val = [], []
        for idx, val in zip(topk_idx, topk_val):
            if idx in unique_indices:
                keep_ids.append(idx.item())
                keep_val.append(val.item())

        text_idx_to_train_data[test_idx] = dict(
            score=test_value,
            indices=keep_ids,
            similarities=keep_val,
        )

    return text_idx_to_train_data
