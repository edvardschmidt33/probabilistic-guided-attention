import torch
from pathlib import Path
import json
import math
from typing import Tuple, Literal
from dataclasses import dataclass
from tqdm import tqdm


def compute_hessian_analytic_InfoNCE(
    source_embeds: torch.Tensor,
    target_embeds: torch.Tensor,
    logit_scale: torch.Tensor,
):
    source_embeds_norm = source_embeds.norm(p=2, dim=-1, keepdim=True)
    target_embeds_norm = target_embeds.norm(p=2, dim=-1, keepdim=True)

    # [B, D]
    normalized_source_embed = source_embeds / source_embeds_norm
    # [C, D]
    normalized_target_embeds = target_embeds / target_embeds_norm

    # [B, C]
    logits = (normalized_source_embed @ normalized_target_embeds.T) * torch.exp(logit_scale)

    # [B, C]
    probs = logits.softmax(dim=-1)

    # [B, D, D]
    J_diag = (normalized_target_embeds * probs.unsqueeze(-1)).mT @ normalized_target_embeds

    # [B, D, 1]
    Yp = normalized_target_embeds.mT @ probs.unsqueeze(-1)
    
    # [B, D, D]
    J_pp = Yp @ Yp.mT

    # [B, D, D]
    norm_eye = torch.eye(source_embeds.shape[-1], device=source_embeds.device) / source_embeds_norm.unsqueeze(-1)
    # [B, D, D]
    norm_outer = torch.einsum('bi,bj->bij', source_embeds, source_embeds) / (source_embeds_norm**3).unsqueeze(-1)
    # [B, D, D]
    J_norm = norm_eye - norm_outer

    # [B, D, D]
    H = J_norm @ (J_diag - J_pp) @ J_norm.mT * torch.exp(logit_scale)**2

    return H.sum(dim=0)

def compute_hessian_analytic_SigLIP(
    x_batch: torch.Tensor, 
    indices_batch: torch.Tensor,
    y: torch.Tensor, 
    logit_scale: torch.Tensor, 
    logit_bias: torch.Tensor,
    chunk_size_j: int = None,
) -> torch.Tensor:
    """
    Compute the Hessian of the siglip loss with respect to x

    Args:
        x: The image tensor [B, D]
        y: The text tensor [B, D]
        log_temp: The log temperature parameter
        log_b: The log bias parameter

    Returns:
        The Hessian of the siglip loss with respect to x [D, D]
    """
    
    if chunk_size_j is None:
        chunk_size_j = y.shape[0]

    _, D_x = x_batch.shape
    N_y, D_y = y.shape

    assert D_x == D_y, "The input and output dimensions must be the same"

    x_batch_norm = x_batch.norm(dim=1, keepdim=True)
    y_norm = y.norm(dim=1, keepdim=True)

    x_normalized = x_batch / x_batch_norm
    y_normalized = y / y_norm

    scale = torch.exp(logit_scale)

    # todo: do this in a chunked way
    logits = x_normalized @ y_normalized.T * scale + logit_bias
    labels = 2 * torch.eye(N_y, device=logits.device, dtype=x_batch.dtype) - 1
    labels = labels[indices_batch, :]

    # [B, B]
    sig = torch.sigmoid(logits * labels)
    scale = scale.square() * sig * (1 - sig)

    hessians = 0

    for j in range(0, N_y, chunk_size_j):
        chunk_y_normalized = y_normalized[j:j + chunk_size_j]
        chunk_scale = scale[:, j:j + chunk_size_j]

        # [B_, D, D]
        outer = torch.einsum("ij,ik->ijk", chunk_y_normalized, chunk_y_normalized)

        # [B_, D, D]
        hess = torch.einsum('ij,jkl->ikl', chunk_scale, outer)

        # [B_, D, D]
        norm_jac_eye = torch.eye(D_x, device=x_batch.device, dtype=x_batch.dtype).unsqueeze(0) / x_batch_norm.unsqueeze(-1)
        norm_jac_outer = torch.einsum("ij,ik->ijk", x_batch, x_batch) / (x_batch_norm.unsqueeze(-1) ** 3)
        norm_jac = norm_jac_eye - norm_jac_outer

        jac_hess_jac = torch.einsum("ijk,ikl,ilm->jm", norm_jac, hess, norm_jac)
        
        hessians += jac_hess_jac 
    
    return hessians


@dataclass 
class KroneckerFactorizedCovariance:
    A_inv: torch.Tensor
    B_inv: torch.Tensor

    def clone(self):
        return KroneckerFactorizedCovariance(
            A_inv=self.A_inv.clone(),
            B_inv=self.B_inv.clone(),
        )
    
    def to(self, device):
        self.A_inv = self.A_inv.to(device)
        self.B_inv = self.B_inv.to(device)
        return self


def load_covariances(
    la_dir: str,
    return_info: bool = False,
) -> Tuple[KroneckerFactorizedCovariance, KroneckerFactorizedCovariance]:
    A_img = torch.load(Path(la_dir) / 'A_img_analytic.pt', map_location='cpu')
    B_img = torch.load(Path(la_dir) / 'B_img_analytic.pt', map_location='cpu')
    A_txt = torch.load(Path(la_dir) / 'A_txt_analytic.pt', map_location='cpu')
    B_txt = torch.load(Path(la_dir) / 'B_txt_analytic.pt', map_location='cpu')

    with open(Path(la_dir) / 'prior_precision_analytic.json') as f:
        info = json.load(f)

    A_img = A_img * math.sqrt(info['n_img']) + math.sqrt(info['lambda_img']) * torch.eye(A_img.size(0), device=A_img.device, dtype=A_img.dtype)
    B_img = B_img * math.sqrt(info['n_img']) + math.sqrt(info['lambda_img']) * torch.eye(B_img.size(0), device=B_img.device, dtype=B_img.dtype)
    A_txt = A_txt * math.sqrt(info['n_txt']) + math.sqrt(info['lambda_txt']) * torch.eye(A_txt.size(0), device=A_txt.device, dtype=A_txt.dtype)
    B_txt = B_txt * math.sqrt(info['n_txt']) + math.sqrt(info['lambda_txt']) * torch.eye(B_txt.size(0), device=B_txt.device, dtype=B_txt.dtype)

    cov_img = KroneckerFactorizedCovariance(
        A_inv=torch.linalg.inv(A_img),
        B_inv=torch.linalg.inv(B_img),
    )

    cov_txt = KroneckerFactorizedCovariance(
        A_inv=torch.linalg.inv(A_txt),
        B_inv=torch.linalg.inv(B_txt),
    )

    if return_info:
        return cov_img, cov_txt, info

    return cov_img, cov_txt


def _compute_covariance(
    A: torch.Tensor,
    B: torch.Tensor,
    n: torch.Tensor,
    lmbda: torch.Tensor,
):
    sqrt_n = torch.sqrt(n)
    sqrt_lmbda = torch.sqrt(lmbda)
    A = A * sqrt_n + sqrt_lmbda * torch.eye(A.size(0), device=A.device, dtype=A.dtype)
    B = B * sqrt_n + sqrt_lmbda * torch.eye(B.size(0), device=B.device, dtype=B.dtype)

    return KroneckerFactorizedCovariance(
        A_inv=torch.linalg.inv(A),
        B_inv=torch.linalg.inv(B),
    )


def compute_covariances(
    A_img: torch.Tensor,
    B_img: torch.Tensor,
    A_txt: torch.Tensor,
    B_txt: torch.Tensor,
    info: dict,
):
    n_img = torch.tensor(info['n_img'], dtype=A_img.dtype, device=A_img.device)
    n_txt = torch.tensor(info['n_txt'], dtype=A_txt.dtype, device=A_txt.device)
    lambda_img = torch.tensor(info['lambda_img'], dtype=A_img.dtype, device=A_img.device)
    lambda_txt = torch.tensor(info['lambda_txt'], dtype=A_txt.dtype, device=A_txt.device)

    cov_img = _compute_covariance(A_img, B_img, n_img, lambda_img)
    cov_txt = _compute_covariance(A_txt, B_txt, n_txt, lambda_txt)
    return cov_img, cov_txt

def load_hessians(
    la_dir: str,
    tag: Literal['img', 'txt'],
    return_info: bool = False,
):
    A = torch.load(Path(la_dir) / f'A_{tag}_analytic.pt', map_location='cpu')
    B = torch.load(Path(la_dir) / f'B_{tag}_analytic.pt', map_location='cpu')

    if not return_info:
        return A, B

    with open(Path(la_dir) / f'prior_precision_analytic.json') as f:
        info = json.load(f)

    return A, B, info

def optimize_prior_precision(
    projection: torch.nn.Module,
    A: torch.Tensor,
    B: torch.Tensor,
    lmbda_init: float,
    n: float,
    lr: float,
    num_steps: int,
    device: str,
    retain_graph: bool = False,
    verbose: bool = False
) -> torch.Tensor:
    for param in projection.parameters():
        param.requires_grad = False

    projection_norm = l2_norm_squared(projection)
    num_params_projection = num_params(projection)

    A = A.to(device)
    B = B.to(device)

    # optimize prior precision
    log_lmbda = torch.nn.Parameter(
        torch.tensor(lmbda_init, device=device, requires_grad=True, dtype=torch.float32).log()
    )
    sqrt_n = torch.tensor(n, device=device, requires_grad=False, dtype=torch.float32).sqrt()

    optimizer = torch.optim.Adam([log_lmbda], lr=lr, maximize=True)

    for epoch in tqdm(range(num_steps), total=num_steps, disable=not verbose):
        optimizer.zero_grad()

        lmbda = log_lmbda.exp()
        sqrt_lmbda = lmbda.sqrt()

        # add prior to the loss
        A_ = A * sqrt_n + sqrt_lmbda * torch.eye(A.shape[0], device=device, dtype=A.dtype)
        B_ = B * sqrt_n + sqrt_lmbda * torch.eye(B.shape[0], device=device, dtype=B.dtype)

        log_prior = compute_log_prior(projection_norm, num_params_projection, lmbda)
        log_det = compute_log_det_kfac(A_, B_)
        marglik = log_prior - log_det

        marglik.backward(retain_graph=retain_graph)
        optimizer.step()

    return log_lmbda.exp()

def l2_norm_squared(module: torch.nn.Module):
    return sum((p ** 2).sum() for p in module.parameters())

def num_params(module: torch.nn.Module):
    return sum(p.numel() for p in module.parameters())

def compute_log_prior(l2_norm_squared: torch.Tensor, num_params: int, lmbda: float):
    return -0.5 * lmbda * l2_norm_squared + 0.5 * num_params * torch.log(lmbda)

def compute_log_det_kfac(A: torch.Tensor, B: torch.Tensor):
    logdet_A = torch.logdet(A)
    logdet_B = torch.logdet(B)
    p, q = A.shape[0], B.shape[0]
    return logdet_A * p + logdet_B * q
