from typing import Optional
from dataclasses import dataclass

import torch
import math
from tqdm import tqdm

from transformers import (
    AutoTokenizer, 
    CLIPTextModelWithProjection, 
    CLIPVisionModelWithProjection, 
    CLIPModel,
    SiglipTextModel,
    SiglipVisionModel,
    SiglipModel,
)

from bayesvlm.hessians import KroneckerFactorizedCovariance


PROJECTION_DIM = {
    'laion/CLIP-ViT-B-32-laion2B-s34B-b79K': 512,
    'laion/CLIP-ViT-L-14-laion2B-s32B-b82K': 768,
    'laion/CLIP-ViT-H-14-laion2B-s32B-b79K': 1024,
}

@dataclass
class EncoderResult:
    embeds: torch.Tensor
    activations: torch.Tensor
    residuals: torch.Tensor

    def __init__(self, embeds, activations, residuals=None):
        self.embeds = embeds
        self.activations = activations
        self.residuals = residuals if residuals is not None else torch.zeros_like(embeds)

    def clone(self):
        return EncoderResult(
            embeds=self.embeds.clone(),
            activations=self.activations.clone(),
            residuals=self.residuals.clone(),
        )
    
    def to(self, device):
        self.embeds = self.embeds.to(device)
        self.activations = self.activations.to(device)
        self.residuals = self.residuals.to(device)
        return self
    
    def __len__(self):
        return len(self.embeds)

    def __getitem__(self, idx):
        if isinstance(idx, (list, torch.Tensor)):
            return EncoderResult(
                embeds=self.embeds[idx],
                activations=self.activations[idx],
                residuals=self.residuals[idx],
            )
        return self.embeds[idx], self.activations[idx], self.residuals[idx]

@dataclass
class ProbabilisticLogits:
    mean: torch.Tensor
    var: torch.Tensor

    def softmax(self, dim=-1, num_samples=400, chunk_size=10000, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
            
        std = torch.sqrt(self.var)

        if num_samples == 0:
            # multiclass probit approximation
            variance = self.var.diagonal(dim1=-2, dim2=-1)
            scaled_mean = self.mean / torch.sqrt(1 + torch.pi / 8 * variance)
            return torch.nn.functional.softmax(scaled_mean, dim=dim)

        probas = torch.zeros_like(self.mean)

        if self.var.ndim == 2:
            for _ in range(num_samples):
                eps = torch.randn(std.shape, device=std.device) * std
                probas += torch.nn.functional.softmax(self.mean + eps, dim=dim)
        
        elif self.var.ndim == 3:
            num_chunks = math.ceil(self.mean.shape[0] / chunk_size)
            mean_chunks = torch.chunk(self.mean, num_chunks, dim=0)
            var_chunks = torch.chunk(self.var, num_chunks, dim=0)

            probas = []
            for mean_chunk, var_chunk in tqdm(zip(mean_chunks, var_chunks), total=num_chunks):
                dist = torch.distributions.MultivariateNormal(mean_chunk, covariance_matrix=var_chunk)
                probas_chunk = 0
                for _ in range(num_samples):
                    sample = dist.sample()
                    probas_chunk += torch.nn.functional.softmax(sample, dim=dim)
                probas.append(probas_chunk)

            probas = torch.concat(probas, dim=0)
        
        return probas / num_samples
    
    def sample_probas(self, num_samples: int, seed=None):
        """
        Sample from the distribution and return the softmax probabilities.

        Args:
            num_samples (int): Number of samples to draw from the distribution.

        Returns:
            torch.Tensor: [N, num_samples, num_classes]
        """

        if seed is not None:
            torch.manual_seed(seed)

        if self.var.ndim == 2:
            std = torch.sqrt(self.var)
            samples = torch.randn((num_samples,) + self.mean.shape, device=self.mean.device) * std + self.mean
            samples = samples.permute(1, 0, 2)
            return torch.nn.functional.softmax(samples, dim=2)
        
        elif self.var.ndim == 3:
            dist = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.var)

            # why does torch allocate too much memory for sample((num_samples, ))?
            samples = []
            for _ in range(num_samples):
                sample = dist.sample((1, ))
                samples.append(sample)
            samples = torch.cat(samples, dim=0)

            samples = samples.permute(1, 0, 2)
            return torch.nn.functional.softmax(samples, dim=2)

        else:
            raise ValueError("Invalid variance tensor shape.")

    
    def expected_aleatoric_entropy(self, num_samples=400, dim=-1):
        entropy = 0

        if self.var.ndim == 2:
            for _ in range(num_samples):
                eps = torch.randn(self.var.shape, device=self.var.device) * torch.sqrt(self.var)
                probas = torch.nn.functional.softmax(self.mean + eps, dim=dim)
                entropy += -(probas * probas.log()).sum(dim=dim)

        elif self.var.ndim == 3:
            dist = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.var)
            for _ in range(num_samples):
                sample = dist.sample()
                probas = torch.nn.functional.softmax(sample, dim=dim)
                entropy += -(probas * probas.log()).sum(dim=dim)
            
        return entropy / num_samples
    
    def __getitem__(self, idx):
        return ProbabilisticLogits(
            mean=self.mean[idx],
            var=self.var[idx],
        )
    
    def to(self, device):
        self.mean = self.mean.to(device)
        self.var = self.var.to(device)
        return self
    
    def detach(self):
        return ProbabilisticLogits(
            mean=self.mean.detach(),
            var=self.var.detach(),
        )

    def cross_entropy(self, target, num_samples=400, reduction='sum'):
        if num_samples == 0:
            variance = self.var.diagonal(dim1=-2, dim2=-1)
            scaled_mean = self.mean / torch.sqrt(1 + torch.pi / 8 * variance)
            return torch.nn.functional.cross_entropy(scaled_mean, target, reduction=reduction)

        loss = 0

        if self.var.ndim == 2:
            variance = self.var.diagonal(dim1=-2, dim2=-1)
            for _ in range(num_samples):
                eps = torch.randn(self.var.shape, device=self.var.device) * torch.sqrt(variance)
                logits = self.mean + eps
                loss += torch.nn.functional.cross_entropy(logits, target, reduction=reduction)
        
        elif self.var.ndim == 3:
            dist = torch.distributions.MultivariateNormal(self.mean, covariance_matrix=self.var)
            for _ in range(num_samples):
                sample = dist.sample()
                loss += torch.nn.functional.cross_entropy(sample, target, reduction=reduction)
        
        return loss / num_samples
    
    def clone(self):
        return ProbabilisticLogits(
            mean=self.mean.clone(),
            var=self.var.clone(),
        )

class CLIPTextEncoder(torch.nn.Module):
    def __init__(
            self, 
            text_model: CLIPTextModelWithProjection,
            tokenizer: AutoTokenizer,
        ):
        super().__init__()
        self.text_encoder = text_model.text_model
        self.text_projection = text_model.text_projection
        self.tokenizer = tokenizer
        self.device = text_model.device
    
    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        if projection_dim is None:
            projection_dim = PROJECTION_DIM[model_name]

        text_model = CLIPTextModelWithProjection.from_pretrained(model_name, projection_dim=projection_dim)
        model = cls(text_model, tokenizer)
        model = model.to(device) if device is not None else model
        model.device = device
        return model
    
    def save_projection_weights(self, path: str):
        torch.save(self.text_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.text_projection.load_state_dict(state_dict)
            return
        
        if path is None:
            raise ValueError("Either path or state_dict must be provided.")
        
        state_dict = torch.load(path)
        self.text_projection.load_state_dict(state_dict)
    
    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers()
        for param in self.text_projection.parameters():
            param.requires_grad = True

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.text_projection.train()
            for param in self.text_projection.parameters():
                param.requires_grad = True
        
        for layer in self.text_encoder.encoder.layers[-k_last_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, batch, return_activations=False):
        texts = batch['text']
        text_input = self.tokenizer(text=texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        text_outputs = self.text_encoder(**text_input)
        text_pooled_output = text_outputs[1]
        text_embeds = self.text_projection(text_pooled_output)

        if return_activations:
            return EncoderResult(embeds=text_embeds, activations=text_pooled_output)
        
        return text_embeds
    
class CLIPImageEncoder(torch.nn.Module):
    def __init__(
            self, 
            vision_model: CLIPVisionModelWithProjection,
        ):
        super().__init__()
        self.vision_encoder = vision_model.vision_model
        self.vision_projection = vision_model.visual_projection
        self.device = vision_model.device

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        projection_dim: Optional[int] = None,
        device: Optional[str] = None,
    ):
        if projection_dim is None:
            projection_dim = PROJECTION_DIM[model_name]

        vision_model = CLIPVisionModelWithProjection.from_pretrained(
            model_name,
            projection_dim=projection_dim,
        )
        model = cls(vision_model)
        model = model.to(device) if device is not None else model
        model.device = device
        return model
    
    def save_projection_weights(self, path: str):
        torch.save(self.vision_projection.state_dict(), path)
    
    def load_projection_weights(
        self, 
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.vision_projection.load_state_dict(state_dict)
            return
        
        if path is None:
            raise ValueError("Either path or state_dict must be provided.")
        
        state_dict = torch.load(path)
        self.vision_projection.load_state_dict(state_dict)

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers()
        for param in self.vision_projection.parameters():
            param.requires_grad = True

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.vision_projection.train()
            for param in self.vision_projection.parameters():
                param.requires_grad = True
        
        for layer in self.vision_encoder.encoder.layers[-k_last_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True
    

    def forward(self, batch, return_activations=False):
        images = batch['image']
        image_input = dict(pixel_values=images.to(self.device))
        image_outputs = self.vision_encoder(**image_input)
        image_pooled_output = image_outputs[1]
        image_embeds = self.vision_projection(image_pooled_output)

        if return_activations:
            return EncoderResult(embeds=image_embeds, activations=image_pooled_output)
        
        return image_embeds

class SiglipTextEncoder(torch.nn.Module):
    def __init__(
        self, 
        model: SiglipTextModel,
        tokenizer: AutoTokenizer,
    ):
        super().__init__()
        self._siglip_text_transformer = model.text_model
        self.text_projection = model.text_model.head
        self.tokenizer = tokenizer

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
    ):
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = SiglipTextModel.from_pretrained(model_name)
        model = cls(model, tokenizer)
        model = model.to(device) if device is not None else model
        model.device = device
        return model
    
    def save_projection_weights(self, path: str):
        torch.save(self.text_projection.state_dict(), path)

    def load_projection_weights(
        self,
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.text_projection.load_state_dict(state_dict)
            return
        
        if path is None:
            raise ValueError("Either path or state_dict must be provided.")
        
        state_dict = torch.load(path)
        self.text_projection.load_state_dict(state_dict)

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers()
        for param in self.text_projection.parameters():
            param.requires_grad = True

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.text_projection.train()
            for param in self.text_projection.parameters():
                param.requires_grad = True
        
        for layer in self._siglip_text_transformer.encoder.layers[-k_last_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True
    
    def forward(self, batch, return_activations=False):
        texts = batch['text']
        text_input = self.tokenizer(text=texts, padding='max_length', truncation=True, return_tensors="pt").to(self.device)
        hidden_states = self._siglip_text_transformer.embeddings(**text_input)
        encoder_outputs = self._siglip_text_transformer.encoder(hidden_states)
        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self._siglip_text_transformer.final_layer_norm(last_hidden_state)
        pooled_output = last_hidden_state[:, -1, :]

        text_embeds = self.text_projection(pooled_output)

        if return_activations:
            return EncoderResult(embeds=text_embeds, activations=pooled_output)
        
        return text_embeds

class SiglipVisionEncoderWithoutProjection(torch.nn.Module):
    def __init__(
        self, 
        model: SiglipVisionModel,
    ):
        super().__init__()
        self.vision_model = model.vision_model 
    
    def forward(self, pixel_values: torch.Tensor):
        hidden_states = self.vision_model.embeddings(pixel_values)
        encoder_outputs = self.vision_model.encoder(inputs_embeds=hidden_states)

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.vision_model.post_layernorm(last_hidden_state)

        batch_size = last_hidden_state.shape[0]
        probe = self.vision_model.head.probe.repeat(batch_size, 1, 1)
        last_hidden_state = self.vision_model.head.attention(probe, last_hidden_state, last_hidden_state)[0]

        residual = last_hidden_state
        last_hidden_state = self.vision_model.head.layernorm(last_hidden_state)
        mlp = self.vision_model.head.mlp

        last_hidden_state = mlp.fc1(last_hidden_state)
        last_hidden_state = mlp.activation_fn(last_hidden_state)

        return last_hidden_state, residual

class SiglipImageEncoder(torch.nn.Module):
    def __init__(
        self, 
        vision_model: SiglipVisionModel,
    ):
        super().__init__()
        self.vision_encoder = SiglipVisionEncoderWithoutProjection(vision_model)
        self.vision_projection = vision_model.vision_model.head.mlp.fc2

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
    ):
        vision_model = SiglipVisionModel.from_pretrained(model_name)
        model = cls(vision_model)
        model = model.to(device) if device is not None else model
        model.device = device
        return model
    
    def save_projection_weights(self, path: str):
        torch.save(self.vision_projection.state_dict(), path)
    
    def load_projection_weights(
        self, 
        *,
        path: Optional[str] = None,
        state_dict: Optional[dict] = None,
    ):
        if state_dict is not None:
            self.vision_projection.load_state_dict(state_dict)
            return
        
        if path is None:
            raise ValueError("Either path or state_dict must be provided.")
        
        state_dict = torch.load(path)
        self.vision_projection.load_state_dict(state_dict)

    def freeze_all_layers(self):
        for param in self.parameters():
            param.requires_grad = False

    def freeze_all_layers_exept_projection(self):
        self.freeze_all_layers()
        for param in self.vision_projection.parameters():
            param.requires_grad = True

    def enable_gradients(
        self,
        k_last_layers: int = 0,
        enable_projection: bool = True,
    ):
        if enable_projection:
            self.vision_projection.train()
            for param in self.vision_projection.parameters():
                param.requires_grad = True
        
        for layer in self.vision_encoder.encoder.layers[-k_last_layers:]:
            layer.train()
            for param in layer.parameters():
                param.requires_grad = True

    def forward(self, batch, return_activations=False):
        images = batch['image']
        image_input = images.to(self.device)
        activations, residuals = self.vision_encoder(image_input)

        activations = activations[:, 0]
        residuals = residuals[:, 0]

        image_embeds = self.vision_projection(activations) + residuals

        if return_activations:
            return EncoderResult(embeds=image_embeds, activations=activations, residuals=residuals)
        
        return image_embeds
    
class CLIP(torch.nn.Module):
    source_projection_has_bias = False
    target_projection_has_bias = False

    def __init__(
        self,
        logit_scale: float,
        logit_bias: float = 0,
        source_covariance: KroneckerFactorizedCovariance = None,
        target_covariance: KroneckerFactorizedCovariance = None,
        device: Optional[str] = None,
    ):
        super().__init__()
        self.logit_scale = torch.nn.Parameter(torch.ones([], device=device) * logit_scale)
        self.logit_bias = torch.nn.Parameter(torch.ones([], device=device) * logit_bias)
        self.source_covariance = source_covariance
        self.target_covariance = target_covariance

    @property
    def device(self):
        return self.logit_scale.data.device

    def set_covariances(
        self,
        source_covariance: KroneckerFactorizedCovariance = None,
        target_covariance: KroneckerFactorizedCovariance = None,
    ):
        self.source_covariance = KroneckerFactorizedCovariance(
            A_inv=source_covariance.A_inv.clone().to(self.device),
            B_inv=source_covariance.B_inv.clone().to(self.device),
        ) if source_covariance is not None else None
        
        self.target_covariance = KroneckerFactorizedCovariance(
            A_inv=target_covariance.A_inv.clone().to(self.device),
            B_inv=target_covariance.B_inv.clone().to(self.device),
        ) if target_covariance is not None else None

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
    ):
        clip = CLIPModel.from_pretrained(model_name)
        model = cls(
            logit_scale=clip.logit_scale.item(),
        )
        model = model.to(device) if device is not None else model
        return model
    
    def _compute_logits(
        self,
        source_embeds: torch.Tensor,
        target_embeds: torch.Tensor,
    ):
        # normalize 
        source_embeds = source_embeds / source_embeds.norm(p=2, dim=-1, keepdim=True)
        target_embeds = target_embeds / target_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity
        similarity = torch.matmul(source_embeds, target_embeds.t()) * self.logit_scale.exp() + self.logit_bias
        return similarity

    def _compute_probabilistic_logits_smith(
        self,
        source_results: EncoderResult,
        target_results: EncoderResult,
        compute_covariance: bool = False,
    ):
        """
        This function compute the expected value and variance of the cosine similarity between two probabilistic embeddings.
        The derivation adopts the approach by Smith et al. (2023).
        """
        
        if compute_covariance:
            raise NotImplementedError("Only the variances are supported for now.")
        
        source_covariance = self.source_covariance
        target_covariance = self.target_covariance

        source_activations = source_results.activations
        target_activations = target_results.activations

        if self.source_projection_has_bias:
            source_activations = torch.cat([source_activations, torch.ones_like(source_activations[:, :1])], dim=-1)
        
        if self.target_projection_has_bias:
            target_activations = torch.cat([target_activations, torch.ones_like(target_activations[:, :1])], dim=-1)

        source_embeds = source_results.embeds
        target_embeds = target_results.embeds

        source_B_factor = source_covariance.B_inv.diagonal()
        target_B_factor = target_covariance.B_inv.diagonal()

        source_diag_cov = torch.einsum('ij,jk,ik->i', source_activations, source_covariance.A_inv, source_activations)[:,None] * source_B_factor
        target_diag_cov = torch.einsum('ij,jk,ik->i', target_activations, target_covariance.A_inv, target_activations)[:,None] * target_B_factor

        norm_source = source_embeds**2 + source_diag_cov
        expect_norm_source = norm_source.sum(dim=-1, keepdim=True)
        norm_target = target_embeds**2 + target_diag_cov
        expect_norm_target = norm_target.sum(dim=-1, keepdim=True)

        # compute expected value
        expected_similarity = torch.matmul(source_embeds/torch.sqrt(expect_norm_source), (target_embeds/torch.sqrt(expect_norm_target)).t())

        # compute variance 
        term1 = torch.matmul(norm_source, target_diag_cov.t())
        term2 = torch.matmul(source_diag_cov, (target_embeds**2).t())
                
        variance_similarity = (term1 + term2)/(expect_norm_source*expect_norm_target.t())

        scale = self.logit_scale.exp()

        return ProbabilisticLogits(
            mean=expected_similarity * scale, 
            var=variance_similarity*(scale**2),
        )
        
    def forward(
            self, 
            source_embeds: torch.Tensor | EncoderResult,
            target_embeds: torch.Tensor | EncoderResult,
            map_estimate: bool = False,
        ):
        """
        Args:
            from_embeds (torch.Tensor): [batch_size, embed_dim]
            to_embeds (torch.Tensor): [batch_size, embed_dim]
            map_estimate (bool)

        Returns:
            similarity (torch.Tensor): [#from, #to]
        """
        
        if isinstance(source_embeds, EncoderResult) and isinstance(target_embeds, EncoderResult):
            if map_estimate:
                logits_map = self._compute_logits(source_embeds.embeds, target_embeds.embeds)
                covar_map = torch.zeros_like(logits_map)
                return ProbabilisticLogits(mean=logits_map, var=covar_map)
            
            return self._compute_probabilistic_logits_smith(source_embeds, target_embeds)
        
        return self._compute_logits(source_embeds, target_embeds)
  
class SIGLIP(CLIP):
    source_projection_has_bias = True
    target_projection_has_bias = True

    @classmethod
    def from_huggingface(
        cls,
        model_name: str,
        device: Optional[str] = None,
    ):
        siglip = SiglipModel.from_pretrained(model_name)
        model = cls(
            logit_scale=siglip.logit_scale.item(),
            logit_bias=siglip.logit_bias.item(),
        )
        model = model.to(device) if device is not None else model
        return model