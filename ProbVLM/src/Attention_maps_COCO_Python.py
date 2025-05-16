
import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import sys
import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np
from tqdm import tqdm

# Add the custom CLIP path
# custom_clip_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../GALS/CLIP'))
# sys.path.insert(0, custom_clip_path)

import clip  # This will now use your custom CLIP

from ds import prepare_coco_dataloaders_extra, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders
import torch.distributions as dist

import matplotlib.pyplot as plt


from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16") 


import torch
device = "cuda"
# Path to the saved model checkpoint ---> this pth needs to be updated
checkpoint_path = '/mimer/NOBACKUP/groups/ulio_inverse/ds-project/ProbVLM/ckpt/ProbVLM_Coco_extra_26.11_best.pth'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)


import torch.nn as nn
class BayesCap_MLP(nn.Module):
    '''
    Baseclass to create a simple MLP
    Inputs
        inp_dim: int, Input dimension
        out_dim: int, Output dimension
        hid_dim: int, hidden dimension
        num_layers: Number of hidden layers
        p_drop: dropout probability 
    '''
    def __init__(
        self, 
        inp_dim, 
        out_dim,
        hid_dim=512, 
        num_layers=1, 
        p_drop=0,
    ):
        super(BayesCap_MLP, self).__init__()
        mod = []
        for layer in range(num_layers):
            if layer==0:
                incoming = inp_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
            elif layer==num_layers//2:
                incoming = hid_dim
                outgoing = hid_dim
                mod.append(nn.Linear(incoming, outgoing))
                mod.append(nn.ReLU())
                mod.append(nn.Dropout(p=p_drop))
            elif layer==num_layers-1:
                incoming = hid_dim
                outgoing = out_dim
                mod.append(nn.Linear(incoming, outgoing))
        self.mod = nn.Sequential(*mod)

        self.block_mu = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

        self.block_alpha = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )

        self.block_beta = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            # nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
        )
    
    def forward(self, x):
        x_intr = self.mod(x)
        #print('dbg', x_intr.shape, x.shape)
        x_intr = x_intr + x
        x_mu = self.block_mu(x_intr)
        x_1alpha = self.block_alpha(x_intr)
        x_beta = self.block_beta(x_intr)
        return x_mu, x_1alpha, x_beta

class BayesCLIP(nn.Module):
    def __init__(
        self,
        model_path=None,
        device='cuda',
    ):
        super(BayesCLIP, self).__init__()
        self.clip_model = load_model_p(device, model_path)
        self.clip_model.eval()
        for param in self.clip_model.parameters():
            param.requires_grad = False

        self.img_BayesCap = BayesCap_MLP(inp_dim=1024, out_dim=1024, hid_dim=512, num_layers=3, p_drop=0.3).to(device)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=1024, out_dim=1024, hid_dim=512, num_layers=3, p_drop=0.3).to(device)

    def forward(self, i_inputs, t_inputs):
        i_features, t_features = self.clip_model(i_inputs, t_inputs)

        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta), (i_features, t_features)


class BayesCap_for_CLIP(nn.Module):
    def __init__(
        self,
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3,
        p_drop=0.1,
    ):
        super(BayesCap_for_CLIP, self).__init__()
        self.img_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, p_drop=p_drop)
        self.txt_BayesCap = BayesCap_MLP(inp_dim=inp_dim, out_dim=out_dim, hid_dim=hid_dim, num_layers=num_layers, p_drop=p_drop)

    def forward(self, i_features, t_features):
        
        # print('dbg', i_features.shape, t_features.shape)
        img_mu, img_1alpha, img_beta = self.img_BayesCap(i_features)
        txt_mu, txt_1alpha, txt_beta = self.txt_BayesCap(t_features)

        return (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta)
    
    
def load_data_loader(dataset, data_dir, dataloader_config):
    prepare_loaders = {
        'coco': prepare_coco_dataloaders_extra,
        'flickr': prepare_flickr_dataloaders,
        'CUB':prepare_cub_dataloaders,
        'FLO':prepare_flo_dataloaders
    }[dataset]
    if dataset == 'CUB':
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            caption_root=data_dir+'/text_c10',
            vocab_path='ds/vocabs/cub_vocab.pkl')
    elif dataset == 'FLO':
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            caption_root=data_dir+'/text_c10',)
    else:
        loaders = prepare_loaders(
            dataloader_config,
            dataset_root=data_dir,
            vocab_path='ds/vocabs/coco_vocab.pkl')
    return loaders

def load_model_p(device, model_path=None):
    # load zero-shot CLIP model
    model, _ = clip.load(name='RN50', #change this to ViT-B/32
                         device=device,
                         loss_type='contrastive')
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model

def load_model(device, model_path=None):
    # load zero-shot CLIP model
    model, _ = clip.load('ViT-B/32',
                         device=device, jit=False)
    if model_path is None:
        # Convert the dtype of parameters from float16 to float32
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    else:
        ckpt = torch.load(model_path)
        model.load_state_dict(ckpt['state_dict'])
        for name, param in model.named_parameters():
            param.data = param.data.type(torch.float32)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


from scipy.special import gammaincinv, gamma
import numpy as np

### This is a new implementation of sample_ggd, not in the original paper

def gamma_inverse_cdf_vectorized(p, shape, scale=1.0):
    """
    Vectorized inverse CDF for gamma distribution.
    
    Args:
        p: Tensor of shape (..., D)
        shape: Tensor of shape (D,) or broadcastable to p
        scale: Scalar or tensor broadcastable to p
    Returns:
        Tensor of same shape as p
    """
    eps = 1e-7
    p = torch.clamp(p, min=eps)
    p_cpu = p.detach().cpu().numpy()
    shape_cpu = shape.detach().cpu().numpy()
    gamma_shape = gamma(shape_cpu)
    p_gamma = np.clip(p_cpu * gamma_shape, eps, 1 - eps)
    result = gammaincinv(shape_cpu, p_gamma) / scale
    return torch.from_numpy(result).to(p.device).to(p.dtype)

def sample_ggd(mu, alpha, beta, num_samples=100):
    """
    Vectorized GGD sampling per feature dimension.
    
    Args:
        mu: Tensor of shape [D]
        alpha: Tensor of shape [D]
        beta: Tensor of shape [D]
        num_samples: Number of samples to draw per dimension
    
    Returns:
        Tensor of shape [num_samples, D]
    """
    D = mu.shape[0]
    shape = (num_samples, D)

    p = torch.rand(shape, device=mu.device)
    sign = torch.sign(p - 0.5)
    abs_p = 2 * torch.abs(p - 0.5)

    inv_gamma = gamma_inverse_cdf_vectorized(abs_p, 1.0 / beta)
    samples = sign * alpha * inv_gamma.pow(1.0 / beta) + mu

    return samples

def sample_ggd_wrapper(mu, alpha, beta, num_samples=100):
    mu = mu.squeeze(0)
    alpha = alpha.squeeze(0)
    beta = beta.squeeze(0)

    samples, = sample_ggd(mu, alpha, beta, num_samples=num_samples),
    return samples.unsqueeze(1)






def sample_ggd_old(x_mu, x_1alpha, x_beta, num_samples=100):
    
    """
    
    Fucntion needs to be modified so that we can draw a sample from the distri
    Sample from a GGD with parameters (mu, alpha, beta).
    
    Args:
        x_mu: Tensor, the location parameter (mean).
        x_1alpha: Tensor, the scale parameter.
        x_beta: Tensor, the shape parameter.
        num_samples: int, number of samples to draw.
        
    Returns:
        feature_vector: Tensor, derived feature vector from GGD samples.
        
        
    """
    # Add a small epsilon to x_1alpha to avoid zero values
    epsilon = 1e-6
    x_1alpha_adjusted = x_1alpha + epsilon

    # Create an approximate normal distribution
    ggd_dist = dist.Normal(x_mu, x_1alpha_adjusted)

    # Sample and compute feature vector (e.g., mean of samples)
    samples = ggd_dist.sample((num_samples,))

    return samples

def sample_ggd_torch(mu, alpha, beta, num_samples=100, eps=1e-6):
    """
    Differentiable PyTorch-based GGD sampling (approximate).
    
    Args:
        mu: [D]
        alpha: [D]
        beta: [D]
        num_samples: int

    Returns:
        Tensor: [num_samples, D]
    """
    D = mu.shape[0]
    shape = (num_samples, D)

    # Sample from uniform
    u = torch.rand(shape, device=mu.device)
    sign = torch.sign(u - 0.5)
    abs_u = torch.abs(u - 0.5) * 2  # Uniform in (0, 1)

    # Sample from Gamma distribution instead of using icdf
    gamma_shape = (1.0 / beta).expand(num_samples, -1)
    gamma_scale = torch.ones_like(gamma_shape)
    gamma_dist = dist.Gamma(gamma_shape, gamma_scale)
    gamma_samples = gamma_dist.sample()

    samples = sign * alpha * gamma_samples.pow(1.0 / beta) + mu
    samples = samples.unsqueeze(1) 
    return samples
import pickle
import os

def save_data_loaders(loaders, filename):
    with open(filename, 'wb') as f:
        pickle.dump(loaders, f)

def load_data_loaders(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    return None

# Usage
dataset = 'coco'  # coco or flickr
data_dir = ospj('/mimer/NOBACKUP/groups/ulio_inverse/ds-project/ProbVLM/Datasets/', dataset)
dataloader_config = mch({
    'batch_size': 64,
    'random_erasing_prob': 0.,
    'traindata_shuffle': True
})

filename = '/mimer/NOBACKUP/groups/ulio_inverse/ds-project/ProbVLM/Datasets/coco/data_loaders_coco_person_extra_26.11.pkl'
loaders = load_data_loaders(filename)

if loaders is None:
    loaders = load_data_loader(dataset, data_dir, dataloader_config)
    save_data_loaders(loaders, filename)

coco_train_loader, coco_valid_loader, coco_test_loader = loaders['train'], loaders['val'], loaders['test']


device='cuda'
CLIP_Net = load_model_p(device=device, model_path=None)
ProbVLM_Net = BayesCap_for_CLIP(inp_dim=1024,
        out_dim=1024,
        hid_dim=512,
        num_layers=3,
        p_drop=0.1,
    )

# When doing this for bayesVLM, maybe change this
ProbVLM_Net = ProbVLM_Net.to(device)
ProbVLM_Net.load_state_dict(checkpoint)
ProbVLM_Net.eval()

from PIL import Image
import matplotlib.pyplot as plt
import torch
text_list = ["an image of a person", "a photo of a person"]
from grad_cam import GradCAM
# import torchray
# from torchray.attribution.grad_cam import grad_cam as tr_gradcam
import attention_utils_p as au

class AttentionVLModel(nn.Module):
    def __init__(self, base_model, gradcam_layer='layer4.2.relu'):
        super(AttentionVLModel, self).__init__()
        self.base_model = base_model.module if hasattr(base_model, "module") else base_model
        self.gradcam = GradCAM(model=self.base_model, candidate_layers=[gradcam_layer])
    
    def forward(self, image_path, img_f, text_f, text_list, tok_t, tok_i, device):

        # Generate attention map
### Maybe change this here to transformer attention for ViT-B/32 based model ie au.transfomer_attention ### 
        attention_data = au.clip_gcam_prob(
            model=self.base_model,
            file_path= image_path,
            layer = self.gradcam.candidate_layers[0],
            text_list=text_list,
            img_f = img_f,
            text_f = text_f,
            tokenized_text = tok_t,
            tokenized_img = tok_i,
            device=device,
            plot_vis=False,
            save_vis_path = False,
            resize = False
        )

        # Extract relevant outputs
        attentions, probs, unnorm_attentions, text = attention_data['attentions'], attention_data['probs'], attention_data['unnormalized_attentions'], attention_data['text_list']
        return attentions, probs, unnorm_attentions, text
    
attention_model = AttentionVLModel(base_model=CLIP_Net).to(device)

def token_to_text(coded_text):
    
    # Convert indices to tokens
    tokens = [tokenizer.convert_ids_to_tokens(indices.tolist()) for indices in coded_text]
    
    # Define a list of unwanted tokens
    unwanted_tokens = {'<|startoftext|>', '<|endoftext|>', '.', '!', '</w>'}

    # Filter the tokens to exclude unwanted ones and keep only the actual words
    filtered_words = [token[:-4] if token.endswith('</w>') else token for token in tokens if token not in unwanted_tokens]

    # Convert the list of words into a single string
    result_string = ' '.join(filtered_words)
    
    return [result_string]


def process_attention_maps(data_loader, save_folder, aggregation_method, max_files=15):
    # Define root and save paths
### This will maybe need to be changed to reflect ViT-B/32 backbone
    ROOT = "/mimer/NOBACKUP/groups/ulio_inverse/ds-project/ProbVLM/Datasets/coco/images"
    vis_root = os.path.join(ROOT, f"clip_RN-50_6_attention_gradcam_{aggregation_method}")
    base = os.path.join(vis_root, 'data/COCO')
    SAVE_PATH = os.path.join(base, save_folder)
    #os.makedirs(SAVE_PATH, exist_ok=True)    

    print(f"Starting to save {save_folder} with {aggregation_method} aggregation method!", flush=True)
      
    num_visualized = 0

    for i, batch in enumerate(data_loader):
        xI, xT, paths = batch[0].to(device), batch[1].to(device), batch[4]
        
        for t, (img, txt, path) in enumerate(zip(xI, xT, paths)):
            text_list = token_to_text(xT[t])

            # Filter by text content
            if text_list[0] not in {'a photo of a person', 'an image of a person'}:
                continue

            # Prepare inputs
            img = img.unsqueeze(0).to(device)  # Add batch dimension and move to device
            txt = txt.unsqueeze(0).to(device)
            #Change to BayesVLM here when doing bayes
            with torch.no_grad():
                xfI, xfT = CLIP_Net(img, txt)
                (img_mu, img_1alpha, img_beta), (txt_mu, txt_1alpha, txt_beta) = ProbVLM_Net(xfI, xfT)
                img_samples = sample_ggd_old(img_mu, img_1alpha, img_beta, 50)
                txt_samples = sample_ggd_old(txt_mu, txt_1alpha, txt_beta, 50)
            
            
            attentions_list = []
            unnormalized_attentions_list = []
            probss = []

            for img_feature_vector, txt_feature_vector in zip(img_samples, txt_samples):
                attentions, probs, unnorm_attentions, text_list = attention_model.forward(
                    image_path=path,
                    img_f=img_feature_vector.to(device),
                    text_f=txt_feature_vector.detach(),
                    text_list=text_list,
                    tok_t=txt,
                    tok_i=img,
                    device=device
                )
                attentions_list.append(attentions)
                unnormalized_attentions_list.append(unnorm_attentions)
                probss.append(probs)

            # Aggregate attentions
            if aggregation_method == 'median':
                aggregated_attention = torch.median(torch.stack(attentions_list), dim=0).values
                aggregated_unnorm_attention = torch.median(torch.stack(unnormalized_attentions_list), dim=0).values
                aggregated_probs = np.median(probss, axis=0)
            elif aggregation_method == 'mean':
                aggregated_attention = torch.mean(torch.stack(attentions_list), dim=0)
                aggregated_unnorm_attention = torch.mean(torch.stack(unnormalized_attentions_list), dim=0)
                aggregated_probs = np.mean(probss, axis=0)
            else:
                raise ValueError("Aggregation method must be 'mean' or 'median'")

            # Save .pth file 
            attention_save_path = os.path.join(SAVE_PATH, f"{os.path.basename(path).replace('.jpg', '.pth')}")
            os.makedirs(os.path.dirname(attention_save_path), exist_ok=True)
            torch.save({
                'attentions': aggregated_attention,
                'unnormalized_attentions': aggregated_unnorm_attention,
                'probs': aggregated_probs,
                'text_list': text_list
            }, attention_save_path)

            # Save visualization
            # os.makedirs(os.path.join(SAVE_PATH, 'vis'), exist_ok=True)
            save_vis_path = os.path.join(vis_root, 'vis', os.path.basename(path).replace('.jpg', '.jpg'))
            os.makedirs(os.path.dirname(save_vis_path), exist_ok=True)
            if num_visualized < max_files and i % 50 == 0:
                au.plot_attention_helper_p(
                    image=img,
                    attentions=[aggregated_attention],
                    unnormalized_attentions=[aggregated_unnorm_attention],
                    probs=[aggregated_probs],
                    text_list=text_list,
                    save_vis_path=save_vis_path,
                    resize=False
                )
                num_visualized += 1
    # Done statement after processing all batches in the data_loader
    print("Done!", flush=True)

process_attention_maps(coco_valid_loader, "val2014", "mean")
process_attention_maps(coco_valid_loader, "val2014", "median")
process_attention_maps(coco_train_loader, "train2014", "mean")
process_attention_maps(coco_train_loader, "train2014", "median") 
