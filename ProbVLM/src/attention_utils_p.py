import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from PIL import Image
from torchvision import transforms

from torchray.attribution.grad_cam import grad_cam as tr_gradcam

import matplotlib
matplotlib.rcParams.update({'font.size': 18})
# import cv2
import matplotlib.pyplot as plt

from grad_cam import GradCAM

"""
Credit for transformer attention:
Generic Attention-model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers
Hila Chefer, Shir Gur, Lior Wolf

https://github.com/hila-chefer/Transformer-MM-Explainability
"""


def transformer_attention_prob(model, file_path , text_list, img_f, text_f, tokenized_text ,tokenized_img, device, layer,
                               plot_vis=False, save_vis_path=None, resize=False):
    image = tokenized_img
    if img_f.dim() == 1:
        img_f = img_f.unsqueeze(0)
    if text_f.dim() == 1:
        text_f = text_f.unsqueeze(0)
    image_features_norm = img_f.norm(dim=-1, keepdim=True)
    image_features_new = img_f / image_features_norm
    text_features_norm = text_f.norm(dim=-1, keepdim=True)
    text_features_new = text_f / text_features_norm
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features_new @ text_features_new.t()
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    # Project sampled embedding to match transformer's expected dimension if needed
    if img_f.shape[-1] != model.visual.transformer.width:
        projector = torch.nn.Linear(img_f.shape[-1], model.visual.transformer.width).to(device)
        img_f = projector(img_f)
    # After normalizing img_f, reprocess sampled embedding through attention blocks
    _, num_tokens, image_attn_blocks = reprocess_img_embedding_through_attention(model, img_f, device)
    attentions = []
    unnormalized_attentions = []
    text_prediction = (text_features_new[[0]] * image_features_norm)
    # Safely get dtype from first available attention block
    attn_dtype = next((blk.attn_probs.dtype for blk in image_attn_blocks if blk.attn_probs is not None), None)

    if attn_dtype is None:
        print("[WARN] No attn_probs were captured — skipping attention computation for this sample.")
        return {
            'unnormalized_attentions': torch.tensor([]),
            'attentions': torch.tensor([]),
            'text_list': text_list,
            'probs': np.array([]),
        }
    R = torch.eye(num_tokens, num_tokens, dtype=attn_dtype).to(device)
    for blk in image_attn_blocks:
        print(f"Checking block: {blk.__class__.__name__}, id={id(blk)}")
        if blk.attn_grad is None or blk.attn_probs is None:
            print('blk.attn_grad or blk.attn_probs is None')
            continue  # Skip blocks without gradient info

        grad = blk.attn_grad
        cam = blk.attn_probs
        cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
        grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
        cam = grad * cam
        cam = cam.clamp(min=0).mean(dim=0)
        R += torch.matmul(cam, R)
    R[0, 0] = 0
    image_relevance = R[0, 1:]

    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = image_relevance.detach().type(torch.float32).cpu()
    if resize:
        image_relevance = F.interpolate(
            image_relevance, image.shape[2:], mode="bilinear", align_corners=False
        )
    unnormalized_attentions.append(image_relevance)
    image_relevance_norm = normalize(image_relevance)
    attentions.append(image_relevance_norm)

    if plot_vis:
        plot_attention_helper_p(image, attentions, unnormalized_attentions, probs, ['prob'],
                                save_vis_path=save_vis_path, resize=resize)

    return {
        'unnormalized_attentions': torch.cat(unnormalized_attentions),
        'attentions': torch.cat(attentions),
        'text_list': ['prob'],
        'probs': probs
    }


# New function: transformer_attention_from_embeddings
def transformer_attention_from_embeddings(model, img_f, text_f, text_list, tokenized_text, tokenized_img ,device,
                                         plot_vis=False, save_vis_path=None, resize=False):
    """
    Transformer-based attention using precomputed image and text embeddings.
    Based on: https://github.com/hila-chefer/Transformer-MM-Explainability
    """
    import torch.nn.functional as F
    import numpy as np
    from attention_utils_p import normalize, plot_attention_helper_p
    image = tokenized_img  # preprocessed image tensor

    # Ensure inputs are batched
    if img_f.dim() == 1:
        img_f = img_f.unsqueeze(0)
    if text_f.dim() == 1:
        text_f = text_f.unsqueeze(0)
    # Normalize features as CLIP does
    image_features_norm = img_f.norm(dim=-1, keepdim=True)
    image_features_new = img_f / image_features_norm
    text_features_norm = text_f.norm(dim=-1, keepdim=True)
    text_features_new = text_f / text_features_norm
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * img_f @ text_f.T
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    attentions = []
    unnormalized_attentions = []

    for idx in range(len(text_list)):
        print("image_features_new.shape:", image_features_new.shape)
        print("text_features_new.shape:", text_features_new.shape)
        print("logits_per_image.shape:", logits_per_image.shape)
        one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
        one_hot[0, idx] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(device)
        one_hot_score = torch.sum(one_hot * logits_per_image)
        model.zero_grad()
        one_hot_score.backward(retain_graph=True)

        image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        for blk in image_attn_blocks:
            if blk.attn_grad is None or blk.attn_probs is None:
                print('blk.attn_grad or blk.attn_probs is None')
                continue
            grad = blk.attn_grad
            cam = blk.attn_probs
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R += torch.matmul(cam, R)
        R[0, 0] = 0
        image_relevance = R[0, 1:]

        image_relevance = image_relevance.reshape(1, 1, 7, 7)
        image_relevance = image_relevance.detach().type(torch.float32).cpu()
        if resize:
            image_relevance = F.interpolate(image_relevance, image.shape[2:], mode="bilinear", align_corners=False)
        unnormalized_attentions.append(image_relevance)
        attentions.append(normalize(image_relevance))

    if plot_vis:
        plot_attention_helper_p(
            image, attentions, unnormalized_attentions, probs, text_list,
            save_vis_path=save_vis_path, resize=resize
        )

    return {
        'unnormalized_attentions': torch.cat(unnormalized_attentions),
        'attentions': torch.cat(attentions),
        'text_list': text_list,
        'probs': probs
    }













def transformer_attention(model, preprocess, file_path, text_list, tokenized_text, device,
                          plot_vis=False, save_vis_path=None, resize=False):
    """
    Credit to: https://github.com/hila-chefer/Transformer-MM-Explainability
    """
    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

    logits_per_image, _ = model(image, tokenized_text)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    attentions = []
    unnormalized_attentions = []
    for idx in range(len(text_list)):
        one_hot = np.zeros((1, logits_per_image.size()[-1]), dtype=np.float32)
        one_hot[0, idx] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        one_hot = torch.sum(one_hot.to(device) * logits_per_image)
        model.zero_grad()
        one_hot.backward(retain_graph=True)

        image_attn_blocks = list(
            dict(model.visual.transformer.resblocks.named_children()).values()
        )
        num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
        R = torch.eye(num_tokens, num_tokens,
                      dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
        for blk in image_attn_blocks:
            grad = blk.attn_grad
            cam = blk.attn_probs
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.clamp(min=0).mean(dim=0)
            R += torch.matmul(cam, R)
        R[0, 0] = 0
        image_relevance = R[0, 1:]

        image_relevance = image_relevance.reshape(1, 1, 7, 7)
        image_relevance = image_relevance.detach().type(torch.float32).cpu()
        if resize:
           image_relevance  = F.interpolate(
               image_relevance, image.shape[2:], mode="bilinear", align_corners=False
            )
        unnormalized_attentions.append(image_relevance)
        image_relevance = normalize(image_relevance)
        attentions.append(image_relevance)

    if plot_vis:
        plot_attention_helper(
            image, attentions, unnormalized_attentions, probs, text_list,
            save_vis_path=save_vis_path, resize=resize
        )

    return {
        'unnormalized_attentions': torch.cat(unnormalized_attentions),
        'attentions': torch.cat(attentions),
        'text_list': text_list,
        'probs': probs
    }


def clip_gcam(model, preprocess, file_path, text_list, tokenized_text, layer, device,
              plot_vis=False, save_vis_path=None, resize=False):

    image = preprocess(Image.open(file_path)).unsqueeze(0).to(device)

    image_features = model.encode_image(image)
    text_features = model.encode_text(tokenized_text).detach()
    image_features_norm = image_features.norm(dim=-1, keepdim=True)
    image_features_new = image_features / image_features_norm
    text_features_norm = text_features.norm(dim=-1, keepdim=True)
    text_features_new = text_features / text_features_norm
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features_new @ text_features_new.t()
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    # Each shape 1 x 1 x H x W
    attentions = []
    unnormalized_attentions = []
    for i in range(len(text_list)):
        # mutliply the normalized text embedding with image norm to get approx image embedding
        text_prediction = (text_features_new[[i]] * image_features_norm)
        saliency = tr_gradcam(model.visual, image.type(model.dtype), text_prediction, saliency_layer=layer)
        saliency = saliency.detach().type(torch.float32).cpu()
        if resize:
            saliency = F.interpolate(
                saliency, image.shape[2:], mode="bilinear", align_corners=False
            )
        unnormalized_attentions.append(saliency)
        sal = normalize(saliency)
        attentions.append(saliency)

    if plot_vis:
        plot_attention_helper(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=save_vis_path, resize=resize)

    return {
        'unnormalized_attentions': torch.cat(unnormalized_attentions),
        'attentions': torch.cat(attentions),
        'text_list': text_list,
        'probs': probs
    }


def clip_gcam_prob(model, file_path, text_list, img_f, text_f, tokenized_text, tokenized_img, layer, device,
              plot_vis=False, save_vis_path=None, resize=False):

    image = tokenized_img

    #image_features = model.encode_image(image)
    #text_features = model.encode_text(tokenized_text).detach()
    image_features_norm = img_f.norm(dim=-1, keepdim=True)
    image_features_new = img_f / image_features_norm
    text_features_norm = text_f.norm(dim=-1, keepdim=True)
    text_features_new = text_f / text_features_norm
    #     print('text_new_features',text_features_new[[0]])
    #     print('text_new_features',text_features_new)
    logit_scale = model.logit_scale.exp()
    logits_per_image = logit_scale * image_features_new @ text_features_new.t()
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()

    
    #print(f"text_features_new shape: {text_features_new.shape}")
    #print(f"image_features_norm shape: {image_features_norm.shape}")
    
    
    if text_features_new.numel() == 0:
        print("text_features_new is empty.")
    if image_features_norm.numel() == 0:
        print("image_features_norm is empty.")
        
    
    # Each shape 1 x 1 x H x W
    attentions = []
    unnormalized_attentions = []
    #for i in range(len(text_list)):
        # mutliply the normalized text embedding with image norm to get approx image embedding
    #print(f"text_prediction shape before multiplication: {text_features_new[[0]].shape}")
    #print(f"image_features_norm shape: {image_features_norm.shape}")

    text_prediction = (text_features_new[[0]] * image_features_norm)
    saliency = tr_gradcam(model.visual, image.type(model.dtype), text_prediction, saliency_layer=layer)
    saliency = saliency.detach().type(torch.float32).cpu()
    if resize:
        saliency = F.interpolate(
            saliency, image.shape[2:], mode="bilinear", align_corners=False
        )
    unnormalized_attentions.append(saliency)
    sal = normalize(saliency)
    attentions.append(saliency)

    if plot_vis:
        plot_attention_helper_p(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=save_vis_path, resize=resize)

    return {
        'unnormalized_attentions': torch.cat(unnormalized_attentions),
        'attentions': torch.cat(attentions),
        'text_list': text_list,
        'probs': probs
    }





def normalize(sal):
    # sal = tensor of shape 1,1,H,W
    B, C, H, W = sal.shape
    sal = sal.view(B, -1)
    sal_max = sal.max(dim=1, keepdim=True)[0]
    sal_max[torch.where(sal_max == 0)] = 1. # prevent divide by 0
    sal -= sal.min(dim=1, keepdim=True)[0]
    sal /= sal_max
    sal = sal.view(B, C, H, W)
    return sal

import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image

def show_cam_on_image(img, attention):

    denom = (attention.max() - attention.min())
    if denom > 0:
        attention = (attention - attention.min()) / (attention.max() - attention.min())
    else:
        attention = torch.zeros_like(torch.from_numpy(attention)) if isinstance(attention, np.ndarray) else torch.zeros_like(attention)
        attention = attention.numpy()
    colormap = cm.get_cmap('jet')
    heatmap = colormap(attention)[:, :, :3]  # Drop alpha channel if present
    heatmap = np.float32(heatmap)
    if img.dtype != np.float32:
        img = np.float32(img) / 255.0

    if heatmap.shape[:2] != img.shape[:2]:
        heatmap = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((img.shape[1], img.shape[0]), Image.BILINEAR)) / 255.0

    cam = heatmap + img
    cam = cam / np.max(cam)
    
    return cam



# def show_cam_on_image(img, attention):
#     heatmap = cv2.applyColorMap(np.uint8(255 * attention), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#     return cam


def plot_attention_helper(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=None, resize=False):
    image_vis = image[0].permute(1, 2, 0).data.cpu().numpy()
    image_vis = (image_vis - image_vis.min()) / (image_vis.max() - image_vis.min())
    attention_vis = []
    for i in range(len(attentions)):
        if not resize:
            sal = F.interpolate(
                unnormalized_attentions[i],
                image.shape[2:], mode="bilinear", align_corners=False
            )
            sal = normalize(sal)[0][0]
        else:
            sal = attentions[i][0][0]
        vis = show_cam_on_image(image_vis, sal.numpy())
        vis = np.uint8(255 * vis)
        attention_vis.append(vis)
    plot(probs, attention_vis, text_list, image_vis, save_path=save_vis_path)

    
def plot_attention_helper_squeeze(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=None, resize=False):
    image_vis = image[0].permute(1, 2, 0).data.cpu().numpy()
    image_vis = (image_vis - image_vis.min()) / (image_vis.max() - image_vis.min())
    attention_vis = []
    for i in range(len(attentions)):
        if not resize:
            sal = F.interpolate(
                unnormalized_attentions[i].unsqueeze(1),
                image.shape[2:], mode="bilinear", align_corners=False
            )
            sal = normalize(sal)[0][0]
        else:
            sal = attentions[i][0][0]
        vis = show_cam_on_image(image_vis, sal.numpy())
        vis = np.uint8(255 * vis)
        attention_vis.append(vis)
    plot(probs, attention_vis, text_list, image_vis, save_path=save_vis_path)
    

def plot_attention_helper_p(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=None, resize=False):
    image_vis = image[0].permute(1, 2, 0).data.cpu().numpy()
    image_vis = (image_vis - image_vis.min()) / (image_vis.max() - image_vis.min())
    
    if not resize:
        sal = F.interpolate(
            unnormalized_attentions[0],
            image.shape[2:], mode="bilinear", align_corners=False
        )
        sal = normalize(sal)[0][0]
    else:
        sal = attentions[0][0]
    vis = show_cam_on_image(image_vis, sal.numpy())
    vis = np.uint8(255 * vis)
    #attention_vis.append(vis)
    plot_p(probs, [vis], text_list, image_vis, save_path=save_vis_path)


def plot(probs, attention_vis, text_list, image_vis, save_path=None):
    sort = np.argsort(probs)[0][::-1]
    attention_vis = np.array(attention_vis)[sort]
    probs_vis = probs[0][sort]
    text_list_vis = np.array(text_list)[sort]

    fig, ax = plt.subplots(1,1+len(text_list),figsize=(5*(1+len(text_list)),6))
    ax[0].imshow(image_vis)
    ax[0].axis('off')

    for idx in range(len(text_list)):
        ax[idx+1].imshow(attention_vis[idx])
        ax[idx+1].axis('off')
        split = text_list_vis[idx].split(' ')
        if len(split) > 6:
            prompt = ' '.join(split[:6]) + '\n ' + ' '.join(split[6:])
        else:
            prompt = text_list_vis[idx]
        ax[idx+1].set_title('{}\n{:.3f}'.format(prompt, probs_vis[idx], 2),
                           fontsize=14)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    return

def plot_p(probs, attention_vis, text_list, image_vis, save_path=None):

    fig, ax = plt.subplots(1,1+len(text_list),figsize=(5*(1+len(text_list)),6))
    # OG image
    ax[0].imshow(image_vis)
    ax[0].axis('off')
    # Saliency map
    ax[1].imshow(attention_vis[0])
    ax[1].axis('off')
    
    ax[1].set_title('{}\n{:.3f}'.format(text_list[0], float(probs[0])),
                       fontsize=14)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    else:
        plt.show()
    return


def parse_attention(attentions, shape, mode='average_nonzero'):
    """
    Chooses b/w different modes of combining attentions.
    attentions: torch tensor of shape B x num_attentions_per_sample x 1 x H x W
    shape: tuple of desired attentions shape to resize to if needed

    mode options:
    - "average_nonzero": average only attentions that are not all zeros.
        If all attentions are nonzero, entry in list is None
    - "max"

    Returns tensor of shape num_nonzero_attentions x 1 x H x W,
    and list of length B which is 1 if attentions[i] was not all 0, and 0 if it was.
    """
    final_attentions = []
    inds = []
    for idx in range(attentions.shape[0]):
        # bool tensor of which attentions are not all zero
        nonzero_atts = torch.stack([~torch.all(att == 0) for att in attentions[idx]])
        if torch.all(nonzero_atts.float() == 0):
            inds.append(0)
        else:
            valid = attentions[idx][nonzero_atts]
            if mode == 'average_nonzero':
                att = valid.mean(dim=0)
            elif mode == 'max':
                att = torch.max(valid, dim=0, keepdim=True).values[0]
            else:
                print('ERROR: attention parsing mode {} not implemented'.format(mode))
                raise NotImplementedError
            if len(att.shape) == 2:
                att = att.unsqueeze(0).unsqueeze(0)
            elif len(att.shape) == 3:
                att = att.unsqueeze(0)
            else:
                print(att.shape)
                raise NotImplementedError
            normalized = normalize(att)
            normalized = normalized[0][0].unsqueeze(0)
            if normalized.shape[-2:] != shape:
                normalized = transforms.functional.resize(normalized, shape)
            final_attentions.append(normalized)
            inds.append(1)
    if len(final_attentions) > 0:
        final_attentions = torch.stack(final_attentions)
    return final_attentions, torch.Tensor(inds)


def compute_gradcam(fmaps, logits, labels, device, resize=False, resize_shape=None):
    if logits.shape[1] == 1:
        probs = nn.Sigmoid()(logits)
        class1_probs = 1 - probs
        y = torch.cat((class1_probs, probs), dim=1)
    else:
        y = logits
    ids     = torch.LongTensor([labels.tolist()]).T.to(device)
    one_hot = torch.zeros_like(y).to(device)
    one_hot.scatter_(1, ids, 1.0)

    grads = torch.autograd.grad(
        y,
        fmaps,
        grad_outputs=one_hot,
        retain_graph=True,
        create_graph=True
    )[0]

    weights = F.adaptive_avg_pool2d(grads, 1)

    gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
    gcam = F.relu(gcam)

    if resize:
        assert resize_shape is not None
        gcam = F.interpolate(
            gcam, resize_shape, mode="bilinear", align_corners=False
        )

    B, C, H, W = gcam.shape
    gcam = gcam.view(B, -1)
    gcam_max = gcam.max(dim=1, keepdim=True)[0]
    gcam_max[torch.where(gcam_max == 0)] = 1. # prevent divide by 0
    gcam -= gcam.min(dim=1, keepdim=True)[0]
    gcam /= gcam_max
    gcam = gcam.view(B, C, H, W)
    return gcam




# Utility: Reprocess image embedding through attention blocks to collect gradients
def reprocess_img_embedding_through_attention(model, img_f, device):
    """
    Re-process a sampled image embedding through the model's attention layers
    so that attention gradients can be collected.
    Assumes img_f is a 1D or 2D tensor (batch_size x dim).
    """
    if img_f.dim() == 1:
        img_f = img_f.unsqueeze(0)  # Add batch dim if needed

    # Fake token embedding map to pass through attention blocks
    # We'll simulate a [CLS] + patch_tokens shape: (1, num_tokens, embed_dim)
    # For ViT-B/32 this is typically (1, 50, 512) but we adapt based on resblock shapes
    example_block = next(iter(model.visual.transformer.resblocks.children()))
    num_tokens = example_block.attn_probs.shape[-1] if hasattr(example_block, "attn_probs") else 50
    embed_dim = img_f.shape[-1]
    # Ensure embedding dim matches transformer's expected width
    expected_embed_dim = model.visual.transformer.width
    assert embed_dim == expected_embed_dim, f"Embedding dim {embed_dim} doesn't match expected {expected_embed_dim}"
    fake_tokens = img_f.unsqueeze(1).repeat(1, num_tokens, 1).clone().detach()
    fake_tokens.requires_grad_(True)
    # Transformer expects input as (seq_len, batch_size, embed_dim)
    fake_tokens = fake_tokens.permute(1, 0, 2).contiguous()

    x = fake_tokens
    for blk in model.visual.transformer.resblocks:
        blk.attn_grad = None
        blk.attn_probs = None
        if hasattr(blk.attn, "set_attn_probs") and hasattr(blk.attn, "set_attn_grad"):
            out, _ = blk.attn(
                x, x, x,
                need_weights=False,
                attn_mask=blk.attn_mask,
                attention_probs_forward_hook=blk.attn.set_attn_probs,
                attention_probs_backwards_hook=blk.attn.set_attn_grad
            )
        else:
            out = blk.attn(x, x, x)[0]
        x = x + out  # skip connection
        x = x + blk.mlp(blk.ln_2(x))  # final transformer step

    # Return also the actual resblock list so that the same objects are accessible elsewhere
    return fake_tokens, num_tokens, list(model.visual.transformer.resblocks)