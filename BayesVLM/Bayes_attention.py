import torch
import clip
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from pathlib import Path
import os
from tqdm import tqdm
from torch.nn import functional as F

import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


### Functions from Attention_maps_COCO...
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



### Functions from attention_utils_p
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

def plot_attention_helper_p(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=None, resize=False):
    image_vis = image[0].permute(1, 2, 0).data.cpu().numpy()
    image_vis = (image_vis - image_vis.min()) / (image_vis.max() - image_vis.min())
    
    if not resize:
        sal = unnorm_attentions[0]
        if sal.dim() == 2:
            sal = sal.unsqueeze(0).unsqueeze(0)
        elif sal.dim() == 3:
            sal = sal.unsqueeze(0)
        sal = F.interpolate(
            sal,
            image.shape[2:], mode="bilinear", align_corners=False
        )
        sal = normalize(sal)[0][0]
    else:
        sal = attentions[0][0]
    vis = show_cam_on_image(image_vis, sal.cpu().numpy())
    vis = np.uint8(255 * vis)
    #attention_vis.append(vis)
    plot_p(probs, [vis], text_list, image_vis, save_path=save_vis_path)

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



# Set number of workers for DataLoader


# Cache for BayesVLM results
_bayes_cache = {}

def get_cached_bayes_results(image_path, hessian_dir):
    cache_key = f"{image_path}_{hessian_dir}"
    if cache_key not in _bayes_cache:
        from get_probcosine import get_probcosine_distribution
        _bayes_cache[cache_key] = get_probcosine_distribution(image_path, str(hessian_dir))
    return _bayes_cache[cache_key]

@torch.no_grad()
def adapter(embedding, num_samples=100, bayes_results=None, image_path=None):
    current_dir = Path(__file__).parent.absolute()
    hessian_dir = current_dir / "BayesVLM-main" / "hessians" / "hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"
    
    # Print the path when getting results
    print(f"Using hessian directory: {hessian_dir}")
    
    try:
        if bayes_results is None:
            bayes_results = get_cached_bayes_results(image_path, str(hessian_dir))
        
        # Get BayesVLM parameters
        bayes_variance = bayes_results['variance'].item()
        kappa = bayes_results['kappa'].item()
        probas = bayes_results['probabilities'].squeeze()
        mean = bayes_results['mean'].squeeze()
        
        # Generate samples using BayesVLM distribution
        D = embedding.shape[-1]
        noise = torch.randn((num_samples, D), device=embedding.device) * torch.sqrt(bayes_variance)
        
        # Weight noise by probability distribution
        noise = noise * probas.unsqueeze(-1)
        
        # Apply kappa scaling and add to embedding
        samples = embedding + noise * kappa
        
        # Add influence from BayesVLM mean
        samples = samples + 0.1 * mean  # Small influence from BayesVLM mean
        
        # Normalize samples
        samples = torch.nn.functional.normalize(samples, p=2, dim=-1)
        return samples
    except Exception as e:
        # print(f"Error in adapter: {str(e)}")
        # Fallback behavior
        D = embedding.shape[-1]
        noise = torch.randn((num_samples, D), device=embedding.device) * 0.1
        samples = embedding + noise
        samples = torch.nn.functional.normalize(samples, p=2, dim=-1)
        return samples

def RISE_with_Bayes(model, file_path, text_list, img_f, text_f, tokenized_text, tokenized_img, layer, device,
         plot_vis=False, save_vis_path=None, resize=False):
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Clear the cache
    global _bayes_cache
    _bayes_cache = {}
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Load and preprocess image
    if file_path is not None:
        image_path = file_path
    else:
        image_path = '/Users/Edvard/Desktop/Kandidatarbete/BayesVLM/Cardinal_0012_18638.jpg'
    
    print(f"Processing image: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    image = image.resize((224, 224))
    if tokenized_text is not None:
        # text = tokenized_text
        text = clip.tokenize(["an image of a bird"]).to(device)

    else:
        text = clip.tokenize(["an image of a bird"]).to(device)

    # Get BayesVLM results - Fix the hessian directory path
    current_dir = Path(__file__).parent.absolute()
    # Change her to alvis directory
    hessian_dir = '/cephyr/users/schmidte/Alvis/Paric_nolavis/BayesVLM/hessians/hessian_CLIP-ViT-B-32-laion2B-s34B-b79K'
    
    # Print the path to verify
    print(f"Looking for hessian files in: {hessian_dir}")
    
    # if not hessian_dir.exists():
    #     raise FileNotFoundError(f"Hessian directory not found at {hessian_dir}")
    
    # Check if required files exist
    required_files = ['A_img_analytic.pt', 'B_img_analytic.pt', 'A_txt_analytic.pt', 'B_txt_analytic.pt', 'prior_precision_analytic.json']
    # for file in required_files:
        # if not (hessian_dir / file).exists():
        #     raise FileNotFoundError(f"Required file {file} not found in {hessian_dir}")
    
    bayes_results = get_cached_bayes_results(image_path, str(hessian_dir))

    # Preprocess
    image_tensor = preprocess(image).unsqueeze(0).to(device)

    # Encode text once
    with torch.no_grad():
        text_embedding = model.encode_text(text)
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=-1)

    # Create a wrapper for the adapter that includes the image path
    def adapter_with_path(embedding, num_samples=50, bayes_results=None):
        return adapter(embedding, num_samples, bayes_results, image_path=image_path)

    # RISE with BayesVLM-informed Adapter
    from RISEWithAdapterBatch import RISEWithAdapter
    from RISE import RISE

    # Generate RISE masks
    input_size = (224, 224)
    rise = RISE(model, input_size)

    if not os.path.exists('masks500.npy'):
        print("Generating new masks...")
        rise.generate_masks(N=500, s=8, p1=0.5, savepath= 'masks500.npy')
    else:
        print("Loading existing masks...")
        rise.load_masks('masks500.npy')

    # Initialize RISEWithAdapter with the wrapped adapter
    explainer = RISEWithAdapter(
        clip_model=model,
        adapter=adapter_with_path,
        text_embedding=text_embedding,
        input_size=input_size,
        gpu_batch= 1024, #make this a lot bigger with alvis
        num_samples=50
    )
    explainer.set_masks(rise.masks)

    # Compute saliency
    print("Computing saliency map...")
    with torch.no_grad():
        try:
            saliency_norm, saliency_unnorm = explainer(image_tensor, bayes_results=bayes_results)
        except Exception as e:
            print(f"Error computing saliency map: {str(e)}")
            raise

    if save_vis_path:
        plot_attention_helper_p(image, attentions, unnormalized_attentions, probs, text_list,
                          save_vis_path=save_vis_path, resize=resize)

    probs = 1

    return {
        'unnorm_attentions': saliency_unnorm,
        'attentions': saliency_norm,
        'text_list': text_list,
        'probs': probs
    }
