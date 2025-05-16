import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))
sys.path.append(str(current_dir / 'BayesVLM-main'))

import torch
from PIL import Image
from bayesvlm.utils import get_model_type_and_size, get_image_size, get_transform, load_model
from bayesvlm.precompute import precompute_image_features, precompute_text_features, make_predictions
from bayesvlm.hessians import load_hessians, compute_covariances
from bayesvlm.vlm import EncoderResult

def get_probcosine_distribution(image_path: str, hessian_dir: str):
    # 1. Setup and model loading
    model_str = 'clip-base'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model components
    print("Loading model components...")
    image_encoder, text_encoder, vlm = load_model(model_str, device=device)
    
    # Get transform for the image
    model_type, _ = get_model_type_and_size(model_str)
    transform_image_size = get_image_size(model_str)
    transform = get_transform(model_type, transform_image_size)
    
    # 2. Load and process your image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    
    # 3. Get image features
    print("Extracting image features...")
    with torch.no_grad():
        # Create a batch dictionary as expected by the encoder
        batch = {
            'image': image_tensor,
            'class_id': torch.tensor([0]),  # Dummy class ID
            'image_id': torch.tensor([0])   # Dummy image ID
        }
        image_outputs = image_encoder(batch, return_activations=True)
    
    # 4. Process text prompt
    print("Processing text prompt...")
    text_prompt = ["a photo of a bird"]
    with torch.no_grad():
        text_outputs = precompute_text_features(
            text_encoder=text_encoder,
            class_prompts=text_prompt,
            batch_size=1
        )
    
    # 5. Load Hessian matrices and compute covariances
    print(f"Loading Hessian matrices from: {hessian_dir}")
    # Load image Hessians
    A_img, B_img, info = load_hessians(hessian_dir, tag='img', return_info=True)
    # Load text Hessians
    A_txt, B_txt = load_hessians(hessian_dir, tag='txt')
    
    cov_img, cov_txt = compute_covariances(A_img, B_img, A_txt, B_txt, info)
    vlm.set_covariances(cov_img, cov_txt)
    
    # 6. Get probabilistic logits
    print("Computing probabilistic logits...")
    prob_logits = make_predictions(
        clip=vlm,
        image_outputs=image_outputs,
        text_outputs=text_outputs,
        batch_size=1,
        device=device,
        map_estimate=False,
    )
    
    # 7. Convert to probabilities using probit approximation
    print("Computing final probabilities...")
    kappa = 1 / torch.sqrt(1. + torch.pi / 8 * prob_logits.var)
    probas = torch.softmax(kappa * prob_logits.mean, dim=-1)
    
    return {
        'probabilities': probas,
        'mean': prob_logits.mean,
        'variance': prob_logits.var,
        'kappa': kappa
    }

if __name__ == "__main__":
    # Example usage
    image_path = "Rusty_Blackbird_0015_6885.jpg"  # Updated to match your actual filename
    
    # Construct hessian directory path
    hessian_dir = Path(current_dir) / "BayesVLM-main" / "hessians" / "hessian_CLIP-ViT-B-32-laion2B-s34B-b79K"
    
    print(f"Starting probability distribution calculation...")
    try:
        results = get_probcosine_distribution(image_path, str(hessian_dir))
        print("\nResults:")
        print(f"Probability distribution: {results['probabilities']}")
        print(f"Mean: {results['mean']}")
        print(f"Variance: {results['variance']}")
        print(f"Kappa (uncertainty scaling): {results['kappa']}")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()