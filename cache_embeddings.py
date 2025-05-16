import torch
import clip
import os
import json
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm

def load_pickle_data(pickle_path):
    """Load the data from the pickle file."""
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)

def preprocess_image(image):
    """Preprocess image for CLIP model."""
    return clip.preprocess(image).unsqueeze(0)

@torch.no_grad()
def cache_embeddings(pickle_path, output_dir, model_name="ViT-B/32", batch_size=512):
    """
    Precompute and cache embeddings for images and captions.
    
    Args:
        pickle_path (str): Path to the data_loaders_coco_person_extra_26.11.pkl file
        output_dir (str): Directory to save the cached embeddings
        model_name (str): CLIP model name to use
        batch_size (int): Batch size for processing
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load(model_name, device=device)
    model.eval()
    
    # Load data from pickle file
    print("Loading data from pickle file...")
    data = load_pickle_data(pickle_path)
    
    # Process images
    print("Processing images...")
    image_embeddings = []
    image_ids = []
    
    for batch_idx in tqdm(range(0, len(data['train'].dataset), batch_size)):
        batch_images = []
        batch_ids = []
        
        for idx in range(batch_idx, min(batch_idx + batch_size, len(data['train'].dataset))):
            image, _ = data['train'].dataset[idx]
            batch_images.append(preprocess_image(image))
            batch_ids.append(idx)
        
        if batch_images:
            batch_images = torch.cat(batch_images, dim=0).to(device)
            with torch.no_grad():
                embeddings = model.encode_image(batch_images)
            image_embeddings.append(embeddings.cpu())
            image_ids.extend(batch_ids)
    
    image_embeddings = torch.cat(image_embeddings, dim=0)
    
    # Process captions
    print("Processing captions...")
    text_embeddings = []
    caption_ids = []
    
    for batch_idx in tqdm(range(0, len(data['train'].dataset), batch_size)):
        batch_captions = []
        batch_ids = []
        
        for idx in range(batch_idx, min(batch_idx + batch_size, len(data['train'].dataset))):
            _, caption = data['train'].dataset[idx]
            batch_captions.append(clip.tokenize(caption, truncate=True))
            batch_ids.append(idx)
        
        if batch_captions:
            batch_captions = torch.cat(batch_captions, dim=0).to(device)
            with torch.no_grad():
                embeddings = model.encode_text(batch_captions)
            text_embeddings.append(embeddings.cpu())
            caption_ids.extend(batch_ids)
    
    text_embeddings = torch.cat(text_embeddings, dim=0)
    
    # Save embeddings
    print("Saving embeddings...")
    torch.save(image_embeddings, os.path.join(output_dir, 'image_embeddings.pt'))
    torch.save(text_embeddings, os.path.join(output_dir, 'text_embeddings.pt'))
    
    # Save mapping
    mapping = {
        'image_ids': image_ids,
        'caption_ids': caption_ids
    }
    with open(os.path.join(output_dir, 'mapping.json'), 'w') as f:
        json.dump(mapping, f)
    
    print("Embeddings cached successfully!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_path', type=str, required=True,
                      help='Path to the data_loaders_coco_person_extra_26.11.pkl file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the cached embeddings')
    parser.add_argument('--model_name', type=str, default='ViT-B/32',
                      help='CLIP model name to use')
    parser.add_argument('--batch_size', type=int, default=512,
                      help='Batch size for processing')
    
    args = parser.parse_args()
    cache_embeddings(args.pickle_path, args.output_dir, args.model_name, args.batch_size)
