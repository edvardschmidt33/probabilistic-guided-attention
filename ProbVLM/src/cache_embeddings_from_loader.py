import torch
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
import clip

import json
from torch.utils.data import DataLoader
import pickle

def cache_embeddings_from_loaders(loaders_path, output_dir):
    # Load the existing dataloaders
    with open(loaders_path, 'rb') as f:
        loaders = pickle.load(f)
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # Setup CLIP model
    device = "cuda:0"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()
    
    # Create output directories
    os.makedirs(f'{output_dir}/train', exist_ok=True)
    os.makedirs(f'{output_dir}/val', exist_ok=True)
    os.makedirs(f'{output_dir}/test', exist_ok=True)
    
    # Function to process a dataloader
    def process_loader(dataloader, split):
        print(f'Processing {split} set...')
        
        # Initialize lists to store embeddings and mappings
        image_embeddings = []
        text_embeddings = []
        cap_id_to_img_id = {}

        # Maintain a map from unique img_id to image index
        img_id_to_index = {}
        current_img_index = 0

        with torch.no_grad():
            for batch_idx, (images, texts, _, ann_ids, img_ids) in enumerate(dataloader):
                print(f'Processing batch {batch_idx+1}/{len(dataloader)}', flush=True)

                # Process images
                images = images.to(device)
                img_emb = model.encode_image(images, is_weights=False)
                image_embeddings.append(img_emb.cpu())

                # Process texts
                texts = texts.to(device)
                txt_emb = model.encode_text(texts)
                text_embeddings.append(txt_emb.cpu())

                # Map caption ID to image index
                for i, (ann_id, img_id) in enumerate(zip(ann_ids, img_ids)):
                    img_id_str = str(img_id)
                    if img_id_str not in img_id_to_index:
                        img_id_to_index[img_id_str] = current_img_index
                        current_img_index += 1
                    cap_id_to_img_id[str(len(cap_id_to_img_id))] = img_id_to_index[img_id_str]
        
        # Concatenate embeddings
        image_embeddings = torch.cat(image_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        
        # Save embeddings and mapping
        torch.save(image_embeddings, f'{output_dir}/{split}/image.pt')
        torch.save(text_embeddings, f'{output_dir}/{split}/text.pt')
        with open(f'{output_dir}/{split}/cap_id_to_img_id.json', 'w') as f:
            json.dump(cap_id_to_img_id, f)
        
        print(f'Finished processing {split} set')
    
    # Process each split
    process_loader(train_loader, 'train')
    process_loader(val_loader, 'val')
    process_loader(test_loader, 'test')

if __name__ == '__main__':
    loaders_path = '/mimer/NOBACKUP/groups/ulio_inverse/ds-project/ProbVLM/Datasets/coco/data_loaders_coco_person_extra_26.11.pkl'
    cache_embeddings_from_loaders(loaders_path)