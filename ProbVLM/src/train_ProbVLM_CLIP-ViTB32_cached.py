#!/usr/bin/env python
# coding: utf-8

import os
import torch
from torch.utils.data import DataLoader, Dataset
from os.path import join as ospj
from munch import Munch as mch
import numpy as np
from utils import *
from networks import *
from train_probVLM import *
import json
import pickle
# import sys
# sys.path.append('/path/to/your/module/directory')  # e.g. '/cephyr/users/schmidte/Alvis/Paric_nolavis/ProbVLM/src'

from cache_embeddings_from_loader import cache_embeddings_from_loaders



class CachedEmbeddingDataset(Dataset):
    def __init__(self, image_embeddings_path, text_embeddings_path, mapping_path):
        self.image_embeddings = torch.load(image_embeddings_path).clone().detach()
        self.text_embeddings = torch.load(text_embeddings_path).clone().detach()
        with open(mapping_path, 'r') as f:
            self.mapping = json.load(f)

    def __len__(self):
        return len(self.text_embeddings)

    def __getitem__(self, idx):
        img_id = self.mapping[str(idx)]
        return self.image_embeddings[img_id], self.text_embeddings[idx]

# ---------- Dataloader Creation ----------
def create_dataloaders(split_dir, batch_size=256, num_workers=12, split_train_val=True):
    dataset = CachedEmbeddingDataset(
        ospj(split_dir, 'image.pt'),
        ospj(split_dir, 'text.pt'),
        ospj(split_dir, 'cap_id_to_img_id.json')
    )

    if split_train_val:
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=16)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=16)
        return train_loader, val_loader
    else:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=16)
        return loader


# Main training script
if __name__ == '__main__':

    dataset = 'coco'
    data_dir = ospj('/mimer/NOBACKUP/groups/ulio_inverse/ds-project/ProbVLM/Datasets/', dataset)
    pkl_file = ospj(data_dir, 'data_loaders_coco_person_extra_26.11.pkl')
    output_dir = 'embeddings/coco'

    # Step 1: Cache embeddings
    # cache_embeddings_from_loaders(pkl_file, output_dir=output_dir)

    # Step 2: Create dataloaders from cached embeddings
    train_split_dir = ospj(output_dir, 'train')
    val_split_dir = ospj(output_dir, 'val')
    test_split_dir = ospj(output_dir, 'test')

    # Get train and val from training split
    train_loader, val_loader = create_dataloaders(train_split_dir, batch_size=8192, num_workers=12, split_train_val=True)

    # Load test set as full loader
    test_loader = create_dataloaders(test_split_dir, batch_size=8192, num_workers=12, split_train_val=False)



    device = 'cuda'
    # Initialize models
    CLIP_Net = load_model(device='cuda', model_path=None)
    ProbVLM_Net = BayesCap_for_CLIP(
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3,
        p_drop=0.05,
    )
    
    # Train the model
    train_ProbVLM(
        CLIP_Net,
        ProbVLM_Net,
        train_loader,
        val_loader,
        Cri=TempCombLoss(),
        device='cuda',
        dtype=torch.cuda.FloatTensor,
        init_lr=8e-5,
        num_epochs=200,
        eval_every=5,
        ckpt_path='/cephyr/users/schmidte/Alvis/Paric_nolavis/ckpt/ProbVLM_Coco_cached',
        T1=1e0,
        T2=1e-4,
        use_cached_embeddings = True
    )

