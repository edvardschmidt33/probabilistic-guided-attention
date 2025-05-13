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
def create_dataloaders(split_dir, batch_size=64, num_workers=2, split_train_val=True):
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

