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
from train_probVLM_FOOD import *
import json
import pickle
# import sys
# sys.path.append('/path/to/your/module/directory')  # e.g. '/cephyr/users/schmidte/Alvis/Paric_nolavis/ProbVLM/src'

from cache_embeddings_from_loader import cache_embeddings_from_loaders

from cache_embedding import CachedEmbeddingDataset, create_dataloaders



if __name__ == '__main__':
    dataset = 'food-101'
    data_dir = ospj('/mimer/NOBACKUP/groups/ulio_inverse/ds-project/GALS/data/', dataset)
    pkl_file = ospj('/mimer/NOBACKUP/groups/ulio_inverse/ds-project/ProbVLM/Datasets/food-101/', 'data_loaders_food_binary_03.10.pkl')
    output_dir = 'embeddings/food'

    # Step 1: Cache embeddings
    cache_embeddings_from_loaders(pkl_file, output_dir=output_dir)

    # Step 2: Create dataloaders from cached embeddings
    train_split_dir = ospj(output_dir, 'train')
    val_split_dir = ospj(output_dir, 'val')
    test_split_dir = ospj(output_dir, 'test')

    # Get train and val from training split
    train_loader, val_loader = create_dataloaders(train_split_dir, batch_size=16384, num_workers=12, split_train_val=True)

    # Load test set as full loader
    test_loader = create_dataloaders(test_split_dir, batch_size=16384, num_workers=12, split_train_val=False)


    # clip_net = load_model('cuda')
    CLIP_Net = load_model(device='cuda', model_path=None)
    ProbVLM_Net = BayesCap_for_CLIP(
        inp_dim=512,
        out_dim=512,
        hid_dim=256,
        num_layers=3,
        p_drop=0.05,
    )


    train_ProbVLM(
        CLIP_Net,
        ProbVLM_Net,
        cub_train_loader,
        cub_valid_loader,
        Cri = TempCombLoss(),
        device='cuda',
        dtype=torch.cuda.FloatTensor,
        init_lr=8e-5,
        num_epochs=200,
        eval_every=5,
        ckpt_path='/cephyr/users/schmidte/Alvis/Paric_nolavis/ckpt/ProbVLM_waterbirds_200epochs',
        T1=1e0,
        T2=1e-4,
        use_cached_embeddings=True
    )
