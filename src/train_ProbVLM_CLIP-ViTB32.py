#!/usr/bin/env python
# coding: utf-8


import sys
# sys.path.append('/Users/Edvard/Desktop/Kandidatarbete/PARIC/LAVIS')
# import lavis

import os

from os.path import join as ospj
from os.path import expanduser
from munch import Munch as mch
import numpy as np

#from ds import prepare_coco_dataloaders_extra 
#load_mnist_data_loader, prepare_flickr_dataloaders, prepare_cub_dataloaders, prepare_flo_dataloaders

from utils import *
from networks import *
from train_probVLM import *

import matplotlib.pyplot as plt

import pickle
import os



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
filename = '/mimer/NOBACKUP/groups/ulio_inverse/ds-project/ProbVLM/Datasets/coco/data_loaders_coco_person_extra_26.11.pkl' #change 
loaders = load_data_loaders(filename)


coco_train_loader, coco_valid_loader, coco_test_loader = loaders['train'], loaders['val'], loaders['test']


# clip_net = load_model('cuda')
CLIP_Net = load_model(device='cuda', model_path=None)
ProbVLM_Net = BayesCap_for_CLIP(
    inp_dim=512,
    out_dim=512,
    hid_dim=256,
    num_layers=3,
    p_drop=0.05,
)
#shouldnt thsi be 512?


train_ProbVLM(
    CLIP_Net,
    ProbVLM_Net,
    coco_train_loader,
    coco_valid_loader,
    Cri = TempCombLoss(),
    device='cuda',
    dtype=torch.cuda.FloatTensor,
    init_lr=8e-5,
    num_epochs=200,
    eval_every=5,
    ckpt_path='/Users/Edvard/Desktop/Kandidatarbete/PARIC/ckpt/ProbVLM_Coco_extra_26.11', #change address to local adress
    T1=1e0,
    T2=1e-4
)

