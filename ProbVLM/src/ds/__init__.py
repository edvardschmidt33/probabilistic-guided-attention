"""Modules for multi-modal datasets

PCME
Copyright (c) 2021-present NAVER Corp.
MIT license
"""

from ds._dataloader import  prepare_cub_dataloaders
from ds._dataloader import prepare_coco_dataloaders, prepare_flickr_dataloaders
from ds._dataloader import prepare_fashion_dataloaders
from ds._dataloader import prepare_flo_dataloaders
from ds._dataloader_extra import prepare_foodmeat_dataloaders, prepare_foodredmeat_dataloaders
# from ds._dataloader import load_mnist_data_loader
from ds.vocab import Vocabulary
from ds._dataloader_extra import prepare_coco_dataloaders_extra
from ds._dataloader_extra import prepare_cub_dataloaders_extra


__all__ = [
    'Vocabulary',
    'prepare_coco_dataloaders',
    'prepare_cub_dataloaders',
    # 'prepare_coco_dataset_with_bbox',
    'prepare_flickr_dataloaders',
    # 'prepare_flickr_dataset_with_bbox',
    'prepare_fashion_dataloaders',
    'prepare_flo_dataloaders',
    'prepare_coco_dataloaders_extra',
    'prepare_cub_dataloaders_extra',
    'prepare_foodmeat_dataloaders',
    'prepare_foodredmeat_dataloaders'
]
