"""Flickr30k image-to-caption retrieval dataset code"""

import os
from os.path import join as ospj
try:
    import ujson as json
except ImportError:
    import json

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

import clip
from torchvision import transforms

# Initialize CLIP preprocessing
_, preprocess = clip.load("ViT-B/32", device="cpu")
vis_processors = {"eval": preprocess}

# Simple text preprocessing
def text_preprocess(text):
    return text.lower().strip()
txt_processors = {"eval": text_preprocess}

class FlickrDataset(Dataset):
    """Flickr30k Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, annFile, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        with open(annFile, 'r') as f:
            self.annotations = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        ann = self.annotations[index]
        img_path = os.path.join(self.root, ann['image'])
        caption = ann['caption']
        
        img = Image.open(img_path).convert('RGB')
        
        # Apply CLIP preprocessing
        img = vis_processors["eval"](img)
        
        if self.target_transform is not None:
            target = self.target_transform(caption)
            target = target.squeeze(0)
        
        # Apply text preprocessing
        target = txt_processors["eval"](target)
        
        img_masked = img
        is_img_masked = False
        
        return img, target, img_masked, is_img_masked, img_path

    def __len__(self):
        return len(self.annotations)

def prepare_flickr_dataloaders(dataloader_config, dataset_root, vocab_path=None):
    """
    Prepare Flickr dataloaders for training, validation and testing.
    
    Args:
        dataloader_config: Configuration for dataloaders
        dataset_root: Root directory of the Flickr dataset
        vocab_path: Path to vocabulary file (optional)
    
    Returns:
        Dictionary containing train, validation and test dataloaders
    """
    # Define paths
    train_ann = os.path.join(dataset_root, 'annotations', 'train.json')
    val_ann = os.path.join(dataset_root, 'annotations', 'val.json')
    test_ann = os.path.join(dataset_root, 'annotations', 'test.json')
    img_dir = os.path.join(dataset_root, 'images')
    
    # Create datasets
    train_dataset = FlickrDataset(
        root=img_dir,
        annFile=train_ann,
        transform=None,
        target_transform=None
    )
    
    val_dataset = FlickrDataset(
        root=img_dir,
        annFile=val_ann,
        transform=None,
        target_transform=None
    )
    
    test_dataset = FlickrDataset(
        root=img_dir,
        annFile=test_ann,
        transform=None,
        target_transform=None
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=dataloader_config['traindata_shuffle'],
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=dataloader_config['batch_size'],
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    } 