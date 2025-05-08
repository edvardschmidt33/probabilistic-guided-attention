"""CUB-200-2011 image-to-caption retrieval dataset code"""

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

class CUBDataset(Dataset):
    """CUB-200-2011 Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        caption_root (string): Path to caption files.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    def __init__(self, root, caption_root, transform=None, target_transform=None):
        self.root = os.path.expanduser(root)
        self.caption_root = os.path.expanduser(caption_root)
        self.transform = transform
        self.target_transform = target_transform
        
        # Load image paths and captions
        self.images = []
        self.captions = []
        
        # Walk through the image directory
        for class_dir in sorted(os.listdir(self.root)):
            class_path = os.path.join(self.root, class_dir)
            if os.path.isdir(class_path):
                for img_name in sorted(os.listdir(class_path)):
                    if img_name.endswith('.jpg'):
                        img_path = os.path.join(class_path, img_name)
                        caption_path = os.path.join(self.caption_root, 
                                                  class_dir, 
                                                  img_name.replace('.jpg', '.txt'))
                        
                        if os.path.exists(caption_path):
                            with open(caption_path, 'r') as f:
                                caption = f.read().strip()
                            
                            self.images.append(img_path)
                            self.captions.append(caption)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is a caption for the annotation.
        """
        img_path = self.images[index]
        caption = self.captions[index]
        
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
        return len(self.images)

def prepare_cub_dataloaders(dataloader_config, dataset_root, caption_root, vocab_path=None):
    """
    Prepare CUB dataloaders for training, validation and testing.
    
    Args:
        dataloader_config: Configuration for dataloaders
        dataset_root: Root directory of the CUB dataset
        caption_root: Path to caption files
        vocab_path: Path to vocabulary file (optional)
    
    Returns:
        Dictionary containing train, validation and test dataloaders
    """
    # Create datasets
    train_dataset = CUBDataset(
        root=os.path.join(dataset_root, 'train'),
        caption_root=os.path.join(caption_root, 'train'),
        transform=None,
        target_transform=None
    )
    
    val_dataset = CUBDataset(
        root=os.path.join(dataset_root, 'val'),
        caption_root=os.path.join(caption_root, 'val'),
        transform=None,
        target_transform=None
    )
    
    test_dataset = CUBDataset(
        root=os.path.join(dataset_root, 'test'),
        caption_root=os.path.join(caption_root, 'test'),
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