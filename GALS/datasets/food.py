import os
import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split


class FoodSubset(datasets.ImageFolder):

    def __init__(
            self,
            root,
            split=None,
            cfg=None,
            transform=None,
            target_transform=None,
            is_valid_file=None,
    ):
        super().__init__(
            f"{root}/food-101/images",  # Use the main image directory
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file
        )

        self.cfg = cfg
        if 'red' in split:
            labels_txt_path = f"{root}/food-101/meta/labels-redmeat.txt"
        else:
            labels_txt_path = f"{root}/food-101/meta/labels-meat.txt"
        #print(labels_txt_path)
        # Load class labels from 'labels.txt'

        if os.path.exists(labels_txt_path):
            with open(labels_txt_path) as f:
                self.classes = [line.strip().replace(' ', '_').lower() for line in f]
        else:
            self.classes = sorted(os.listdir(f"{root}/food-101/images"))  # Infer from folder names

        # Load train/test split from text files
        split_file = f"{root}/food-101/meta/{split}.txt"
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found!")

        with open(split_file, "r") as f:
            img_list = [line.strip() for line in f] 
            # Read image IDs from train/test file
        #print('number of imgs:', len(img_list), 'in split file:', split_file )
        # Collect image paths based on split
        self.imgs = []
        self.filename_array = []
        for img_id in img_list:
            class_name, img_name = img_id.split('/')
            img_path = os.path.join(f"{root}/food-101/images", f"{img_id}.jpg")  # Full path

            if os.path.exists(img_path):  # Ensure the file exists
                self.filename_array.append(f"{img_id}.jpg")  # Relative path
                #print('classes originally defined', self.classes.index(class_name))
                #print('classes as we defined them', img_path.split('/')[-2])
                self.imgs.append((img_path, img_path.split('/')[-2]))
                self.imgs.append((img_path, 'an image of animal based meat'))
                self.imgs.append((img_path, 'a photo of animal based meat'))

        self.data = np.array([img[0] for img in self.imgs])
        self.samples = self.imgs
        self.groups = [0] * len(self.imgs)  # Default grouping

        
        # Handle attention maps
        self.return_attention = cfg.DATA.ATTENTION_DIR != "NONE"
        self.size = cfg.DATA.SIZE

        if self.return_attention:
            self.attention_data = np.array([
                os.path.join(root, cfg.DATA.ATTENTION_DIR, f.replace('.jpg', '.pth'))
                for f in self.filename_array
            ])

    def __getitem__(self, index):     
        
        path, target = self.imgs[index]
        group = self.groups[index]
        #print(self.classes[self.groups[index]])
        
        target_tokenized = target#self.classes[self.groups[index]]
        #print('target before transform: ', target_tokenized)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            #print('before tranform', self.classes[self.groups[index]])
            target_tokenized = self.target_transform(target).squeeze()
            #print('after tranform', target_tokenized)
      

        if self.return_attention:
            att = torch.load(self.attention_data[index])
            if self.cfg.DATA.ATTENTION_DIR == 'deeplabv3_attention':
                att = att['mask']
            else:
                att = att['unnormalized_attentions']
            att = F.interpolate(att, size=(self.size, self.size), mode='bilinear', align_corners=False)
        else:
            att = torch.Tensor([-1])  # Placeholder for batching

        NULL = torch.Tensor([-1])  # Placeholder for segmentation/bbox
        
        cap = path.split('/')[-2]#self.classes[self.groups[index]]
        
        return {
            'image_path': path,
            'image': sample,
            'label': target,
            'tokenized label':  target_tokenized,
            'caption': cap,
            'seg': NULL,
            'group': group,
            'bbox': NULL,
            'attention': att,
        }
