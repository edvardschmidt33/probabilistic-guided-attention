import csv
import os
import random
from pathlib import Path
from typing import Literal, Sequence

import pytorch_lightning as L
import torch
import torch.utils
from PIL import Image

from .common import default_transform


def load_class_mapping(path):
    """
    Make a dictionary mapping between the subdirectory identifier and class description
    """
    mapping_dict = {}
    with open(path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            wnid_key = row['wnid']
            words_value = row['words'].split(',')[0]
            mapping_dict[wnid_key] = words_value
    return mapping_dict


class ImagenetVariationsDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            data: Sequence[dict],
            label_names: Sequence[str],
            text_prompt: str, 
            transform=None,
        ):
        self._data = data
        self._label_names = label_names
        self._text_prompt = text_prompt
        self._transform = transform

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        img_path, class_id = self._data[idx]['img_path'], self._data[idx]['class_id']

        text = self._text_prompt.format(
            class_name=self._label_names[class_id]
        )
        
        image = Image.open(img_path)
        if self._transform is not None:
            image = self._transform(image)
        
        return dict(image=image, text=text, class_id=class_id, image_id=idx, image_path=img_path)

def homeoffice_da_collate_fn(batch: dict):
    images = [item['image'] for item in batch]
    texts = [item['text'] for item in batch]
    image_ids = torch.tensor([item['image_id'] for item in batch]) if 'image_id' in batch[0] else None
    class_ids = torch.tensor([item['class_id'] for item in batch]) if 'class_id' in batch[0] else None
    image_paths = [item['image_path'] for item in batch] if 'image_path' in batch[0] else None

    # if images are tensors then stack them along the batch dimension
    # otherwise, return the list of images as is
    if len(images) > 0 and isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)

    d = dict(image=images, text=texts)

    if image_ids is not None:
        d['image_id'] = image_ids
    
    if class_ids is not None:
        d['class_id'] = class_ids

    if image_paths is not None:
        d['image_path'] = image_paths
    
    return d

class ImagenetDADataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'imagenet_variations'
    
    def __init__(
        self,
        data_dir: str,
        variant: Literal['imagenet-a', 'imagenet-r', 'imagenet-sketch'],
        batch_size: int = 32,
        num_workers: int = 4,
        text_prompt: str = "An image of a {class_name}",
        train_transform=default_transform(image_size=244),
        test_transform=default_transform(image_size=244),
        shuffle_train: bool = True,
        subset_indices: Sequence[int] = None,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices
        self.variant = variant

        self.domains = ['imagenet-a', 'imagenet-r', 'imagenet-sketch']
        self.class_mapping = load_class_mapping(data_dir / 'classes.csv')

    
    def scan_dir(self, data_dir, class_filter=None):
        classes = next(os.walk(data_dir))[1]

        if class_filter is not None:
            classes = [c for c in classes if c in class_filter]

        classes = sorted(classes)

        data = []
        for i, class_name in enumerate(classes):
            class_dir = data_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix in ['.jpg', '.JPEG']:
                    data.append(dict(img_path=img_path, class_id=i))

        data = sorted(data, key=lambda x: x['img_path'])

        return data, classes

    def setup(self, stage: str = None, filter_classes: bool = True):
        
        train_ds_all = {}
        val_ds_all = {}
        test_ds_all = {}

        if filter_classes:
            test_set_classes = next(os.walk(self.data_dir / self.variant))[1]
        else:
            test_set_classes = None

        for domain_name in self.domains:
            data, classes = self.scan_dir(self.data_dir / domain_name, class_filter=test_set_classes)

            label_names = [self.class_mapping[c] for c in classes]

            random.seed(42)
            random.shuffle(data)
            
            data_trainval, data_test = data[:int(0.8*len(data))], data[int(0.8*len(data)):]
            data_train, data_val = data_trainval[:int(0.8*len(data_trainval))], data_trainval[int(0.8*len(data_trainval)):]

            train_ds = ImagenetVariationsDataset(
                data=data_train,
                label_names=label_names,
                text_prompt=self.text_prompt,
                transform=self.train_transform,
            )
            if self.subset_indices is not None:
                train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)
            train_ds_all[domain_name] = train_ds

            val_ds = ImagenetVariationsDataset(
                data=data_val,
                label_names=label_names,
                text_prompt=self.text_prompt,
                transform=self.train_transform,
            )
            val_ds_all[domain_name] = val_ds
            
            test_ds = ImagenetVariationsDataset(
                data=data_test,
                label_names=label_names,
                text_prompt=self.text_prompt,
                transform=self.train_transform,
            )
            test_ds_all[domain_name] = test_ds

        # training/val/test domain selection
        # training dataset is concatenating the other datasets that are not in the target domain
        data_train_concat = []
        for v in self.domains:
            data_train_concat += train_ds_all[v]._data
        # random shuffling is disabled in the self.shuffle_train, so we shuffle the train data here instead
        random.shuffle(data_train_concat)
        self.train_ds = ImagenetVariationsDataset(
            data=data_train_concat,
            label_names=label_names,
            text_prompt=self.text_prompt,
            transform=self.train_transform,
        )

        self.val_ds = val_ds_all[self.variant]
        self.test_ds = test_ds_all[self.variant]

        print(f"Number of training images: {len(self.train_ds)}.")
        print(f"Number of test images: {len(self.test_ds)}.")
        print(f"Number of validation images: {len(self.val_ds)}.")
        print(f"Number of test classes: {len(self.train_ds._label_names)}.")
        print(f"Classes: {self.train_ds._label_names}")
            

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=homeoffice_da_collate_fn,
            shuffle=self.shuffle_train,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=homeoffice_da_collate_fn,
            persistent_workers=True,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=homeoffice_da_collate_fn,
            persistent_workers=True,
            shuffle=False,
        )

    @property
    def class_prompts(self):
        return [self.text_prompt.format(class_name=x) for x in self.train_ds._label_names]
    

class ImagenetDAAdversarialDataModule(ImagenetDADataModule):
    def __init__(self, **kwargs):
        super().__init__(variant='imagenet-a', **kwargs)

class ImagenetDARenditionsDataModule(ImagenetDADataModule):
    def __init__(self, **kwargs):
        super().__init__(variant='imagenet-r', **kwargs)

class ImagenetDASketchDataModule(ImagenetDADataModule):
    def __init__(self, **kwargs):
        super().__init__(variant='imagenet-sketch', **kwargs)
