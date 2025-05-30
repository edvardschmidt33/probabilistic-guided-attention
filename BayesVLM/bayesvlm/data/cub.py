import torch
import pytorch_lightning as L
import json
from typing import Sequence
from pathlib import Path
from PIL import Image
from collections import defaultdict
import numpy as np
import torch.utils

from .common import default_collate_fn, default_transform

def _label_names_from_split_info(split_info):
    class_folders = set(path.split("/")[0] for path in split_info)
    class_list = sorted(class_folders)
    return class_list

class CUBDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            image_dir: str,
            split_info: list,
            text_prompt: str, 
            transform=None,
            use_few_shot = False,
            shots_per_class = 5,
            few_shot_sample_seed = 0,
        ):
        self._data_dir = Path(image_dir)
        self._split_info = split_info
        self.image_paths = [self._data_dir / path for path in split_info]
        self._label_names = _label_names_from_split_info(split_info)
        self._text_prompt = text_prompt
        self._transform = transform

        self.use_few_shot = use_few_shot
        if self.use_few_shot:
            self.shots_per_class = shots_per_class
            self.few_shot_sample_seed = few_shot_sample_seed

            # get the index for each class
            self.class_index = defaultdict(list)
            for i in range(self.__len__()):
                class_id = self.get_class_id(i)
                self.class_index[class_id].append(i)
            
            # create few-shot dataset through sampling
            selected_data = []
            for indices in self.class_index.values():
                np.random.seed(self.few_shot_sample_seed)
                selected_data.extend(np.random.choice(indices, self.shots_per_class, replace=False))
            self.selected_data = selected_data

    def __len__(self):
        return len(self._split_info)

    def get_class_id(self, idx):
        path = self._split_info[idx]
        class_folder = path.split("/")[0]
        class_id = self._label_names.index(class_folder)
        return class_id
    
    def __getitem__(self, idx):
        path = self._split_info[idx]
        class_folder = path.split("/")[0]
        class_id = self._label_names.index(class_folder)

        text = self._text_prompt.format(class_name=self._label_names[class_id])

        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        if self._transform is not None:
            image = self._transform(image)
            
        return dict(image=image, text=text, class_id=class_id, image_id=idx)


class CUBDataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'cub'

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        text_prompt: str = "An image of a {class_name}",
        train_transform=default_transform(image_size=244),
        test_transform=default_transform(image_size=244),
        shuffle_train: bool = True,
        subset_indices: Sequence[int] = None,
        shots_per_class: int = 10,
        use_few_shot: bool = False,
        few_shot_sample_seed: int = 42,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices
        
        self.use_few_shot = use_few_shot
        self.shots_per_class = shots_per_class
        self.few_shot_sample_seed = few_shot_sample_seed

    def setup(self, stage: str = None):
        splits_file = Path(self.data_dir) / 'split_CUB.json'
        with open(splits_file) as f:
            splits_info = json.load(f)

        if self.use_few_shot:
            self.train_ds = CUBDataset(
                image_dir=self.data_dir / 'images',
                split_info=splits_info['train'],
                text_prompt=self.text_prompt,
                transform=self.train_transform,
                use_few_shot = True,
                shots_per_class = self.shots_per_class,
                few_shot_sample_seed = self.few_shot_sample_seed
            )
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.train_ds.selected_data)
        else:
            self.train_ds = CUBDataset(
                image_dir=self.data_dir / 'images',
                split_info=splits_info['train'],
                text_prompt=self.text_prompt,
                transform=self.train_transform,
            )
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.val_ds = CUBDataset(
            image_dir=self.data_dir / 'images',
            split_info=splits_info['val'],
            text_prompt=self.text_prompt,
            transform=self.test_transform,
        )
        
        self.test_ds = CUBDataset(
            image_dir=self.data_dir / 'images',
            split_info=splits_info['test'],
            text_prompt=self.text_prompt,
            transform=self.test_transform,
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=default_collate_fn,
            shuffle=self.shuffle_train,
            persistent_workers=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=default_collate_fn,
            persistent_workers=True,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=default_collate_fn,
            persistent_workers=True,
            shuffle=False,
        )

    @property
    def class_prompts(self):
        if self.use_few_shot:
            return [self.text_prompt.format(class_name=name) for name in self.test_ds._label_names]
        else:
            return [self.text_prompt.format(class_name=name) for name in self.train_ds._label_names]