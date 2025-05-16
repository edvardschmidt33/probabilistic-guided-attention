import torch
import pytorch_lightning as L
import random
from typing import Sequence
from pathlib import Path
from PIL import Image
from typing import Literal

import torch.utils

from .common import default_collate_fn, default_transform


class HomeOfficeDataset(torch.utils.data.Dataset):
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
            
        return dict(image=image, text=text, class_id=class_id, image_id=idx)


class HomeOfficeDataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'homeoffice'
    
    def __init__(
        self,
        data_dir: str,
        variant: Literal['Art', 'Clipart', 'Product', 'Real World'],
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
        self.data_dir = Path(data_dir) / variant
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices

    
    def scan_dir(self, data_dir):
        classes = []
        for class_dir in data_dir.iterdir():
            if class_dir.is_dir():
                classes.append(class_dir.name)
        classes = sorted(classes)

        data = []
        for i, class_name in enumerate(classes):
            class_dir = data_dir / class_name
            for img_path in class_dir.iterdir():
                if img_path.suffix in ['.jpg']:
                    data.append(dict(img_path=img_path, class_id=i))

        data = sorted(data, key=lambda x: x['img_path'])
        return data, classes

    def setup(self, stage: str = None):
        data, classes = self.scan_dir(self.data_dir)

        print(f"Found {len(data)} images in {len(classes)} classes")
        print(f"Classes: {classes}")

        random.seed(42)
        random.shuffle(data)
        
        data_trainval, data_test = data[:int(0.8*len(data))], data[int(0.8*len(data)):]
        data_train, data_val = data_trainval[:int(0.8*len(data_trainval))], data_trainval[int(0.8*len(data_trainval)):]

        self.train_ds = HomeOfficeDataset(
            data=data_train,
            label_names=classes,
            text_prompt=self.text_prompt,
            transform=self.train_transform,
        )
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.val_ds = HomeOfficeDataset(
            data=data_val,
            label_names=classes,
            text_prompt=self.text_prompt,
            transform=self.train_transform,
        )
        
        self.test_ds = HomeOfficeDataset(
            data=data_test,
            label_names=classes,
            text_prompt=self.text_prompt,
            transform=self.train_transform,
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
        return [self.text_prompt.format(class_name=x) for x in self.train_ds._label_names]
    

class HomeOfficeArtDataModule(HomeOfficeDataModule):
    def __init__(self, **kwargs):
        super().__init__(variant='Art', **kwargs)

class HomeOfficeClipartDataModule(HomeOfficeDataModule):
    def __init__(self, **kwargs):
        super().__init__(variant='Clipart', **kwargs)

class HomeOfficeProductDataModule(HomeOfficeDataModule):
    def __init__(self, **kwargs):
        super().__init__(variant='Product', **kwargs)

class HomeOfficeRealWorldDataModule(HomeOfficeDataModule):
    def __init__(self, **kwargs):
        super().__init__(variant='Real World', **kwargs)
