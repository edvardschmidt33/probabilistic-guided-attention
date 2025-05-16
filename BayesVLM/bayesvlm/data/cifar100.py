from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import torch
import pytorch_lightning as L
import datasets
from typing import Sequence

from .common import default_collate_fn, default_transform


class CIFAR100Dataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data: datasets.Dataset,
            text_prompt: str, 
            transform=None,
        ):
        self._data = data
        self._label_names = self._data.features['fine_label'].names
        self._text_prompt = text_prompt
        self._transform = transform
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        text = self._text_prompt.format(
            class_name=self._label_names[self._data[idx]['fine_label']]
        )

        image = self._data[idx]['img']
        if self._transform is not None:
            image = self._transform(image)
            
        return dict(image=image, text=text, class_id=self._data[idx]['fine_label'], image_id=idx)
    

class CIFAR100DataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'cifar100'

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
            # few shot parameters
            shots_per_class: int = 10,
            use_few_shot: bool = False,
            few_shot_sample_seed: int = 42,
        ):
        if use_few_shot:
            raise ValueError("Few shot not supported for this dataset")

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices

    def setup(self, stage: str = None):
        dataset = datasets.load_dataset('cifar100', cache_dir=self.data_dir)
        
        train_val_split = dataset['train'].train_test_split(test_size=0.2, seed=0)
        train_ds = train_val_split['train']
        val_ds = train_val_split['test']

        self.train_ds = CIFAR100Dataset(train_ds, text_prompt=self.text_prompt, transform=self.train_transform)
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.val_ds = CIFAR100Dataset(val_ds, text_prompt=self.text_prompt, transform=self.test_transform)
        self.test_ds = CIFAR100Dataset(dataset['test'], text_prompt=self.text_prompt, transform=self.test_transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=self.shuffle_train, 
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=default_collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=default_collate_fn,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds, 
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            collate_fn=default_collate_fn,
        )

    @property
    def class_prompts(self):
        return [self.text_prompt.format(class_name=name) for name in self.train_ds._label_names]
    