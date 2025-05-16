import torch
import pytorch_lightning as L
import datasets
from typing import Sequence

from .common import default_collate_fn, default_transform


class ImagenetSketchDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            data: datasets.Dataset,
            text_prompt: str, 
            transform=None,
        ):
        self._data = data
        self._label_names = self._data.features['label'].names
        self._text_prompt = text_prompt
        self._transform = transform
    
    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, idx):
        text = self._text_prompt.format(
            class_name=self._label_names[self._data[idx]['label']]
        )

        image = self._data[idx]['image']
        if self._transform is not None:
            image = self._transform(image)
            
        return dict(image=image, text=text, class_id=self._data[idx]['label'], image_id=idx)


class ImagenetSketchDataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'imagenet-sketch'

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
        ):
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
        dataset = datasets.load_dataset('songweig/imagenet_sketch', cache_dir=self.data_dir)

        train_test_split = dataset['train'].train_test_split(test_size=0.025, seed=0)
        train_ds = train_test_split['train']
        test_ds = train_test_split['test']

        train_val_split = train_ds.train_test_split(test_size=0.025, seed=0)
        train_ds = train_val_split['train']
        val_ds = train_val_split['test']

        train_ds = train_ds.train_test_split(test_size=0.03, seed=0)['test']
        
        self.train_ds = ImagenetSketchDataset(train_ds, text_prompt=self.text_prompt, transform=self.train_transform)
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.val_ds = ImagenetSketchDataset(val_ds, text_prompt=self.text_prompt, transform=self.test_transform)
        self.test_ds = ImagenetSketchDataset(test_ds, text_prompt=self.text_prompt, transform=self.test_transform)

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
