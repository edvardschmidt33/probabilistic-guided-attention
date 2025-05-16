from typing import Sequence
import torch
from pathlib import Path
import pytorch_lightning as L
from torchvision.datasets import StanfordCars
from .common import default_collate_fn, default_transform
from collections import defaultdict
import numpy as np

class StanfordCarsWithLabels(StanfordCars):
    def __init__(self, *args, **kwargs):
        self._text_prompt = kwargs['text_prompt']
        del kwargs['text_prompt']
        
        self.use_few_shot = kwargs['use_few_shot']
        if self.use_few_shot:
            self.shots_per_class = kwargs['shots_per_class']
            self.few_shot_sample_seed = kwargs['few_shot_sample_seed']
            del kwargs['shots_per_class']
            del kwargs['few_shot_sample_seed']
        del kwargs['use_few_shot']

        super().__init__(*args, **kwargs)
        
        self.indices = list(range(super().__len__()))
        
        if self.use_few_shot:
            
            # get the index for each class
            self.class_index = defaultdict(list)
            
            for img_index in range(super().__len__()):
                _, class_id = self._samples[img_index]
                self.class_index[class_id].append(img_index)

            # create few-shot dataset through sampling
            selected_data = []
            for indices in self.class_index.values():
                np.random.seed(self.few_shot_sample_seed)
                selected_data.extend(np.random.choice(indices, self.shots_per_class, replace=False))
            self.selected_data = selected_data
       
    @property
    def _label_names(self):
        return self.classes
    
    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index):
        img, class_id = super().__getitem__(index)

        text = self._text_prompt.format(
            class_name=self.classes[class_id],
        )

        return dict(
            image=img,
            text=text,
            class_id=class_id,
            image_id=index,
        )

class StanfordCarsDataModule(L.LightningDataModule):
    DATASET_SUBDIR = ''

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 4,
        text_prompt: str = "An image of a {class_name}",
        train_transform=default_transform(image_size=224),
        test_transform=default_transform(image_size=224),
        shuffle_train: bool = True,
        subset_indices: Sequence[int] = None,
        shots_per_class: int = 10,
        use_few_shot: bool = False,
        few_shot_sample_seed: int = 42,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = Path(data_dir)
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices
        
        self.use_few_shot = use_few_shot
        self.shots_per_class = shots_per_class
        self.few_shot_sample_seed = few_shot_sample_seed

    def setup(self, stage: str = None):
        if self.use_few_shot:
            self.train_ds = StanfordCarsWithLabels(
                root = self.data_dir,
                split='train',
                transform=self.train_transform,
                download=False,
                text_prompt=self.text_prompt,
                use_few_shot = True,
                shots_per_class = self.shots_per_class,
                few_shot_sample_seed = self.few_shot_sample_seed
            )
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.train_ds.selected_data)
        else:
            self.train_ds = StanfordCarsWithLabels(
                root = self.data_dir,
                split='train',
                transform=self.train_transform,
                download=False,
                text_prompt=self.text_prompt,
                use_few_shot = False,
            )
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.test_ds = StanfordCarsWithLabels(
            root = self.data_dir,
            split='test',
            transform=self.test_transform,
            download=False,
            text_prompt=self.text_prompt,
            use_few_shot = False,
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
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=default_collate_fn,
            shuffle=False,
            persistent_workers=True,
        )
    
    @property
    def class_prompts(self):
        if self.use_few_shot:
            return [self.text_prompt.format(class_name=name) for name in self.test_ds._label_names]
        else:
            return [self.text_prompt.format(class_name=name) for name in self.train_ds._label_names]