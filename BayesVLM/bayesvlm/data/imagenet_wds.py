import torch
import webdataset as wds
import lightning as L
from PIL import Image
import io
import os
from typing import Sequence
from datasets import load_dataset_builder
import numpy as np

from .common import default_transform


def make_label_lookup():
    ds_builder = load_dataset_builder("ILSVRC/imagenet-1k")
    classes = ds_builder.info.features['label'].names
    classes = np.array(classes)
    class_indices = np.arange(len(classes))
    return dict(zip(class_indices, classes))

class ImagenetWDSModule(L.LightningDataModule):
    DATASET_SUBDIR = 'imagenet_val_wds'

    def __init__(
            self, 
            data_dir: str,
            batch_size: int = 32, 
            num_workers: int = 4,
            text_prompt: str = "An image of a {class_name}",
            train_transform=default_transform(image_size=244),
            test_transform=default_transform(image_size=244),
            shuffle_train: bool = False,
            subset_indices: Sequence[int] = None,
            # few shot parameters
            shots_per_class: int = 10,
            use_few_shot: bool = False,
            few_shot_sample_seed: int = 42,
        ):
        if use_few_shot:
            raise ValueError("Few shot not supported for this dataset")

        super().__init__()

        tarfiles = list(data_dir.glob("*.tar"))
        print(f'found {len(tarfiles)} tar files in {data_dir}')

        self.data_path = sorted([os.path.join(data_dir, tarfile) for tarfile in tarfiles])
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.label_lookup = make_label_lookup()
        self.text_prompt = text_prompt

        if subset_indices is not None:
            raise ValueError("Subset indices are not supported for this dataset")

    def _preprocess(self, item):
        output = {}
        image_data = item['jpg']
        image = Image.open(io.BytesIO(image_data))
        image_tensor = self.train_transform(image)
        
        class_id = int(item['cls'])
        text = self.text_prompt.format(class_name=self.label_lookup[class_id])
        
        output["image"] = image_tensor
        output["text"] = text
        output["class_id"] = class_id
        output["image_id"] = int(item["__key__"].split("_")[-1])

        return output

    def setup(self, stage: str = None):
        print(f"Setting up datamodule using {len(self.data_path)} tar files")
        self.dataset = wds.WebDataset(
            self.data_path, 
            cache_size=10 ** 10, 
            handler=wds.handlers.warn_and_continue
        )

        if self.shuffle_train:
            self.dataset = self.dataset.shuffle(1000)

        self.dataset = self.dataset.map(self._preprocess, handler=wds.handlers.warn_and_continue)

    def train_dataloader(self):
        print(f"WARNING: train_dataloader is the same as val_dataloader, left for compatibility")

        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size,  
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size,  
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
        )
    
    def test_dataloader(self):
        print(f"WARNING: test_dataloader is the same as val_dataloader, left for compatibility")

        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size,  
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
        )
    
    @property
    def class_prompts(self):
        labels = self.label_lookup.values()
        return [self.text_prompt.format(class_name=name) for name in labels]
