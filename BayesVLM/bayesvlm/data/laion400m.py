import torch
import webdataset as wds
import lightning as L
from PIL import Image
import io
import os
from typing import Sequence

from .common import default_transform


class Laion400mDataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'laion400m'

    def __init__(
            self, 
            data_dir: str,
            batch_size: int = 32, 
            num_workers: int = 4,
            text_prompt: str = None, # this is not used
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

        if subset_indices is not None:
            raise ValueError("Subset indices are not supported for this dataset")

    def _preprocess(self, item):
        output = {}
        image_data = item['jpg']
        image = Image.open(io.BytesIO(image_data))
        image_tensor = self.train_transform(image)
        output["image"] = image_tensor

        text = item['txt']
        caption = text.decode("utf-8")
        output["text"] = caption

        output["image_id"] = int(item["__key__"])

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
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size,  
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset, 
            batch_size=self.batch_size,  
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
        )
        