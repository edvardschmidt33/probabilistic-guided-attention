import torch
import pytorch_lightning as L
import re
from typing import Sequence
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

from .common import default_collate_fn, default_transform


def _label_names_from_readme(readme_path):
    with open(readme_path, "r") as f:
        lines = [x.split() for x in f.readlines() if re.match(r"n\d+", x)]
        if len(lines) != 200:
            raise ValueError("Expected 200 lines with label information in the README file")
        
    label_names = [x[1].strip() for x in lines]
    dir_to_label_idx = {x[0].strip(): i for i, x in enumerate(lines)}

    return label_names, dir_to_label_idx

def _find_all_images(data_dir: Path, dir_to_label_idx: dict):
    all_images = []
    for dir_name, label_idx in dir_to_label_idx.items():
        dir_path = data_dir / dir_name
        for file in dir_path.iterdir():
            if file.suffix == '.jpg':
                all_images.append((dir_path / file, label_idx))
    
    # sort by file name to ensure reproducibility
    all_images = sorted(all_images, key=lambda x: x[0])

    return all_images


class ImagenetRDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        image_label_pairs: Sequence[tuple],
        label_names: Sequence[str],
        text_prompt: str, 
        transform=None,
    ):
        self._image_label_pairs = image_label_pairs
        self._label_names = label_names
        self._text_prompt = text_prompt
        self._transform = transform

    def __len__(self):
        return len(self._image_label_pairs)
   
    def __getitem__(self, idx):
        image_path, class_id = self._image_label_pairs[idx]

        text = self._text_prompt.format(
            class_name=self._label_names[class_id]
        )

        image = Image.open(image_path)
        if self._transform is not None:
            image = self._transform(image)
            
        return dict(image=image, text=text, class_id=class_id, image_id=idx)
       
    

class ImagenetRDataModule(L.LightningDataModule):
    DATASET_SUBDIR = "imagenet-r"
    
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
        self.data_dir = Path(data_dir)
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices

    def setup(self, stage: str = None):
        readme_file = self.data_dir / "README.txt"
        label_names, dir_to_label_idx = _label_names_from_readme(readme_file)
        image_label_pairs = _find_all_images(self.data_dir, dir_to_label_idx)

        # create train/test split
        split_idx = int(0.75 * len(image_label_pairs))
        np.random.seed(0)
        perm = np.random.permutation(len(image_label_pairs))

        train_pairs = [image_label_pairs[i] for i in perm[:split_idx]]
        test_pairs = [image_label_pairs[i] for i in perm[split_idx:]]

        # create train / val split
        train_pairs, val_pairs = train_test_split(train_pairs, test_size=0.2, random_state=0)


        self.train_ds = ImagenetRDataset(
            train_pairs, 
            label_names=label_names,
            text_prompt=self.text_prompt, 
            transform=self.train_transform,
        )
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.val_ds = ImagenetRDataset(
            val_pairs, 
            label_names=label_names,
            text_prompt=self.text_prompt, 
            transform=self.test_transform,
        )

        self.test_ds = ImagenetRDataset(
            test_pairs, 
            label_names=label_names,
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
            shuffle=False,
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
        return [self.text_prompt.format(class_name=x) for x in self.train_ds._label_names]
