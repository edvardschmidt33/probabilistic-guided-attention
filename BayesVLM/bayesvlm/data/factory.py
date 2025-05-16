import os
import pathlib
from typing import Sequence

import pytorch_lightning as L
from dotenv import load_dotenv

from .common import default_transform

from .flowers102 import Flowers102DataModule
from .food101 import Food101DataModule
from .eurosat import EuroSATDataModule
from .cifar100 import CIFAR100DataModule
from .stanfordcars import StanfordCarsDataModule
from .dtd import DTDDataModule
from .sun397 import SUN397DataModule
from .oxfordpets import OxfordpetsDataModule
from .ucf101 import UCF101DataModule
from .cub import CUBDataModule

from .homeoffice import (
    HomeOfficeArtDataModule,
    HomeOfficeClipartDataModule,
    HomeOfficeProductDataModule,
    HomeOfficeRealWorldDataModule,
)
from .homeoffice_da import (
    HomeOfficeDAArtDataModule,
    HomeOfficeDAClipartDataModule,
    HomeOfficeDAProductDataModule,
    HomeOfficeDARealWorldDataModule,
)

from .imagenet_wds import ImagenetWDSModule
from .imagenet_1k import (
    Imagenet50DataModule,
    Imagenet100DataModule,
)

from .imagenet_r import ImagenetRDataModule
from .imagenet_sketch import ImagenetSketchDataModule

from .imagenet_da import (
    ImagenetDARenditionsDataModule,
    ImagenetDASketchDataModule,
)

# pretraining
from .laion400m import Laion400mDataModule


SUPPORTED_MODULES = {
    'laion400m': Laion400mDataModule,

    # downstream datasets
    'flowers102': Flowers102DataModule,
    'food101': Food101DataModule,
    'eurosat': EuroSATDataModule,
    'cifar100': CIFAR100DataModule,
    'stanfordcars': StanfordCarsDataModule,
    'dtd': DTDDataModule,
    'sun397': SUN397DataModule,
    'oxfordpets': OxfordpetsDataModule,
    'ucf101': UCF101DataModule,
    'cub': CUBDataModule,

    # homeoffice datasets
    'homeoffice-art': HomeOfficeArtDataModule,
    'homeoffice-clipart': HomeOfficeClipartDataModule,
    'homeoffice-product': HomeOfficeProductDataModule,
    'homeoffice-realworld': HomeOfficeRealWorldDataModule,

    'homeoffice-da-art': HomeOfficeDAArtDataModule,
    'homeoffice-da-clipart': HomeOfficeDAClipartDataModule,
    'homeoffice-da-product': HomeOfficeDAProductDataModule,
    'homeoffice-da-realworld': HomeOfficeDARealWorldDataModule,

    # imagenet datasets
    'imagenet-val-wds': ImagenetWDSModule,
    'imagenet-100': Imagenet100DataModule,
    'imagenet-50': Imagenet50DataModule,

    'imagenet-r': ImagenetRDataModule,
    'imagenet-sketch': ImagenetSketchDataModule,

    'imagenet-da-r': ImagenetDARenditionsDataModule,
    'imagenet-da-sketch': ImagenetDASketchDataModule,
}

class DataModuleFactory:
    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 4, 
        text_prompt: str = "An image of a {class_name}",
        train_transform=default_transform(image_size=244),
        test_transform=default_transform(image_size=244),
        shuffle_train: bool = True,
        base_path: str = None,
        shots_per_class: int = 10,
        use_few_shot: bool = False,
        few_shot_sample_seed: int = 42,
    ):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        
        self.shots_per_class = shots_per_class
        self.use_few_shot = use_few_shot
        self.few_shot_sample_seed = few_shot_sample_seed

        self.base_path = base_path
        if self.base_path is None:
            load_dotenv()
            self.base_path = os.getenv("DATA_BASE_DIR")
            

    def create(self, dataset_name: str, subset_indices: Sequence[int] = None) -> L.LightningDataModule:
        if dataset_name in SUPPORTED_MODULES:
            module = SUPPORTED_MODULES[dataset_name]
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        data_dir = pathlib.Path(self.base_path) / module.DATASET_SUBDIR
        
        if self.use_few_shot:
            return module(
                data_dir=data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                text_prompt=self.text_prompt,
                train_transform=self.train_transform,
                test_transform=self.test_transform,
                shuffle_train=self.shuffle_train,
                subset_indices=subset_indices,
                shots_per_class = self.shots_per_class,
                few_shot_sample_seed = self.few_shot_sample_seed,
                use_few_shot = self.use_few_shot
            )
        else:
            return module(
                data_dir=data_dir,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                text_prompt=self.text_prompt,
                train_transform=self.train_transform,
                test_transform=self.test_transform,
                shuffle_train=self.shuffle_train,
                subset_indices=subset_indices,
            )

