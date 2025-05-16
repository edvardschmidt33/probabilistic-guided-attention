import numpy as np
import torch
import pytorch_lightning as L
from typing import Tuple
import dask.dataframe as dd
import pandas as pd
from typing import Sequence
from datasets import load_dataset_builder
from collections import OrderedDict
from PIL import Image
import io

from .common import default_collate_fn, default_transform


def select_classes_subset(ds_builder, num_classes:int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    classes = ds_builder.info.features['label'].names
    classes = np.array(classes)
    class_indices = np.arange(len(classes))
    
    np.random.seed(seed)
    indices = np.random.choice(class_indices, num_classes, replace=False)
    indices = np.sort(indices)

    return classes[indices], indices

def prepare_data(trainval_path: str, test_path: str, classes_seed: int, num_classes: int):
    ds_builder = load_dataset_builder("ILSVRC/imagenet-1k")
    subset_classes, subset_classes_ids = select_classes_subset(ds_builder, num_classes, classes_seed)

    ddf_trainval = dd.read_parquet(trainval_path)
    ddf_trainval = ddf_trainval[ddf_trainval.cls.isin(set(subset_classes_ids))]

    # take 80% of the data for training
    df_trainval = ddf_trainval.compute()
    df_train = df_trainval.iloc[:int(0.8*len(ddf_trainval))]
    df_val = df_trainval.iloc[int(0.8*len(ddf_trainval)):]

    ddf_test = dd.read_parquet(test_path)
    ddf_test = ddf_test[ddf_test.cls.isin(set(subset_classes_ids))]
    df_test = ddf_test.compute()

    return df_train, df_val, df_test, subset_classes, subset_classes_ids


def get_wnid(row):
    return row['json']['filename'].split('/')[0]

def prepare_data_wnids(trainval_path: str, test_path: str, class_wnids: Sequence[str]):
    ds_builder = load_dataset_builder("ILSVRC/imagenet-1k")
    classes, _ = select_classes_subset(ds_builder, 1000, 0)

    ddf_trainval = dd.read_parquet(trainval_path)

    ddf_trainval['wnid'] = ddf_trainval.map_partitions(lambda df: df.apply(get_wnid, axis=1), meta=('wnid', 'str'))
    ddf_trainval = ddf_trainval[ddf_trainval.wnid.isin(set(class_wnids))]

    # take 80% of the data for training
    df_trainval = ddf_trainval.compute()
    df_train = df_trainval.iloc[:int(0.8*len(ddf_trainval))]
    df_val = df_trainval.iloc[int(0.8*len(ddf_trainval)):]

    ddf_test = dd.read_parquet(test_path)
    ddf_test['wnid'] = ddf_test.map_partitions(lambda df: df.apply(get_wnid, axis=1), meta=('wnid', 'str'))
    ddf_test = ddf_test[ddf_test.wnid.isin(set(class_wnids))]
    df_test = ddf_test.compute()

    subset_classes_ids = sorted(df_train.cls.unique())
    subset_classes = [classes[i] for i in subset_classes_ids]

    return df_train, df_val, df_test, subset_classes, subset_classes_ids


class Imagenet1kDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        df: pd.DataFrame, 
        text_prompt: str, 
        class_id_to_class_name: dict, 
        class_id_to_subset_class_id: dict,
        transform=None,
    ):
        self._df = df
        self._transform = transform
        self._text_prompt = text_prompt
        self._class_id_to_class_name = class_id_to_class_name
        self._class_id_to_subset_class_id = class_id_to_subset_class_id
        self._label_names = list(self._class_id_to_class_name.values())

    def __len__(self):
        return len(self._df)
    
    def __getitem__(self, index):
        row = self._df.iloc[index]

        bytes = row.jpg['bytes']
        img = Image.open(io.BytesIO(bytes))
        if self._transform:
            img = self._transform(img)

        class_id = row['cls']
        text = self._text_prompt.format(
            class_name=self._class_id_to_class_name[class_id],
        )
        return dict(
            image=img,
            text=text,
            class_id=self._class_id_to_subset_class_id[class_id],
            image_id=index,
        )

class Imagenet1kDataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'imagenet'

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
            class_seed: int = 42,
            num_classes: int = 100,
            class_wids: Sequence[str] = None,
            # few shot parameters
            shots_per_class: int = 10,
            use_few_shot: bool = False,
            few_shot_sample_seed: int = 42,
        ):
        if use_few_shot:
            raise ValueError("Few shot sampling is not supported for ImageNet")

        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.text_prompt = text_prompt
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.shuffle_train = shuffle_train
        self.subset_indices = subset_indices
        self.class_seed = class_seed
        self.num_classes = num_classes
        self.class_wids = class_wids

        if class_wids is not None:
            print("Using custom class WIDs: this will override the num_classes parameter")

    def setup(self, stage: str = None):

        if self.class_wids is None:
            df_train, df_val, df_test, classes, class_ids = prepare_data(
                self.data_dir / "train",
                self.data_dir / "validation",
                classes_seed=self.class_seed,
                num_classes=self.num_classes,
            )
        else:
            df_train, df_val, df_test, classes, class_ids = prepare_data_wnids(
                self.data_dir / "train",
                self.data_dir / "validation",
                class_wnids=self.class_wids,
            )

        classes = [c.split(",")[0] for c in classes]

        class_id_to_class_name = OrderedDict(zip(class_ids, classes))
        class_id_to_subset_class_id = OrderedDict(zip(class_ids, range(len(class_ids))))
        
        self.train_ds = Imagenet1kDataset(
            df_train, 
            text_prompt=self.text_prompt, 
            transform=self.train_transform,
            class_id_to_class_name=class_id_to_class_name,
            class_id_to_subset_class_id=class_id_to_subset_class_id,
        )
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.val_ds = Imagenet1kDataset(
            df_val, 
            text_prompt=self.text_prompt, 
            transform=self.test_transform,
            class_id_to_class_name=class_id_to_class_name,
            class_id_to_subset_class_id=class_id_to_subset_class_id,
        )
        self.test_ds = Imagenet1kDataset(
            df_test, 
            text_prompt=self.text_prompt, 
            transform=self.test_transform,
            class_id_to_class_name=class_id_to_class_name,
            class_id_to_subset_class_id=class_id_to_subset_class_id,
        )

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
    

class Imagenet100DataModule(Imagenet1kDataModule):
    def __init__(self, **kwargs):
        super().__init__(num_classes=100, **kwargs)

class Imagenet50DataModule(Imagenet1kDataModule):
    def __init__(self, **kwargs):
        super().__init__(num_classes=50, **kwargs)

class ImagenetRClassesDataModule(Imagenet1kDataModule):
    def __init__(self, **kwargs):
        super().__init__(
            num_classes=None, 
            class_wids=['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677'],
        **kwargs)

class Imagenet50StupidDataModule(Imagenet1kDataModule):
    DATASET_SUBDIR = 'imagenet-stupid'

    def __init__(self, **kwargs):
        super().__init__(num_classes=50, **kwargs)
