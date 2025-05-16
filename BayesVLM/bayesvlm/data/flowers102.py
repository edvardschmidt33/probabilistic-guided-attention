from typing import Sequence
import torch
from pathlib import Path
import pytorch_lightning as L
from torchvision.datasets import Flowers102
from collections import defaultdict
import numpy as np
from .common import default_collate_fn, default_transform

CLASS_ID_TO_NAME = {"21": "fire lily", "3": "canterbury bells", "45": "bolero deep blue", "1": "pink primrose", "34": "mexican aster", "27": "prince of wales feathers", "7": "moon orchid", "16": "globe-flower", "25": "grape hyacinth", "26": "corn poppy", "79": "toad lily", "39": "siam tulip", "24": "red ginger", "67": "spring crocus", "35": "alpine sea holly", "32": "garden phlox", "10": "globe thistle", "6": "tiger lily", "93": "ball moss", "33": "love in the mist", "9": "monkshood", "102": "blackberry lily", "14": "spear thistle", "19": "balloon flower", "100": "blanket flower", "13": "king protea", "49": "oxeye daisy", "15": "yellow iris", "61": "cautleya spicata", "31": "carnation", "64": "silverbush", "68": "bearded iris", "63": "black-eyed susan", "69": "windflower", "62": "japanese anemone", "20": "giant white arum lily", "38": "great masterwort", "4": "sweet pea", "86": "tree mallow", "101": "trumpet creeper", "42": "daffodil", "22": "pincushion flower", "2": "hard-leaved pocket orchid", "54": "sunflower", "66": "osteospermum", "70": "tree poppy", "85": "desert-rose", "99": "bromelia", "87": "magnolia", "5": "english marigold", "92": "bee balm", "28": "stemless gentian", "97": "mallow", "57": "gaura", "40": "lenten rose", "47": "marigold", "59": "orange dahlia", "48": "buttercup", "55": "pelargonium", "36": "ruby-lipped cattleya", "91": "hippeastrum", "29": "artichoke", "71": "gazania", "90": "canna lily", "18": "peruvian lily", "98": "mexican petunia", "8": "bird of paradise", "30": "sweet william", "17": "purple coneflower", "52": "wild pansy", "84": "columbine", "12": "colt's foot", "11": "snapdragon", "96": "camellia", "23": "fritillary", "50": "common dandelion", "44": "poinsettia", "53": "primula", "72": "azalea", "65": "californian poppy", "80": "anthurium", "76": "morning glory", "37": "cape flower", "56": "bishop of llandaff", "60": "pink-yellow dahlia", "82": "clematis", "58": "geranium", "75": "thorn apple", "41": "barbeton daisy", "95": "bougainvillea", "43": "sword lily", "83": "hibiscus", "78": "lotus", "88": "cyclamen", "94": "foxglove", "81": "frangipani", "74": "rose", "89": "watercress", "73": "water lily", "46": "wallflower", "77": "passion flower", "51": "petunia"}

class Flowers102WithLabels(Flowers102):
    class_id_to_name = {int(k): v for k, v in CLASS_ID_TO_NAME.items()}

    def __init__(self, *args, **kwargs):
        self._text_prompt = kwargs['text_prompt']
        sorted_classes = sorted([(class_id, name) for class_id, name in self.class_id_to_name.items()], key=lambda x: x[0])
        self.classes = [name for _, name in sorted_classes]

        if 'classbalanced' in kwargs:
            self.classbalanced = kwargs['classbalanced']
            del kwargs['classbalanced']
        else:
            self.classbalanced = False
        
        self.use_few_shot = kwargs['use_few_shot']
        if self.use_few_shot:
            self.shots_per_class = kwargs['shots_per_class']
            self.few_shot_sample_seed = kwargs['few_shot_sample_seed']
            del kwargs['shots_per_class']
            del kwargs['few_shot_sample_seed']
        del kwargs['use_few_shot']
        
        del kwargs['text_prompt']

        super().__init__(*args, **kwargs)

        if self.classbalanced:
            index_class_pairs = defaultdict(list)
            for i in range(super().__len__()):
                _, class_id = super().__getitem__(i)

                if len(index_class_pairs[class_id]) < 20:
                    index_class_pairs[class_id].append(i)
            
            self.indices = []
            for i in range(102):
                self.indices.extend(index_class_pairs[i])
        else:
            self.indices = list(range(super().__len__()))

            
            if self.use_few_shot:
                                
                # get the index for each class
                self.class_index = defaultdict(list)
                
                for img_index in range(super().__len__()):
                    class_id = self._labels[img_index]
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
        ds_index = self.indices[index]

        img, class_id = super().__getitem__(ds_index)

        text = self._text_prompt.format(
            class_name=self.class_id_to_name[class_id+1],
        )

        return dict(
            image=img,
            text=text,
            class_id=class_id,
            image_id=index,
        )

class Flowers102DataModule(L.LightningDataModule):
    DATASET_SUBDIR = 'flowers102'

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
            self.train_ds = Flowers102WithLabels(
                self.data_dir,
                split='train',
                transform=self.train_transform,
                download=True,
                text_prompt=self.text_prompt,
                use_few_shot = True,
                shots_per_class = self.shots_per_class,
                few_shot_sample_seed = self.few_shot_sample_seed
            )
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.train_ds.selected_data)
        else:
            self.train_ds = Flowers102WithLabels(
                self.data_dir,
                split='train',
                transform=self.train_transform,
                download=True,
                text_prompt=self.text_prompt,
                use_few_shot = False,
            )
        if self.subset_indices is not None:
            self.train_ds = torch.utils.data.Subset(self.train_ds, self.subset_indices)

        self.val_ds = Flowers102WithLabels(
            self.data_dir,
            split='val',
            transform=self.test_transform,
            download=True,
            text_prompt=self.text_prompt,
            use_few_shot = False,
        )

        self.test_ds = Flowers102WithLabels(
            self.data_dir,
            split='test',
            transform=self.test_transform,
            download=True,
            text_prompt=self.text_prompt,
            classbalanced=False,
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
        if self.use_few_shot:
            return [self.text_prompt.format(class_name=name) for name in self.test_ds._label_names]
        else:
            return [self.text_prompt.format(class_name=name) for name in self.train_ds._label_names]