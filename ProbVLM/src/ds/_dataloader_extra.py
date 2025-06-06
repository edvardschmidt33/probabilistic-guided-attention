"""libaray for multi-modal dataset loaders.

Acknowledgements:
`image_to_caption_collate_fn` is based on
https://github.com/yalesong/pvse/blob/master/data.py
"""
import os
from os.path import join as ospj
from omegaconf import OmegaConf

import numpy as np
from PIL import Image
from typing import Union, List

import torch
from torch.utils.data import DataLoader, random_split
import torchvision as tv
from torchvision import datasets, transforms

from ds.flickr import FlickrCap, FlickrBboxes
from ds.coco import CocoCaptionsCap, CocoBboxes
from ds.cub import CUBCaption, CUBSampler
from ds.fashion200k import Fashion200k,BaseDataset
from ds.flo import FLOCaption, FLOSampler
from ds.food import FoodSubset
from ds.vocab import Vocabulary
#from datasets._transforms import imagenet_transform
from ds._transforms import  caption_transform
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

from transformers import CLIPTokenizer

# Initialize the tokenizer outside the function (globally or in the class)
tokenizer_clip = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch16")


def tokenize_labels(label):
        return tokenizer_clip(label, padding="max_length", truncation=True, max_length=16, return_tensors="pt")


try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

# imagenet_transform = tv.transforms.Compose([
#      tv.transforms.Resize(224, interpolation=BICUBIC),
#      tv.transforms.CenterCrop(224),
#      _convert_image_to_rgb,
#      tv.transforms.ToTensor(),
#      tv.transforms.Normalize(
#          (0.48145466, 0.4578275, 0.40821073), 
#          (0.26862954, 0.26130258, 0.27577711)),
# ])
def imagenet_normalize():
    """Standard ImageNet normalize transform
    """
#     return transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                 std=[0.229, 0.224, 0.225])
    return transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711))
def imagenet_transform_fn(resize_size=224,
                       crop_size=224,
                       random_resize_crop=False,
                       random_erasing_prob=0.0,
                       custom_transforms=None):
    """Standard ImageNet transform with resize/crop/normalize.

    Args:
        resize_size (int, Default: 256): resize for validation
            (only used when random_resize_crop is False).
        crop_size (int, Default: 224): final crop size.
        random_resize_crop (bool, Default: False): if True, use random transform (for training),
            if False, use center crop (for validation).
        custom_transforms (list of transform, Default: None): additional transforms.
    """
    if custom_transforms is not None:
        if not isinstance(custom_transforms, list):
            raise TypeError(f'custom_transforms should be list, not {type(custom_transforms)}')
    transform = []
    if random_resize_crop:
        transform.append(transforms.RandomResizedCrop(crop_size))
        transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.Resize(resize_size))
        transform.append(transforms.CenterCrop(crop_size))
    transform.append(transforms.ToTensor())
    transform.append(imagenet_normalize())

    if custom_transforms:
        transform.extend(custom_transforms)

#     if random_erasing_prob > 0:
#         print(f'adding cutout {random_erasing_prob}')
#         transform.append(RandomErasing(random_erasing_prob,
#                                        mode='const',
#                                        max_count=1, num_splits=0, device='cpu'))
    #transform.append(RandomErasing(random_erasing_prob,
    #                               mode='const',
    #                               max_count=1, num_splits=0, device='cpu'))

    transform = transforms.Compose(transform)
    #print("Transform Called")
    return transform



imagenet_transform = imagenet_transform_fn()
_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def image_to_caption_collate_fn(data):
    """Build mini-batch tensors from a list of (image, sentence) tuples.
    Args:
      data: list of (image, sentence) tuple.
        - image: torch tensor of shape (3, 256, 256) or (?, 3, 256, 256).
        - sentence: torch tensor of shape (?); variable length.

    Returns:
      images: torch tensor of shape (batch_size, 3, 256, 256) or
              (batch_size, padded_length, 3, 256, 256).
      targets: torch tensor of shape (batch_size, padded_length).
      lengths: list; valid length for each padded sentence.
    """
    # Sort a data list by sentence length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, sentences, ann_ids, image_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge sentences (convert tuple of 1D tensor to 2D tensor)
    cap_lengths = [len(cap) for cap in sentences]
    targets = torch.zeros(len(sentences), max(cap_lengths)).long()
    for i, cap in enumerate(sentences):
        end = cap_lengths[i]
        targets[i, :end] = cap[:end]

    cap_lengths = torch.Tensor(cap_lengths).long()
    return images, targets, cap_lengths, ann_ids, image_ids


def load_vocab(vocab_path):
    if isinstance(vocab_path, str):
        vocab = Vocabulary()
        vocab.load_from_pickle(vocab_path)
    else:
        vocab = vocab_path
    return vocab


def prepare_foodredmeat_dataloaders(transform, root, cfg, seed=42):  # Added seed argument
    cfg = OmegaConf.load(cfg)
    
    # Create train and test datasets
    train_dataset = FoodSubset(
        root=root,
        cfg=cfg,
        transform=transform,
        target_transform=tokenize, #_labels,  # Use the tokenization function for target_transform
        split='train-redmeat'
    )
    print(f"Total images in train split: {len(train_dataset)}")
    print(f"First image path: {train_dataset.imgs[0][0]}")
    
    test_dataset = FoodSubset(
        root=root,
        cfg=cfg,
        transform=transform,
        target_transform=tokenize, #_labels,  # Use the tokenization function for target_transform
        split='test-redmeat'
    )
    print(f"Total images in test split: {len(test_dataset)}")
    print(f"First image path: {test_dataset.imgs[0][0]}")
    
    # Split the training dataset into train and validation sets
    val_size = int(0.2 * len(train_dataset))  # 20% for validation
    train_size = len(train_dataset) - val_size
    
    # Set random seed for reproducibility
    generator = torch.manual_seed(seed)

    # Split dataset with fixed generator for reproducibility
    train_data, val_data = random_split(train_dataset, [train_size, val_size], generator=generator)

    # Create DataLoader for train, validation, and test datasets
    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg['DATA']['BATCH_SIZE'],
        num_workers=cfg['DATA']['NUM_WORKERS'],
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg['DATA']['BATCH_SIZE'],
        num_workers=cfg['DATA']['NUM_WORKERS'],
        shuffle=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg['DATA']['BATCH_SIZE'],
        num_workers=cfg['DATA']['NUM_WORKERS'],
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader

def prepare_foodmeat_dataloaders(transform, root, cfg, seed=42):  # Added seed argument
    cfg = OmegaConf.load(cfg)
    # Create train and test datasets
    train_dataset = FoodSubset(
        root=root,
        cfg=cfg,
        transform=transform,
        target_transform=tokenize,  # Use the tokenization function for target_transform
        split='train-meat'
    )
    #print(f"Total images in test split: {len(train_dataset)}")
    #print(f"First image path: {train_dataset.imgs[0][0]}")
    
    test_dataset = FoodSubset(
        root=root,
        cfg=cfg,
        transform=transform,
        target_transform=tokenize,  # Use the tokenization function for target_transform
        split='test-meat'
    )
    #print(f"Total images in test split: {len(test_dataset)}")
    #print(f"First image path: {test_dataset.imgs[0][0]}")
    # Split the training dataset into train and validation sets
    val_size = int(0.2 * len(train_dataset))  # 20% for validation
    train_size = len(train_dataset) - val_size
    
    # Set random seed for reproducibility
    generator = torch.manual_seed(seed)

    # Split dataset with fixed generator for reproducibility
    train_data, val_data = random_split(train_dataset, [train_size, val_size], generator=generator)

    # Create DataLoader for train, validation, and test datasets
    train_dataloader = DataLoader(
        train_data,
        batch_size=cfg['DATA']['BATCH_SIZE'],
        num_workers=cfg['DATA']['NUM_WORKERS'],
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_data,
        batch_size=cfg['DATA']['BATCH_SIZE'],
        num_workers=cfg['DATA']['NUM_WORKERS'],
        shuffle=False
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg['DATA']['BATCH_SIZE'],
        num_workers=cfg['DATA']['NUM_WORKERS'],
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader


def _get_cub_file_paths(dataset_name, dataset_root, caption_root):
    """Select proper train / val classes and omit id files.
    The split is based on CVPR'17 Zero-Shot Learning -- The Good, the Bad and the Ugly
    See more details in
    https://arxiv.org/abs/1703.04394
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly
    Args:
      dataset_name: name of dataset
        - cub_trainval{idx} (idx in [1, 2, 3]):
            3-fold validation splits to search hyperparameters.
            Each split contains 100 train classes / 50 validation classes.
        - cub:
            The final split used for the final benchmark.
            This split conntains 150 train classes / 50 unseen test classes (not in trainval)
            
    """
    print(dataset_name)
    if dataset_name == 'cub_trainval1':
        train_classes = './ds/annotations/cub/trainclasses1.txt'
        val_classes = './ds/annotations/cub/valclasses1.txt'
        omit_ids = './ds/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub_trainval2':
        train_classes = './ds/annotations/cub/trainclasses2.txt'
        val_classes = './ds/annotations/cub/valclasses2.txt'
        omit_ids = './ds/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub_trainval3':
        train_classes = './ds/annotations/cub/trainclasses3.txt'
        val_classes = './ds/annotations/cub/valclasses3.txt'
        omit_ids = './ds/annotations/cub/seen_test_images.txt'
    elif dataset_name == 'cub' or 'waterbird_1.0_forest2water2':
        train_classes = '../ProbVLM/src/ds/annotations/cub/trainvalclasses.txt'
        val_classes = '../ProbVLM/src/ds/annotations/cub/testclasses.txt'
        omit_ids = '../ProbVLM/src/ds/annotations/cub/seen_test_images.txt'
    else:
        raise ValueError(f'Invalide dataset_name: {dataset_name}')

    #image_root = os.path.join(dataset_root, 'images/')
    image_root = dataset_root
    
    return train_classes, val_classes, omit_ids, image_root, caption_root


def _get_cub_loader_extra(image_root, caption_root,
                    data_classes, 
                    #vocab,
                    num_workers,
                    batch_size=64,
                    train=False,
                    omit_ids=None,
                    ids=None,
                    cutout_prob=0.0,
                    caption_drop_prob=0.0):


    #_image_transform = imagenet_transform(
    #    random_resize_crop=train,
    #    random_erasing_prob=cutout_prob,
    #)

    _caption_transform = tokenize

    cub_dataset = CUBCaption(image_root, caption_root,
                             data_classes,
                             #transform=_image_transform,
                             imagenet_transform,
                             #caption_transform(vocab, caption_drop_prob),
                             omit_ids=omit_ids,
                             target_transform=_caption_transform,
                             ids=ids)
    if train:
        """
        sampler = CUBSampler(cub_dataset, len(cub_dataset.target_classes))
        dataloader = DataLoader(cub_dataset, batch_sampler=sampler,
                                num_workers=num_workers,
#                                collate_fn=image_to_caption_collate_fn,
                                pin_memory=True)"""
        dataloader = DataLoader(cub_dataset, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers,
#                                collate_fn=image_to_caption_collate_fn,
                                pin_memory=True)
    else:
        dataloader = DataLoader(cub_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
#                                collate_fn=image_to_caption_collate_fn,
                                pin_memory=True)
    print(f'Loading CUB Caption: n_images {cub_dataset.n_images} n_captions {len(cub_dataset.targets)}...')
    return dataloader


def prepare_cub_dataloaders_extra(dataloader_config,
                            dataset_root,
                            caption_root,
                            dataset_name='cub',
#                            vocab_path='./vocabs/cub_vocab.pkl',
                            num_workers=6):
    """Prepare CUB Caption train / val / test dataloaders
    CUB Caption loader has a fixed batch size
    - train loader: # classes (trainval = 100, full = 150)
    - test loader: 64 (hard coded at L#203)
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_name (str): name of dataset
            - cub_trainval{idx} (idx in [1, 2, 3]):
                3-fold validation splits to search hyperparameters.
                Each split contains 100 train classes / 50 validation classes.
            - cub:
                The final split used for the final benchmark.
                This split conntains 150 train classes / 50 unseen test classes (not in trainval)
        dataset_root (str): root of your CUB images (see README.md for detailed dataset hierarchy)
        caption_root (str): root of your CUB captions (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/cub_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "val_in"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
  #  vocab = load_vocab(vocab_path)
    train_classes, val_classes, omit_ids, image_root, caption_root = _get_cub_file_paths(
        dataset_name, dataset_root, caption_root)

    cutout_prob = dataloader_config.get('random_erasing_prob', 0.0)
    caption_drop_prob = dataloader_config.get('caption_drop_prob', 0.0)

    dataloaders = {}
    print('preparing train', flush=True)
    dataloaders['train'] = _get_cub_loader_extra(
        image_root, caption_root,
        train_classes,
        #vocab, 
        num_workers,
        train=True,
        omit_ids=omit_ids,
        cutout_prob=cutout_prob,
        caption_drop_prob=caption_drop_prob,
    )
    print('preparing test', flush=True)
    dataloaders['test'] = _get_cub_loader_extra(
        image_root, caption_root,
        val_classes,
        #vocab, 
        num_workers,
        train=False,
    )
    print('preparing val', flush=True)
    dataloaders['val'] = _get_cub_loader_extra(
        image_root, caption_root,
        train_classes,
 #       vocab, 
        num_workers,
        train=False,
        ids=omit_ids
    )

    return dataloaders


def load_mnist_data_loader(data_dir, batch_size=64, shuffle=True, num_workers=4):
    val_split=0.2
    # Define transformations for the MNIST dataset (resizing and normalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize to 0-1 range, mean and std for MNIST
    ])
    
    # Load the full MNIST training dataset
    full_train_dataset = datasets.MNIST(root=data_dir, train=True, transform=transform, download=True)
    
    # Split the dataset into training and validation sets
    total_train_samples = len(full_train_dataset)
    val_size = int(val_split * total_train_samples)
    train_size = total_train_samples - val_size
    
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
    
    # Load the MNIST test dataset
    test_dataset = datasets.MNIST(root=data_dir, train=False, transform=transform, download=True)
    
    # Create data loaders for training, validation, and test sets
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader


def _get_coco_loader(flag,image_root,
                     annotation_path,
                     ids, 
#vocab,
                     num_workers,
                     batch_size=64,
                     train=False,
                     extra_ids=None,
                     extra_annotation_path=None,
                     cutout_prob=0.0):
    #_image_transform = imagenet_transform(
    #    random_resize_crop=train,
    #    random_erasing_prob=cutout_prob,
    #)
    _caption_transform = tokenize

    coco_dataset = CocoCaptionsCap(flag,image_root, annotation_path,
                                   extra_annFile=extra_annotation_path,
                                   ids=ids,
                                   extra_ids=extra_ids,
                                   transform=imagenet_transform,
                                   target_transform=_caption_transform)

    dataloader = DataLoader(coco_dataset,
                            batch_size=batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            pin_memory=True)
    print(f'Loading COCO Caption: n_images {coco_dataset.n_images} n_captions {len(coco_dataset)}...')
    return dataloader


def _get_coco_file_paths(dataset_root):
    """Select proper train / val classes and omit id files.
    """
    train_ids = np.load('../ProbVLM/src/ds/annotations/coco/coco_train_ids_extra.npy')
    train_extra_ids = np.load('../ProbVLM/src/ds/annotations/coco/coco_restval_ids.npy')
    val_ids = np.load('../ProbVLM/src/ds/annotations/coco/coco_dev_ids_extra.npy')#[:5000]
    te_ids = np.load('../ProbVLM/src/ds/annotations/coco/coco_test_ids.npy')

# Paths to the image folders
#    train_folder = os.path.join(dataset_root, 'images/train2014')
#    val_folder = os.path.join(dataset_root, 'images/val2014')
    
    # Extracting IDs from the image filenames in the train folder
#    train_ids = [int(fname.split('_')[-1].split('.')[0]) for fname in os.listdir(train_folder) if fname.endswith('.jpg')]
    
    # Extracting IDs from the image filenames in the validation folder
#    val_ids = [int(fname.split('_')[-1].split('.')[0]) for fname in os.listdir(val_folder) if fname.endswith('.jpg')]
    

#     image_root = os.path.join(dataset_root, 'images/trainval35k')
    image_root = os.path.join(dataset_root, 'images/total') # train + valid
    #train_ann = os.path.join(dataset_root, 'captions_train2014.json')
    #val_ann = os.path.join(dataset_root, 'captions_val2014.json')
    train_ann = os.path.join(dataset_root, 'captions_train2014_extra.json')
    val_ann = os.path.join(dataset_root, 'captions_val2014_extra.json')
    
    return train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann

def prepare_coco_dataloaders_extra(dataloader_config,
                             dataset_root,
                             vocab_path=None,
                             num_workers=32):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    batch_size = dataloader_config.batch_size
    tr_cutout_prob = dataloader_config.random_erasing_prob
    eval_batch_size = dataloader_config.batch_size
    traindata_shuffle = dataloader_config.traindata_shuffle

#    vocab = load_vocab(vocab_path)
    train_ids, train_extra_ids, val_ids, te_ids, image_root, train_ann, val_ann = _get_coco_file_paths(dataset_root)
    #print('train_ids', train_ids)

    dataloaders = {}

    dataloaders['train'] = _get_coco_loader('train',
        image_root, train_ann, train_ids, 
#vocab,
        num_workers=num_workers, batch_size=batch_size,
        train=traindata_shuffle,
        #extra_annotation_path=val_ann,
        #extra_ids=train_extra_ids,
        cutout_prob=tr_cutout_prob,
    )

    dataloaders['val'] = _get_coco_loader('val',
        image_root, val_ann, val_ids,
# vocab,/
        num_workers=num_workers, batch_size=eval_batch_size,
        train=False,
    )

    dataloaders['test'] = _get_coco_loader('test',
        image_root, val_ann, te_ids,
# vocab,
        num_workers=num_workers, batch_size=eval_batch_size,
        train=False,
    )
	
    return dataloaders

def _get_flickr_file_paths(dataset_root):
    image_root = ospj(dataset_root, 'flickr30k_images')
    train_ids_path = './ds/annotations/flickr/train.txt'
    valid_ids_path = './ds/annotations/flickr/val.txt'
    test_ids_path = './ds/annotations/flickr/test.txt'
    return image_root, train_ids_path, valid_ids_path, test_ids_path

def _get_flickr_loader(image_root,
                     image_ids_path,
                     num_workers,
                     batch_size=64,
                     train=False,
                     cutout_prob=0.0):
    #_image_transform = imagenet_transform(
    #    random_resize_crop=train,
    #    random_erasing_prob=cutout_prob,
    #)
    _caption_transform = tokenize

    flickr_dataset = FlickrCap(image_root, image_ids_path,
                             transform=imagenet_transform,
                             target_transform=_caption_transform)

    dataloader = DataLoader(flickr_dataset,
                            batch_size=batch_size,
                            shuffle=train,
                            num_workers=num_workers,
                            pin_memory=True)
    print(f'Loading Flickr Caption: n_captions {len(flickr_dataset)}...')
    return dataloader

def prepare_flickr_dataloaders(dataloader_config,
                               dataset_root,
                               vocab_path='./vocabs/coco_vocab.pkl',
                               num_workers=32):
    """Prepare MS-COCO Caption train / val / test dataloaders
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_root (str): root of your MS-COCO dataset (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/coco_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "te"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    batch_size = dataloader_config.batch_size
    tr_cutout_prob = dataloader_config.random_erasing_prob
    eval_batch_size = dataloader_config.batch_size
    traindata_shuffle = dataloader_config.traindata_shuffle
    
    image_root, train_ids_path, valid_ids_path, test_ids_path = _get_flickr_file_paths(dataset_root)

    dataloaders = {}

    dataloaders['train'] = _get_flickr_loader(
        image_root,
        image_ids_path=train_ids_path,
        num_workers=num_workers, batch_size=batch_size,
        train=traindata_shuffle,
        cutout_prob=tr_cutout_prob,
    )

    dataloaders['val'] = _get_flickr_loader(
        image_root,
        image_ids_path=valid_ids_path,
        num_workers=num_workers, batch_size=eval_batch_size,
        train=False,
    )

    dataloaders['test'] = _get_flickr_loader(
        image_root,
        image_ids_path=test_ids_path,
        num_workers=num_workers, batch_size=eval_batch_size,
        train=False,
    )

    return dataloaders

def _get_fashion_loader(dataset_root,split='train',batch_size=128,num_workers=32):

    _caption_transform = tokenize

    fashion_dataset = Fashion200k(dataset_root,split,transform=imagenet_transform,target_transform=_caption_transform)

    if split == 'train':
        dataloader = DataLoader(fashion_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,pin_memory=True)
    else:
        dataloader = DataLoader(fashion_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,pin_memory=True)

    return dataloader


def prepare_fashion_dataloaders(dataloader_config,dataset_root,num_workers=32):

    batch_size = dataloader_config.batch_size

    dataloaders = {}

    dataloaders['train'] = _get_fashion_loader(dataset_root,split='train',batch_size=batch_size,num_workers=num_workers)

    dataloaders['test'] = _get_fashion_loader(dataset_root,split='test',batch_size=batch_size,num_workers=num_workers)

    return dataloaders


def _get_flo_file_paths(dataset_name, dataset_root, caption_root):
    """Select proper train / val classes and omit id files.
    The split is based on CVPR'17 Zero-Shot Learning -- The Good, the Bad and the Ugly
    See more details in
    https://arxiv.org/abs/1703.04394
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly
    Args:
      dataset_name: name of dataset
        - cub_trainval{idx} (idx in [1, 2, 3]):
            3-fold validation splits to search hyperparameters.
            Each split contains 100 train classes / 50 validation classes.
        - flo:
            The final split used for the final benchmark.
            This split conntains 82 train classes / 20 unseen test classes (not in trainval)
    """

    if dataset_name == 'flo':
        train_classes = './ds/annotations/flo/trainvalclasses.txt'
        val_classes = './ds/annotations/flo/testclasses.txt'

    else:
        raise ValueError(f'Invalide dataset_name: {dataset_name}')

    image_root = dataset_root

    return train_classes, val_classes, image_root, caption_root


def _get_flo_loader(image_root, caption_root,
                    data_classes,
                    num_workers,
                    batch_size=64,
                    train=False):



    _caption_transform = tokenize

    flo_dataset = FLOCaption(image_root, caption_root,
                             data_classes,
                             imagenet_transform,
                             target_transform=_caption_transform,)
    if train:
        sampler = FLOSampler(flo_dataset, len(flo_dataset.target_classes))
        dataloader = DataLoader(flo_dataset, batch_sampler=sampler,
                                num_workers=num_workers,
                                pin_memory=True)
    else:
        dataloader = DataLoader(flo_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=num_workers,
                                pin_memory=True)
    print(f'Loading FLO Caption: n_images {flo_dataset.n_images} n_captions {len(flo_dataset.targets)}...')
    return dataloader


def prepare_flo_dataloaders(dataloader_config,
                            dataset_root,
                            caption_root,
                            dataset_name='flo',
                            num_workers=6):
    """Prepare FLO Caption train / val / test dataloaders
    FLO Caption loader has a fixed batch size
    - train loader: # classes (trainval = 100, full = 150)
    - test loader: 64 (hard coded at L#203)
    Args:
        dataloader_config (dict): configuration file which should contain "batch_size"
        dataset_name (str): name of dataset
            - flo_trainval{idx} (idx in [1, 2, 3]):
                3-fold validation splits to search hyperparameters.
                Each split contains 100 train classes / 50 validation classes.
            - cub:
                The final split used for the final benchmark.
                This split conntains 150 train classes / 50 unseen test classes (not in trainval)
        dataset_root (str): root of your CUB images (see README.md for detailed dataset hierarchy)
        caption_root (str): root of your CUB captions (see README.md for detailed dataset hierarchy)
        vocab_path (str, optional): path for vocab pickle file (default: ./vocabs/cub_vocab.pkl).
        num_workers (int, optional): num_workers for the dataloaders (default: 6)
    Returns:
        dataloaders (dict): keys = ["train", "val", "val_in"], values are the corresponding dataloaders.
        vocab (Vocabulary object): vocab object
    """
    train_classes, val_classes, image_root, caption_root = _get_flo_file_paths(
        dataset_name, dataset_root, caption_root)


    dataloaders = {}
    dataloaders['train'] = _get_flo_loader(
        image_root, caption_root,
        train_classes,
        num_workers,
        train=True,
    )

    dataloaders['test'] = _get_flo_loader(
        image_root, caption_root,
        val_classes,
        num_workers,
        train=False,
    )

    dataloaders['val'] = _get_flo_loader(
        image_root, caption_root,
        train_classes,
        num_workers,
        train=False,
    )

    return dataloaders,None
