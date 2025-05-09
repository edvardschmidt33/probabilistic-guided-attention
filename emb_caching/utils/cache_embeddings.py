import torch
import clip
import os
import json
from torch.utils.data import DataLoader
from utils.preprocess import preprocess
import importlib
import argparse

#maybe change to different paths
PATH_DICT = {
    "coco":{
        "train": {
            "root": "/mimer/NOBACKUP/Datasets/Microsoft-COCO/train2017",
            "ann_file": "/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/captions_train2017.json",},
        "val": {
            "root": "/mimer/NOBACKUP/Datasets/Microsoft-COCO/val2017",
            "ann_file": "/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/captions_val2017.json",},},
    }


@torch.no_grad()
def main(dataset, split='train'):
    device = "cuda:0"
    model, _ = clip.load("ViT-B/32", device=device)
    model.eval()

    # paths
    root = PATH_DICT[dataset][split]["root"]
    ann_file = PATH_DICT[dataset][split]["ann_file"]

    embedding_path = f'embeddings/{dataset}/{split}'
    os.makedirs(embedding_path, exist_ok=True)
    img_embedding_path = f'{embedding_path}/image.pt'
    txt_embedding_path = f'{embedding_path}/text.pt'
    cap_id_to_img_id_path = f'{embedding_path}/cap_id_to_img_id.json'

    # dataset
    ImageDataset = importlib.import_module(f'datasets.{dataset}').ImageDataset
    CaptionDataset = importlib.import_module(f'datasets.{dataset}').CaptionDataset

    num_workers = 4 if "T4" in torch.cuda.get_device_name() else 8

    # extract image embeddings
    dataset = ImageDataset(
        root=root, ann_file=ann_file, transform=preprocess(),)

    batch_size = 512
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    image_embeddings = []
    with torch.no_grad():
        for idx, images in enumerate(dataloader):
            print(f'Processing batch {idx+1}/{len(dataloader)}', flush=True)
            images = images.to(device)
            embedding = model.encode_image(images)
            image_embeddings.append(embedding.cpu())

    image_embeddings = torch.cat(image_embeddings, dim=0)

    torch.save(image_embeddings, img_embedding_path)

    # ========================================================================

    # extract text embeddings
    dataset = CaptionDataset(
        root=root, ann_file=ann_file,
        target_transform=lambda x: clip.tokenize(x, truncate=True))

    batch_size = 512
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2)

    text_embeddings = []
    with torch.no_grad():
        for idx, captions in enumerate(dataloader):
            print(f'Processing batch {idx+1}/{len(dataloader)}', flush=True)
            captions = captions.to(device).reshape(-1, 77)
            embedding = model.encode_text(captions)
            text_embeddings.append(embedding.cpu())

    text_embeddings = torch.cat(text_embeddings, dim=0)

    # save the text embeddings in half precision
    torch.save(text_embeddings, txt_embedding_path)

    # save the cap_img_id_dict as json
    with open(cap_id_to_img_id_path, 'w') as f:
        json.dump(dataset.cap_id_to_img_id, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='coco')
    parser.add_argument('--split', type=str, default='train')
    args = parser.parse_args()


    main(**vars(args))
