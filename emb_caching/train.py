import torch
import clip
from datasets.embedding import EmbeddingDataset
from torch.utils.data import DataLoader
import os
import argparse


def main(args):
    data = args.dataset

    dataset = EmbeddingDataset(
        f'embeddings/{data}/train/image.pt',
        f'embeddings/{data}/train/text.pt',
        f'embeddings/{data}/train/cap_id_to_img_id.json')

    num_workers = 4 if "T4" in torch.cuda.get_device_name() else 16
    batch_size = 512

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, prefetch_factor=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    main(args)