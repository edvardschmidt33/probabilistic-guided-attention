from torch.utils.data import Dataset
import torch
import json


class EmbeddingDataset(Dataset):
    def __init__(self, img_path, cap_path, cap_id_to_img_id_path):
        super().__init__()
        self.imgs = torch.load(img_path).float()
        self.caps = torch.load(cap_path).float()
        with open(cap_id_to_img_id_path, 'r') as f:
            self.cap_id_to_img_id = json.load(f)
        print(f"Loaded {len(self.imgs)} images and {len(self.caps)} captions")

    def __len__(self):
        return self.caps.size(0)

    def __getitem__(self, cap_id):
        img_id = self.cap_id_to_img_id[str(cap_id)]
        img, text = self.imgs[img_id], self.caps[cap_id]
        return img, text


if __name__ == '__main__':
    dataset = EmbeddingDataset(
        f'embeddings/coco_val/image.pt',
        f'embeddings/coco_val/text.pt',
        f'embeddings/coco_val/cap_id_to_img_id.json')
    print(len(dataset))
