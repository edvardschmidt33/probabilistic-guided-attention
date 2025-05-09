from torchvision.datasets.coco import CocoDetection


class ImageDataset(CocoDetection):
    def __init__(self, root, ann_file, transform=None):
        super().__init__(root, ann_file, transform=transform)
        self.img_id_to_name = list(sorted(self.coco.imgs.keys()))
    
    def __len__(self):
        return len(self.img_id_to_name)
    
    def __getitem__(self, img_id):
        name = self.img_id_to_name[img_id]
        # Load image
        img = self._load_image(name)

        if self.transform is not None:
            img = self.transform(img)

        return img


class CaptionDataset(CocoDetection):
    def __init__(self, root, ann_file, target_transform=None):
        super().__init__(root, ann_file, target_transform=target_transform)
        img_name_to_id = {
            img_name: img_id for img_id, img_name in enumerate(sorted(self.coco.imgs.keys()))
        }
        
        cap_id_to_img_id = {}
        cap_id_to_text = []

        for cap_id, cap_name in enumerate(sorted(self.coco.anns.keys())):
            img_name = self.coco.anns[cap_name]['image_id']

            cap_id_to_img_id[str(cap_id)] = img_name_to_id[img_name]

            cap_id_to_text.append(self.coco.anns[cap_name]['caption'])

        self.cap_id_to_text = cap_id_to_text
        self.cap_id_to_img_id = cap_id_to_img_id

    def __len__(self):
        return len(self.cap_id_to_text)
    
    def __getitem__(self, cap_id):
        target = self.cap_id_to_text[cap_id]

        if self.target_transform is not None:
            target = self.target_transform(target)

        return target


if __name__ == '__main__':
    # root = "/mimer/NOBACKUP/Datasets/Microsoft-COCO/train2017"
    # ann_file = "/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/captions_train2017.json"

    root = "/mimer/NOBACKUP/Datasets/Microsoft-COCO/val2017"
    ann_file = "/mimer/NOBACKUP/Datasets/Microsoft-COCO/annotations/captions_val2017.json"


    img_dataset = ImageDataset(root, ann_file)
    print(len(img_dataset))

    cap_dataset = CaptionDataset(root, ann_file)

    print(len(cap_dataset))

    # print(dataset[0][1])
    # # save the first image
    # dataset[0][0].save('test.jpg')