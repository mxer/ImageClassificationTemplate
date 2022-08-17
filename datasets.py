import os
import cv2
import torch
import albumentations as A

from utils.paths import list_images


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, transforms=None):
        self.transforms = transforms
        self.labels = []
        self.img_paths = list(list_images(data_paths))
        for img_path in self.img_paths:
            self.labels.append(img_path.split('/')[-2]) # dir names are class names

        classes = [d.name for d in os.scandir(data_paths) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        label = self.class_to_idx[self.labels[idx]]

        return image, label, img_path

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, transforms=None):
        self.transforms = transforms
        self.img_paths = list(list_images(data_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, img_path


def get_transforms(size, crop, mode="train", pretrained=True):
    if pretrained:
        normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), always_apply=True)
    else:
        normalize = A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), always_apply=True)

    if mode == "train":
        return A.Compose(
            [
                A.Resize(size, size, interpolation=cv2.INTER_LINEAR, always_apply=True),
                A.RandomCrop(width=crop, height=crop, always_apply=True),
                A.HorizontalFlip(p=0.5),
                A.Rotate(30),
                #A.GaussNoise(),
                #A.ImageCompression(),
                #A.RandomBrightnessContranst(p=0.2),
                #A.ColorJitter(0.4, 0.4, 0.4),
                normalize,
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(crop, crop, interpolation=cv2.INTER_LINEAR, always_apply=True),
                #A.CenterCrop(height=crop, width=crop),
                normalize,
            ]
        )

