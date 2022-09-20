import os
import cv2
import torch
from PIL import Image
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from utils.paths import list_images, list_filtering_images

_DEFAULT_CROP_PCT = 0.875
_IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
_IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
_IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, transform=None):
        super().__init__()
        self.transform = transform
        self.labels = []
        #self.img_paths = list(list_filtering_images(data_paths)) # could filtering corrupt images onling, but may be slow
        self.img_paths = list(list_images(data_paths)) # after use rewrite_images() function in paths.py offline
        for img_path in self.img_paths:
            self.labels.append(img_path.split('/')[-2]) # dir names are class names

        #self.img_paths = []
        #for root, _, fnames in sorted(os.walk(data_paths, followlinks=True)):
        #    for fname in sorted(fnames):
        #        img_path = os.path.join(root, fname)
        #        self.img_paths.append(img_path)

        #        self.labels.append(img_path.split('/')[-2]) # dir names are class names

        classes = [d.name for d in os.scandir(data_paths) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        self.classes = classes
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        #image = cv2.imread(img_path)
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1) # 1：colorful；2：gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # equal to `image = image[:, :, (2, 1, 0)]`

        if self.transform is not None:
            image = Image.fromarray(image)  #need convert to PIL first
            image = self.transform(image)

        label = self.class_to_idx[self.labels[idx]]

        return image, label, img_path


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_paths, transform=None):
        super().__init__()
        self.transform = transform
        self.img_paths = list(list_images(data_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        #image = cv2.imread(img_path)
        image = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), 1) # 1：colorful；2：gray
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)['image']
            image = torch.from_numpy(image).permute(2, 0, 1).float()

        return image, img_path

def build_transform_timm():
    """TODO: use timm.data.create_transform"""
    pass

def build_transform(input_size, mode="train", pretrained=True):
    resize_size = int(input_size / _DEFAULT_CROP_PCT)
    if pretrained:
        mean = _IMAGENET_DEFAULT_MEAN
        std = _IMAGENET_DEFAULT_STD
    else:
        mean = _IMAGENET_INCEPTION_MEAN
        std = _IMAGENET_INCEPTION_STD

    if mode == "train":
        return transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                #transforms.RandomResizedCrop(224),
                transforms.RandomCrop(input_size),
                transforms.RandomRotation(30),
                #transforms.RandomHorizontalFlip(p=0.5),
                #transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                #transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                #transforms.RandomGrayscale(p=0.4),
                #transforms.Grayscale(num_output_channels=3),
                #transforms.RandomAffine(45, shear=0.2),
                #transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                #transforms.Lambda(utils.randomColor),
                #transforms.Lambda(utils.randomBlur),
                #transforms.Lambda(utils.randomGaussian),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ]
        )
    else:
        return transforms.Compose([
                transforms.Resize((input_size, input_size)),
                #transforms.CenterCrop(299),
                #transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
        ]
        )

def build_dataset(data_path, input_size, mode='train', pretrained=True):
    if mode == 'train':
        prefix = 'train'
    elif mode== 'val':
        prefix = 'val'
    elif mode == 'test':
        prefix == 'test'
    root = os.path.join(data_path, prefix)
    if mode != 'test':
        dataset = ImageDataset(
            root,
            transform=build_transform(input_size, mode=mode, pretrained=pretrained))
    else:
        dataset = TestDataset(
            root,
            transform=build_transform(input_size, mode=mode, pretrained=pretrained))

    #dataset_size = len(dataset)

    return dataset

def build_loader(data_path, input_size, batch_size, num_workers):
    dataset_train = build_dataset(data_path, input_size, mode='train', pretrained=True)
    train_loader = DataLoader(
            dataset_train, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=True
    )

    dataset_val = build_dataset(data_path, input_size, mode='val', pretrained=True)
    valid_loader = DataLoader(
            dataset_val, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=True
    )

    return train_loader, valid_loader

def build_test_loader(data_path, input_size, batch_size, num_workers):
    dataset_test = build_dataset(data_path, input_size, mode='test', pretrained=True)
    test_loader = DataLoader(
            dataset_test, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=True
    )

    return test_loader
