import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader#, Subset


# Training transforms
def get_train_transform(image_size, crop_size, pretrained):
    train_transform = transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    #transforms.RandomResizedCrop(224),
                    transforms.RandomCrop(crop_size),
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
                    normalize_transform(pretrained)])
    return train_transform

# Validation transforms
def get_valid_transform(crop_size, pretrained):
    valid_transform = transforms.Compose([
                    transforms.Resize((crop_size, crop_size)),
                    #transforms.CenterCrop(299),
                    #transforms.Grayscale(num_output_channels=3),
                    transforms.ToTensor(),
                    normalize_transform(pretrained)])
    return valid_transform

# Image normalization transforms
def normalize_transform(pretrained):
    if pretrained: # Normalization for pre-trained weighted, using imagenet mean and std
        normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
                )
    else: # Normalization when training from scratch
        normalize = transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
                )
    
    return normalize

def get_datasets(traindir, valdir, image_size, crop_size, pretrained):
    """
    Function to prepare the Datasets.
    :param traindir: String, the Dir of trainset.
    :param valdir: String, the Dir of valset.
    :param image_size: Resize size.
    :param crop size: Crop size. crop_size <= image_size.
    :param pretrained: Boolean, True or False.
    Return the training and validation datasets along with the class names.
    """
    dataset = datasets.ImageFolder(
            traindir,
            transform=(get_train_transform(image_size, crop_size, pretrained))
    )
    dataset_test = datasets.ImageFolder(
            valdir,
            transform=(get_valid_transform(crop_size, pretrained))
    )

    #dataset_size = len(dataset)

    return dataset, dataset_test

def get_data_loaders(dataset_train, dataset_valid, batch_size, num_workers):
    """
    Prepares the training and validation data loaders.
    :param dataset_train: The training dataset.
    :param dataset_valid: The Validation dataset.
    :param batch_size: Batch Size.
    :param num_workers:Number of parallel processes for data preparation.
    Returns the training and validation data loaders.
    """
    train_loader = DataLoader(
            dataset_train, batch_size=batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=True
    )
    valid_loader = DataLoader(
            dataset_valid, batch_size=batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=True
    )

    return train_loader, valid_loader
