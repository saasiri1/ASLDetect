from torchvision import transforms
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms




def calculate_custom_stats(dataset):
    """Calculate dataset-specific mean and std"""
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    mean = 0.
    std = 0.
    for images, _ in loader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
    mean /= len(loader.dataset)
    std /= len(loader.dataset)
    return mean.tolist(), std.tolist()


def dataPreprocess(path):
    # First get base dataset without heavy augs to calculate stats
    temp_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    temp_dataset = datasets.ImageFolder(root=path+"/training", transform=temp_transform)
    custom_mean, custom_std = calculate_custom_stats(temp_dataset)

    # Enhanced training augmentations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Higher initial resize
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Perspective transform
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomCrop(224),  # Final crop size
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std),  # Custom stats
    ])

    # Validation/Test transforms
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std),
    ])
    test_path = path+"/testing"
    # Load datasets
    train_dataset = datasets.ImageFolder(root=path+"/training", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=path+"/validation", transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)

    batch_size = 64  # Increased batch size for better generalization
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True)
    
    return train_loader,val_loader, test_loader, train_dataset, val_dataset,test_dataset, test_path

def dataPreprocessNew(path):
    # First get base dataset without heavy augs to calculate stats
    temp_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    temp_dataset = datasets.ImageFolder(root=path+"/training", transform=temp_transform)
    custom_mean, custom_std = calculate_custom_stats(temp_dataset)

    # Enhanced training augmentations
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Higher initial resize
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # Perspective transform
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
        transforms.RandomCrop(224),  # Final crop size
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std),  # Custom stats
    ])

    # Validation/Test transforms
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=custom_mean, std=custom_std),
    ])
    test_path = path+"/testing"
    # Load datasets
    train_dataset = datasets.ImageFolder(root=path+"/training", transform=train_transform)
    val_dataset = datasets.ImageFolder(root=path+"/validation", transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_path, transform=test_transform)
    # concatenate the train and val datasets
    train_dataset = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
    # Create loaders with balanced sampler if needed
    batch_size = 64  # Increased batch size for better generalization
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8,pin_memory=True)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,num_workers=8,pin_memory=True)

    
    return train_loader, test_loader, train_dataset, test_dataset, test_path