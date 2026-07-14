"""
MAI/IDL SS26 - Final assignment. 

MG 6/6/2026
"""
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def image_augmentation(image):
    train_transform = transforms.Compose([
        transforms.RandomApply([ 
            transforms.RandomAffine(degrees=10,translate=(0.05, 0.05),scale=(0.9, 1.1),) ], p=0.5),
        transforms.RandomApply([
                transforms.ColorJitter(brightness=0.2, contrast=0.2,)], p=0.5),])
    return train_transform(image)


class MedicalDataset(Dataset):
    def __init__(self, images, labels, mean, std, transform=False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform :
            image = image_augmentation(image)

        image = (image - self.mean) / self.std

        return image, label
    


def get_loaders(data, data_path, batch_size, val_split=0.1, seed=42, transform=False):

    d_path = Path(data_path) / f"{data}.pt"
    data_dict = torch.load(d_path)

    total_samples = data_dict['train_images'].shape[0]
    val_size = int(total_samples * val_split)
    val_start = total_samples - val_size

    g = torch.Generator().manual_seed(seed)
    data_index = torch.randperm(total_samples, generator=g)
    train_idx = data_index[:val_start]
    val_idx   = data_index[val_start:]

    train_data = data_dict['train_images'][train_idx]
    train_labels = data_dict['train_labels'][train_idx]
    val_data = data_dict['train_images'][val_idx]
    val_labels = data_dict['train_labels'][val_idx]
    test_data = data_dict['test_images']
    test_labels = data_dict['test_labels']
    
    mean = train_data.mean(dim=(0, 2, 3)).view(-1, 1, 1)
    std = train_data.std(dim=(0, 2, 3)).clamp_min(1e-8).view(-1, 1, 1)
    train_dataset = MedicalDataset(train_data, train_labels, mean, std, transform=transform)
    val_dataset = MedicalDataset(val_data, val_labels, mean, std, transform= False)
    test_dataset = MedicalDataset(test_data, test_labels, mean, std, transform = False)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader
