import kagglehub
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_transform(test=False):
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(150),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(150),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    if test:
        return test_transform
    else:
        return train_transform


def fetch_deepfake_images():
    path = kagglehub.dataset_download("manjilkarki/deepfake-and-real-images")

    train_path = os.path.join(path, "Dataset/Train")
    test_path = os.path.join(path, "Dataset/Test")
    validation_path = os.path.join(path, "Dataset/Validation")

    train_transform = get_transform()
    test_transform = get_transform(test=True)

    training_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    testing_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    validation_dataset = datasets.ImageFolder(validation_path, transform=test_transform)

    return training_dataset, testing_dataset, validation_dataset



def dataloader(dataset, shuffle=False, batch_size=64, num_workers=8, pin_memory=True):
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader