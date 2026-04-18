"""CIFAR-100 DataLoader factory.

Images are normalised to [−1, 1] (mean=0.5, std=0.5 per channel).
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_cifar100_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 4,
    image_size: int = 32,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader) for CIFAR-100.

    Training split uses random horizontal flipping; val split does not.
    """
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    train_transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm]
    )
    val_transform = transforms.Compose(
        [transforms.Resize(image_size), transforms.ToTensor(), norm]
    )

    train_ds = datasets.CIFAR100(data_dir, train=True, download=True, transform=train_transform)
    val_ds = datasets.CIFAR100(data_dir, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
