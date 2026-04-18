from .cifar import get_cifar_loaders
from .cifar100 import get_cifar100_loaders  # kept for backwards compatibility

__all__ = ["get_cifar_loaders", "get_cifar100_loaders"]
