from .mnist import get_mnist
from .usps import get_usps
from .svhn import get_svhn
from .data_loader import load_images
from .loader import load

__all__ = (get_mnist, get_usps, get_svhn, load_images, load)
