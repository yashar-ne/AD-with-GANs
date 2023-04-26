import sys

sys.path.append('utils')

from utils.dataset_generator import get_ano_mnist_dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

dataset = get_ano_mnist_dataset(transform=None, root_dir="../data")
dataset[3][0].show()

