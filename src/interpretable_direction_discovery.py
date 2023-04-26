import sys

import numpy as np
import torch

sys.path.append('tools')

from tools.dataset_generator import get_ano_mnist_dataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

dataset = get_ano_mnist_dataset(transform=None, root_dir="../data")
# dataset[3][0].show()

x: torch.Tensor = torch.Tensor([[1, 2, 3],
                   [4, 5, 6]])
print(int(x[1, 2].item()))
