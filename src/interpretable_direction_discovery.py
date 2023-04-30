import sys

import numpy as np
import torch

from src.latent_direction_discoverer import LatentDirectionDiscoverer

sys.path.append('tools')

from tools.ano_mnist_dataset_generator import get_ano_mnist_dataset
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

dataset = get_ano_mnist_dataset(transform=transform, root_dir="../data")
trainer = LatentDirectionDiscoverer(z_dim=100, latent_dim=100, directions_count=100, batch_size=1, device=device)
trainer.load_generator("../saved_models/generator.pkl")
trainer.train()
