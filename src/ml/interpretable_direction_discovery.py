import os
import sys

sys.path.append('./ml/tools')
sys.path.append('./ml/models')

import torch
from latent_direction_explorer import LatentDirectionExplorer
from latent_direction_visualizer import LatentDirectionVisualizer, get_random_strip_as_numpy_array
from tools.utils import generate_noise
from tools.ano_mnist_dataset_generator import get_ano_mnist_dataset

from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

useBias = False

# dataset = get_ano_mnist_dataset(transform=transform, root_dir=os.path.abspath("../../data"))
# visualizer = LatentDirectionVisualizer(matrix_a_linear=trainer.matrix_a, generator=trainer.g, device=device)

trainer = LatentDirectionExplorer(z_dim=100, latent_dim=100, directions_count=100, batch_size=1, bias=useBias, device=device)
trainer.load_generator(os.path.abspath("../../saved_models/generator.pkl"))
trainer.train_and_save(num_steps=20000)

# noise_batches = generate_noise(batch_size=4, z_dim=100, device=device)
# visualizer.visualize(noise_batches=noise_batches, shifts_range=10, output_directory=os.path.abspath("../out_dir"))
# images = visualizer.generate_random_batches_as_numpy_array(batch_size=3, shifts_range=8, output_directory=os.path.abspath("../out_dir"))

# print(get_random_strip_as_numpy_array("../../out_dir/data.npy").shape)
