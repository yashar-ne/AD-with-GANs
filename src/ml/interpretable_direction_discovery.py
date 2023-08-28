import os
import torch
from latent_direction_explorer import LatentDirectionExplorer

from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

base_path = '/home/yashar/git/python/AD-with-GANs/data/DS01_mnist_9_6_20_percent/'
useBias = True
direction_count = 10
steps = 3000

# dataset = get_ano_mnist_dataset(transform=transform, root_dir=os.path.abspath("../../data"))
# visualizer = LatentDirectionVisualizer(matrix_a_linear=trainer.matrix_a, generator=trainer.g, device=device)


trainer = LatentDirectionExplorer(z_dim=100, directions_count=direction_count, bias=useBias, device=device, saved_models_path=os.path.join(base_path, 'matrix_a'))
trainer.load_generator(os.path.join(base_path, 'generator.pkl'))

b = 'bias' if useBias else 'nobias'
trainer.train_and_save(filename=f'DS1_matrix_a_steps_{steps}_{b}_k_{direction_count}.pkl', num_steps=steps)

# noise_batches = generate_noise(batch_size=4, z_dim=100, device=device)
# visualizer.visualize(noise_batches=noise_batches, shifts_range=10, output_directory=os.path.abspath("../out_dir"))
