import os
import torch
from latent_direction_explorer import LatentDirectionExplorer

from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(.5,), std=(.5,))
])

useBias = True
direction_count = 50
steps = 20000

# dataset = get_ano_mnist_dataset(transform=transform, root_dir=os.path.abspath("../../data"))
# visualizer = LatentDirectionVisualizer(matrix_a_linear=trainer.matrix_a, generator=trainer.g, device=device)


trainer = LatentDirectionExplorer(z_dim=100, latent_dim=100, directions_count=50, batch_size=1, bias=useBias,
                                  device=device, saved_models_path='/home/yashar/git/python/AD-with-GANs/data/DS01_mnist_9_6_20_percent/matrix_a/')
trainer.load_generator(
    os.path.abspath("/home/yashar/git/python/AD-with-GANs/data/DS01_mnist_9_6_20_percent/generator.pkl"))

b = 'bias' if useBias else 'nobias'
trainer.train_and_save(filename=f'DS1_matrix_a_steps_{steps}_{b}_k_{direction_count}.pkl', num_steps=steps)

# noise_batches = generate_noise(batch_size=4, z_dim=100, device=device)
# visualizer.visualize(noise_batches=noise_batches, shifts_range=10, output_directory=os.path.abspath("../out_dir"))
# images = visualizer.generate_random_batches_as_numpy_array(batch_size=3, shifts_range=8, output_directory=os.path.abspath("../out_dir"))

# print(get_random_strip_as_numpy_array("../../out_dir/data.npy").shape)
