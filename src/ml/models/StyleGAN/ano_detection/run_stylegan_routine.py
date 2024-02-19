import os.path

import PIL
import torch
from torch import nn
from torchvision.transforms import transforms

from src.ml.dataset_generation.dataset_generation_core import get_dataloader
from src.ml.latent_space_mapper import LatentSpaceMapper
from src.ml.models.StyleGAN import dnnlib, legacy
from src.ml.models.StyleGAN.ano_detection.stylegan_reconstructor import StyleGANReconstructor
from src.ml.models.base.matrix_a_linear import MatrixALinear

device = torch.device('cuda')

network_pkl = '/home/yashar/git/AD-with-GANs/data/StyleGAN2_CelebA/stylegan2-celebahq-256x256.pkl'
num_channels = 3
z_dim = 512
reconstructor_lr = 0.002
truncation_psi = 0.7
noise_mode = 'const'
cross_entropy = nn.CrossEntropyLoss().to(device)
label_weight = 1.0
shift_weight = 0.25
min_shift = 0.5
shift_scale = 6.0

print('Loading networks from "%s"...' % network_pkl)
with dnnlib.util.open_url(network_pkl) as f:
    networks = legacy.load_network_pkl(f)
    G = networks['G_ema'].to(device)
    D = networks['D'].to(device)

lsm: LatentSpaceMapper = LatentSpaceMapper(G, D, device)


def generate_noise():
    return torch.randn(14, 512, device=device)


def generate_stylegan2_image(z, class_id=None, show=False):
    img = G(z, class_id, truncation_psi=truncation_psi, noise_mode='const')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(device)
    if show:
        PIL.Image.fromarray(img.to(torch.uint8)[0].cpu().numpy(), 'RGB').show()
    return img


def generate_shifted_stylegan2_image(z, shift):
    return generate_stylegan2_image(z + shift)


def train_direction_matrix(steps):
    print('TRAINING DIRECTION MATRIX')
    train_and_save_directions(num_steps=steps)


def make_shifts(latent_dim, direction_count, batch_size=1):
    target_indices = torch.randint(0, direction_count, [batch_size])

    # Casting from uniform distribution
    # See https://github.com/anvoynov/GANLatentDiscovery/blob/5ca8d67bce8dcb9a51de07c98e2d3a0d6ab69fe3/trainer.py#L75
    shifts = 2.0 * torch.rand(target_indices.shape, device=device) - 1.0

    shifts = shift_scale * shifts
    shifts[(shifts < min_shift) & (shifts > 0)] = min_shift
    shifts[(shifts > -min_shift) & (shifts < 0)] = -min_shift

    z_shift = torch.zeros([batch_size] + [latent_dim], device=device)
    for i, (index, val) in enumerate(zip(target_indices, shifts)):
        z_shift[i][index] += val

    return target_indices, shifts, z_shift


def train_and_save_directions(save_path, num_steps=1000, direction_count=40, bias=True):
    # init optimizers for MatrixA, Reconstructor
    direction_matrix = MatrixALinear(input_dim=direction_count, bias=bias, output_dim=z_dim).to(device)
    direction_matrix_opt = torch.optim.Adam(direction_matrix.parameters(), lr=reconstructor_lr)
    reconstructor = StyleGANReconstructor(directions_count=direction_count,
                                          num_channels=num_channels,
                                          width=2).to(device)
    reconstructor_opt = torch.optim.Adam(reconstructor.parameters(), lr=reconstructor_lr)

    # start training loop
    for step in range(num_steps):
        if step % 100 == 0:
            print(f'Step {step} of {num_steps}')

        G.zero_grad()
        direction_matrix.zero_grad()
        reconstructor.zero_grad()

        # cast random noise z
        z = generate_noise()

        # generate shifts
        # cast random integer that represents the k^th column  --> e_k
        target_indices, shifts, basis_shift = make_shifts(direction_matrix.input_dim, direction_count, 14)
        shift = direction_matrix(basis_shift)

        # generate images --> from z and from z + A(epsilon * e_k)
        images = generate_stylegan2_image(z)
        images_shifted = generate_shifted_stylegan2_image(z, shift)

        logits, shift_prediction = reconstructor(images, images_shifted)
        logit_loss = label_weight * cross_entropy(logits.to(device), target_indices.to(device))
        shift_loss = shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

        # total loss
        loss = logit_loss + shift_loss
        loss.backward()

        direction_matrix_opt.step()
        reconstructor_opt.step()
    # display image from A(x) with shift epsilon
    filename = os.path.join(save_path, f'direction_matrix_{direction_count}_directions_{num_steps}_steps.pt')
    torch.save(direction_matrix, filename)
    print(direction_matrix)


def create_latent_space_dataset(root_dir, dataset_name):
    print('MAPPING LATENT SPACE POINTS')
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    dataset_folder = os.path.join(root_dir, dataset_name, 'dataset')
    dataset_raw_folder = os.path.join(root_dir, dataset_name, 'dataset_raw')
    if os.path.exists(dataset_folder):
        # shutil.rmtree(dataset_folder)
        print('Dataset already exists')
        return

    os.makedirs(dataset_folder, exist_ok=True)
    csv_path = os.path.join(dataset_folder, "latent_space_mappings.csv")
    dataset = get_dataloader(dataset_folder=dataset_raw_folder,
                             batch_size=1,
                             transform=transform,
                             shuffle=True)

# train_and_save_directions(
#     save_path=f'/data/StyleGAN2_CelebA/direction_matrices/',
#     num_steps=3000,
#     direction_count=40,
# )

# generate_stylegan2_image(
#     z=torch.randn(14, 512, device=device),
#     show=True,
# )
