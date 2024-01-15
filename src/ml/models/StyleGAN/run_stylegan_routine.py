import torch
from torch import nn

import dnnlib
import legacy
from src.ml.models.StyleGAN.stylegan_reconstructor import StyleGANReconstructor
from src.ml.models.base.matrix_a_linear import MatrixALinear

device = torch.device('cuda')

network_pkl = '/home/yashar/git/AD-with-GANs/data/StyleGAN2_CelebA/stylegan2-celebahq-256x256.pkl'
direction_count = 100
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
    G = networks['G'].to(device)
    D = networks['D'].to(device)
reconstructor = StyleGANReconstructor(directions_count=direction_count,
                                      num_channels=num_channels,
                                      width=2).to(device)


def generate_noise():
    return torch.randn(14, 512, device=device)


def generate_stylegan2_image(z):
    img = G(z, None, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(device)
    # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').show()
    return img


def generate_shifted_stylegan2_image(z, shift):
    return generate_stylegan2_image(z + shift)


def train_direction_matrix(steps):
    print('TRAINING DIRECTION MATRIX')
    train_and_save_directions(num_steps=steps)


def make_shifts(latent_dim, batch_size=1):
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


def train_and_save_directions(num_steps=1000, bias=True):
    # init optimizers for MatrixA, Reconstructor
    matrix_a = MatrixALinear(input_dim=direction_count, bias=bias, output_dim=z_dim).to(device)
    matrix_a_opt = torch.optim.Adam(matrix_a.parameters(), lr=reconstructor_lr)
    reconstructor_opt = torch.optim.Adam(reconstructor.parameters(), lr=reconstructor_lr)

    # start training loop
    for step in range(num_steps):
        G.zero_grad()
        matrix_a.zero_grad()
        reconstructor.zero_grad()

        # cast random noise z
        z = generate_noise()

        # generate shifts
        # cast random integer that represents the k^th column  --> e_k
        target_indices, shifts, basis_shift = make_shifts(matrix_a.input_dim, 14)
        shift = matrix_a(basis_shift)

        # generate images --> from z and from z + A(epsilon * e_k)
        images = generate_stylegan2_image(z)
        images_shifted = generate_shifted_stylegan2_image(z, shift)

        logits, shift_prediction = reconstructor(images, images_shifted)
        logit_loss = label_weight * cross_entropy(logits.to(device), target_indices.to(device))
        shift_loss = shift_weight * torch.mean(torch.abs(shift_prediction - shifts))

        # total loss
        loss = logit_loss + shift_loss
        loss.backward()

        matrix_a_opt.step()
        reconstructor_opt.step()
    # display image from A(x) with shift epsilon
    print(matrix_a)


train_and_save_directions(num_steps=10)

# generate_stylegan2_image(
#     z=torch.randn(14, 512, device=device),
# )
