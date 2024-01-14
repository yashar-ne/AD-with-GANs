import PIL.Image
import torch

import dnnlib
import legacy

device = torch.device('cuda')


def generate_stylegan2_image(
        z,
        network_pkl,
        truncation_psi: float,
        noise_mode: str,
):
    print('Loading networks from "%s"...' % network_pkl)

    with dnnlib.util.open_url(network_pkl) as f:
        networks = legacy.load_network_pkl(f)
        G = networks['G'].to(device)
        D = networks['D'].to(device)

        # Generate images.
    img = G(z, None, truncation_psi=truncation_psi, noise_mode=noise_mode)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').show()


generate_stylegan2_image(
    z=torch.randn(14, 512, device=device),
    network_pkl='/home/yashar/git/AD-with-GANs/data/stylegan2-celebahq-256x256.pkl',
    truncation_psi=0.7,
    noise_mode='const',
)
