import torch
from torchvision.transforms import ToPILImage


def generate_noise(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, 1, 1, device=device)


def to_image(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    tensor = (tensor + 1) / 2
    tensor.clamp(0, 1)
    return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def one_hot(dims, value, index):
    vec = torch.zeros(dims)
    vec[index] = value
    return vec
