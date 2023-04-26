import torch


def make_noise(batch, dim):
    if isinstance(dim, int):
        dim = [dim]
    return torch.randn([batch] + dim)
