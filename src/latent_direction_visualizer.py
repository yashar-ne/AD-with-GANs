import torch
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid

from src.models.generator import Generator
from src.models.matrix_a import MatrixA

import matplotlib as plt
import numpy as np


class LatentDirectionVisualizer:
    def __init__(self, generator: Generator, matrix_a: MatrixA, device):
        super(LatentDirectionVisualizer, self).__init__()
        self.g: Generator = generator
        self.matrix_a: MatrixA = matrix_a
        self.dim = self.g.size_z
        self.device = device

    @torch.no_grad()
    def create_visualization(self, z, dims_count, shifts_count=5):
        self.g.eval()
        original_img = self.g(z).cpu()
        images = []

        dims = range(dims_count)
        for i in dims:
            images.append(self.create_shifted_images(z=z, shifts_r=10, shifts_count=5, dim=i))

        rows_count = len(images) + 1
        fig, axs = plt.subplots(rows_count)

        axs[0].axis('off')
        axs[0].imshow(self.__to_image(original_img, True))

        texts = dims
        for ax, shifts_imgs, text in zip(axs[1:], images, texts):
            ax.axis('off')
            plt.subplots_adjust(left=0.5)
            ax.imshow(self.__to_image(make_grid(shifts_imgs, nrow=(2 * shifts_count + 1), padding=1), True))
            ax.text(-20, 21, str(text), fontsize=10)

        self.matrix_a.train()

        return fig

    @torch.no_grad()
    def create_shifted_images(self, z, shifts_r, shifts_count, dim):
        shifted_images = []
        for shift in np.arange(-shifts_r, shifts_r + 1e-9, shifts_r / shifts_count):
            latent_shift = self.matrix_a(self.__one_hot(self.matrix_a.input_dim, shift, dim).cuda())
            shifted_image = self.g.gen_shifted(z, latent_shift).cpu()[0]
            shifted_images.append(shifted_image)

        return shifted_image

    def __to_image(self, tensor):
        if len(tensor.shape) == 4:
            tensor = tensor[0]

        tensor = (tensor + 1) / 2
        tensor.clamp(0, 1)
        return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))

    def __one_hot(self, dims, value, index):
        vec = torch.zeros(dims)
        vec[index] = value
        return vec
