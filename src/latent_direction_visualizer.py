import os
from matplotlib import pyplot as plt
import numpy as np
import torch

from torchvision.utils import make_grid
from src.models.generator import Generator
from src.models.matrix_a import MatrixA
from src.tools.utils import one_hot, to_image
from PIL import Image


class LatentDirectionVisualizer:
    def __init__(self, generator: Generator, matrix_a: MatrixA, device):
        super(LatentDirectionVisualizer, self).__init__()
        self.g: Generator = generator
        self.matrix_a: MatrixA = matrix_a
        self.dim = self.g.size_z
        self.device = device

    @torch.no_grad()
    def visualize(self, noise_batches, output_directory, shifts_range=8):
        os.makedirs(output_directory, exist_ok=True)

        step = 20
        max_dim = self.g.size_z
        shifts_count = noise_batches.shape[0]

        for start in range(0, max_dim - 1, step):
            images = []
            vis_range = range(start, min(start + step, max_dim))
            for z in noise_batches:
                z = z.unsqueeze(0)
                fig = self.create_visualization(z=z,
                                                vis_range=vis_range,
                                                shifts_range=shifts_range,
                                                shifts_count=shifts_count,
                                                figsize=(int(shifts_count * 4.0), int(0.5 * step) + 2))
                fig.canvas.draw()
                plt.close(fig)
                img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

                # crop borders
                nonzero_columns = np.count_nonzero(img != 255, axis=0)[:, 0] > 0
                img = img.transpose(1, 0, 2)[nonzero_columns].transpose(1, 0, 2)
                images.append(img)

            out_file = os.path.join(output_directory, '{}_{}.jpg'.format(vis_range[0], vis_range[-1]))
            print('saving chart to {}'.format(out_file))
            Image.fromarray(np.hstack(images)).save(out_file)

    @torch.no_grad()
    def create_visualization(self, z, vis_range, shifts_range, shifts_count=5, **kwargs):
        self.g.eval()
        original_img = self.g(z).cpu()
        images = []

        for i in vis_range:
            images.append(self.create_shifted_images(z=z,
                                                     shifts_range=shifts_range,
                                                     shifts_count=5,
                                                     dim=i))

        rows_count = len(images) + 1
        fig, axs = plt.subplots(rows_count, **kwargs)

        axs[0].axis('off')
        axs[0].imshow(to_image(original_img))

        texts = vis_range
        for ax, shifts_images, text in zip(axs[1:], images, texts):
            ax.axis('off')
            plt.subplots_adjust(left=0.5)
            ax.imshow(to_image(make_grid(shifts_images, nrow=(2 * shifts_count + 1), padding=1)))
            ax.text(-20, 21, str(text), fontsize=10)

        self.matrix_a.train()

        return fig

    @torch.no_grad()
    def create_shifted_images(self, z, shifts_range, shifts_count, dim):
        shifted_images = []
        for shift in np.arange(-shifts_range, shifts_range + 1e-9, shifts_range / shifts_count):
            latent_shift = self.matrix_a(one_hot(dims=self.matrix_a.input_dim, value=shift, index=dim).to(self.device))
            shifted_image = self.g.gen_shifted(z, latent_shift).cpu()[0]
            shifted_images.append(shifted_image)

        return shifted_images
