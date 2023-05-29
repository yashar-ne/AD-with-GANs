import torch
import numpy as np
import base64
import io
from PIL import Image

from src.ml.latent_direction_visualizer import LatentDirectionVisualizer
from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.backend.models.ImageStripModel import ImageStripModel


class MainController:
    def __init__(self, device, generator_path, z_dim):
        self.z_dim = z_dim
        self.device = device
        self.g: Generator = Generator(size_z=self.z_dim, num_feature_maps=64, num_color_channels=1)
        self.g.load_state_dict(torch.load(generator_path, map_location=torch.device(self.device)))
        self.matrix_a: MatrixALinear = MatrixALinear(input_dim=self.z_dim, output_dim=self.z_dim)

    def get_image_strip(self, z, shifts_range, shifts_count, dim):
        image_list = []
        visualizer = LatentDirectionVisualizer(matrix_a=self.matrix_a, generator=self.g, device=self.device)
        shifted_images = visualizer.create_shifted_images(z, shifts_range, shifts_count, dim)

        for idx, i in enumerate(shifted_images):
            two_d = (np.reshape(i, (28, 28)) * 255).astype(np.uint8)
            img = Image.fromarray(two_d, 'L')

            with io.BytesIO() as buf:
                img.save(buf, format='PNG')
                img_str = base64.b64encode(buf.getvalue())

            image_list.append(ImageStripModel(position=idx, image=img_str))

        return image_list
