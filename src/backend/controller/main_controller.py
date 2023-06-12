import os
import torch
import numpy as np
import base64
import io
from PIL import Image

from src.backend.db import save_to_db
from src.ml.latent_direction_visualizer import LatentDirectionVisualizer, get_random_strip_as_numpy_array
from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.backend.models.ImageStripModel import ImageStripModel
from src.ml.tools.utils import generate_noise
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class MainController:
    def __init__(self, generator_path, matrix_a_path, z_dim, bias=False):
        self.z_dim = z_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.g: Generator = Generator(size_z=self.z_dim, num_feature_maps=64, num_color_channels=1)
        self.g.load_state_dict(torch.load(generator_path, map_location=torch.device(self.device)))
        self.matrix_a_linear: MatrixALinear = MatrixALinear(input_dim=100, output_dim=100, bias=bias)
        self.matrix_a_linear.load_state_dict(torch.load(matrix_a_path, map_location=torch.device(self.device)))

    @staticmethod
    def get_image_strip():
        image_list = []
        img_arr = get_random_strip_as_numpy_array(os.path.abspath("../out_dir/data.npy"))
        for idx, i in enumerate(img_arr):
            two_d = (np.reshape(i, (28, 28)) * 255).astype(np.uint8)
            img = Image.fromarray(two_d, 'L')

            with io.BytesIO() as buf:
                img.save(buf, format='PNG')
                img_str = base64.b64encode(buf.getvalue())

            image_list.append(ImageStripModel(position=idx, image=img_str))

        return image_list

    def get_shifted_images(self, z, shifts_range, shifts_count, dim, pca_component_count=0,
                           pca_skipped_components_count=0, pca_apply_standard_scaler=False):
        image_list = []
        z = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(z), 0), -1), 2)

        if pca_component_count > 0:
            a = self.__apply_pca(pca_component_count, pca_skipped_components_count, pca_apply_standard_scaler)
        else:
            a = self.matrix_a_linear

        visualizer = LatentDirectionVisualizer(matrix_a_linear=a, generator=self.g, device=self.device)
        shifted_images = visualizer.create_shifted_images(z, shifts_range, shifts_count, dim)

        for idx, i in enumerate(shifted_images):
            two_d = (np.reshape(i.numpy(), (28, 28)) * 255).astype(np.uint8)
            img = Image.fromarray(two_d, 'L')

            with io.BytesIO() as buf:
                img.save(buf, format='PNG')
                img_str = base64.b64encode(buf.getvalue())

            image_list.append(ImageStripModel(position=idx, image=img_str))

        return image_list

    def __apply_pca(self, component_count, skipped_components_count, apply_standard_scaler):
        matrix_a_np = self.matrix_a_linear.linear.weight.data.numpy()

        if apply_standard_scaler:
            matrix_a_np = StandardScaler().fit_transform(matrix_a_np)
        pca = PCA(n_components=component_count+skipped_components_count)
        principal_components = pca.fit_transform(matrix_a_np)
        principal_components = principal_components[:, skipped_components_count:]

        new_weights = torch.from_numpy(principal_components)
        matrix_a_linear_after_pca = MatrixALinear(input_dim=component_count, output_dim=100, bias=False)
        new_dict = {
            'linear.weight': new_weights
        }
        matrix_a_linear_after_pca.load_state_dict(new_dict)

        return matrix_a_linear_after_pca

    @staticmethod
    def save_to_db(z, shifts_range, shifts_count, dim, is_anomaly):
        save_to_db(z=z, shifts_range=shifts_range, shifts_count=shifts_count, dim=dim, is_anomaly=is_anomaly)

    def get_random_noise(self, z_dim):
        return generate_noise(batch_size=1, z_dim=z_dim, device=self.device)
