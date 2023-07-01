import os
import torch
import numpy as np
import base64
import io
from PIL import Image

from src.backend.db import save_to_db, save_session_labels_to_db
from src.backend.models.SaveLabelToDbModel import SaveLabelToDbModel
from src.backend.models.SessionLabelsModel import SessionLabelsModel
from src.ml.latent_direction_visualizer import LatentDirectionVisualizer, get_random_strip_as_numpy_array
from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.backend.models.ImageStripModel import ImageStripModel
from src.ml.tools.utils import generate_noise, apply_pca_to_matrix_a, generate_base64_images_from_tensor_list, \
    generate_base64_images_from_tensor
from src.ml.validation import load_latent_space_data_points, get_roc_auc_for_given_dims


class MainController:
    def __init__(self, generator_path, matrix_a_path, z_dim, bias=False):
        self.z_dim = z_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.g: Generator = Generator(size_z=self.z_dim, num_feature_maps=64, num_color_channels=1)
        self.g.load_state_dict(torch.load(generator_path, map_location=torch.device(self.device)))
        self.matrix_a_linear: MatrixALinear = MatrixALinear(input_dim=100, output_dim=100, bias=bias)
        self.matrix_a_linear.load_state_dict(torch.load(matrix_a_path, map_location=torch.device(self.device)))
        self.latent_space_data_points, self.latent_space_data_labels = load_latent_space_data_points(
            '../data/LatentSpaceMNIST')

    def get_shifted_images(self, z, shifts_range, shifts_count, dim, pca_component_count=0,
                           pca_skipped_components_count=0, pca_apply_standard_scaler=False):
        z = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(z), 0), -1), 2)
        a = apply_pca_to_matrix_a(self.matrix_a_linear, pca_component_count, pca_skipped_components_count,
                                  pca_apply_standard_scaler) \
            if pca_component_count > 0 \
            else self.matrix_a_linear

        visualizer = LatentDirectionVisualizer(matrix_a_linear=a, generator=self.g, device=self.device)
        shifted_images = visualizer.create_shifted_images(z, shifts_range, shifts_count, dim)

        return generate_base64_images_from_tensor_list(shifted_images)

    def get_shifted_image_from_dimension_labels(self, data: SessionLabelsModel, pca_component_count=0,
                                                pca_skipped_components_count=0, pca_apply_standard_scaler=False):
        a = apply_pca_to_matrix_a(self.matrix_a_linear, pca_component_count, pca_skipped_components_count,
                                  pca_apply_standard_scaler) \
            if pca_component_count > 0 \
            else self.matrix_a_linear
        visualizer = LatentDirectionVisualizer(matrix_a_linear=a, generator=self.g, device=self.device)
        shifted_images = visualizer.create_shifted_image_from_dimension_labels(data)

        return [ImageStripModel(position=0, image=generate_base64_images_from_tensor(shifted_images))]

    def get_roc_auc_for_given_dims(self, weighted_dims, pca_component_count, skipped_components_count, n_neighbours):
        base64_jpeg, _ = get_roc_auc_for_given_dims(weighted_dims=weighted_dims,
                                                    latent_space_data_points=self.latent_space_data_points,
                                                    latent_space_data_labels=self.latent_space_data_labels,
                                                    pca_component_count=pca_component_count,
                                                    skipped_components_count=skipped_components_count,
                                                    n_neighbours=n_neighbours)
        return base64_jpeg

    @staticmethod
    def get_image_strip_from_prerendered_sample():
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

    @staticmethod
    def save_to_db(z, shifts_range, shifts_count, dim, is_anomaly):
        save_to_db(z=z, shifts_range=shifts_range, shifts_count=shifts_count, dim=dim, is_anomaly=is_anomaly)

    @staticmethod
    def save_session_labels_to_db(session_labels: SessionLabelsModel):
        save_session_labels_to_db(session_labels=session_labels)

    def get_random_noise(self, z_dim):
        return generate_noise(batch_size=1, z_dim=z_dim, device=self.device)
