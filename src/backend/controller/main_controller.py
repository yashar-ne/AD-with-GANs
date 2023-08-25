import os
import torch
import numpy as np
import base64
import io
from PIL import Image

from src.backend.db import save_to_db, save_session_labels_to_db
from src.backend.models.SessionLabelsModel import SessionLabelsModel
from src.backend.models.ValidationResultsModel import ValidationResultsModel
from src.ml.latent_direction_visualizer import LatentDirectionVisualizer, get_random_strip_as_numpy_array
from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.backend.models.ImageStripModel import ImageStripModel
from src.ml.tools.utils import generate_noise, apply_pca_to_matrix_a, generate_base64_images_from_tensor_list, \
    generate_base64_images_from_tensor, extract_weights_from_model_and_apply_pca
from src.ml.validation import load_latent_space_data_points_ano_class, get_lof_roc_auc_for_given_dims, get_tsne_for_original_data, \
    get_roc_auc_for_average_distance_metric


class MainController:
    def __init__(self, generator_path, matrix_a_path, z_dim, bias=False):
        self.z_dim = z_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.g: Generator = Generator(size_z=self.z_dim, num_feature_maps=64, num_color_channels=1)
        self.g.load_state_dict(torch.load(generator_path, map_location=torch.device(self.device)))
        self.matrix_a_linear: MatrixALinear = MatrixALinear(input_dim=100, output_dim=100, bias=bias)
        self.matrix_a_linear.load_state_dict(torch.load(matrix_a_path, map_location=torch.device(self.device)))
        self.latent_space_data_points, self.latent_space_data_labels = load_latent_space_data_points_ano_class(
            '../data/LatentSpaceMNIST')

    def get_shifted_images(self, z, shifts_range, shifts_count, dim, direction, pca_component_count=0,
                           pca_skipped_components_count=0):
        z = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(z), 0), -1), 2)
        a = apply_pca_to_matrix_a(self.matrix_a_linear, pca_component_count, pca_skipped_components_count) \
            if pca_component_count > 0 \
            else self.matrix_a_linear

        visualizer = LatentDirectionVisualizer(matrix_a_linear=a, generator=self.g, device=self.device)
        shifted_images = visualizer.create_shifted_images(z, shifts_range, shifts_count, dim, direction)

        return generate_base64_images_from_tensor_list(shifted_images)

    def get_validation_results(self, anomalous_directions, pca_component_count, skipped_components_count):
        roc_auc, _ = get_roc_auc_for_average_distance_metric(
            direction_matrix=self.matrix_a_linear,
            anomalous_directions=anomalous_directions,
            latent_space_data_points=self.latent_space_data_points,
            latent_space_data_labels=self.latent_space_data_labels,
            pca_component_count=pca_component_count,
            pca_skipped_components_count=skipped_components_count,
        )

        # roc_auc_inverted, _ = get_roc_auc_for_average_distance_metric(
        #     direction_matrix=self.matrix_a_linear,
        #     anomalous_directions=anomalous_directions,
        #     latent_space_data_points=self.latent_space_data_points,
        #     latent_space_data_labels=self.latent_space_data_labels,
        #     pca_component_count=pca_component_count,
        #     pca_skipped_components_count=skipped_components_count,
        #     invert_labels=True
        # )

        # roc_auc, _ = get_lof_roc_auc_for_given_dims(
        #     direction_matrix=self.matrix_a_linear,
        #     anomalous_directions=anomalous_directions,
        #     latent_space_data_points=self.latent_space_data_points,
        #     latent_space_data_labels=self.latent_space_data_labels,
        #     pca_component_count=pca_component_count,
        #     pca_skipped_components_count=skipped_components_count,
        #     n_neighbours=n_neighbours
        # )

        roc_auc_ignore_labels, _ = get_lof_roc_auc_for_given_dims(
            direction_matrix=self.matrix_a_linear,
            anomalous_directions=anomalous_directions,
            latent_space_data_points=self.latent_space_data_points,
            latent_space_data_labels=self.latent_space_data_labels,
            pca_component_count=pca_component_count,
            pca_skipped_components_count=skipped_components_count,
            n_neighbours=20,
            use_default_distance_metric=True
        )

        # roc_auc_plain_mahalanobis, _ = get_roc_auc_for_plain_mahalanobis_distance(
        #     direction_matrix=self.matrix_a_linear,
        #     anomalous_directions=anomalous_directions,
        #     pca_component_count=pca_component_count,
        #     pca_skipped_components_count=skipped_components_count
        # )
        #
        # t_sne_plot_one_hot_weighted_data = get_tsne_with_dimension_weighted_metric(
        #     weighted_dims=labeled_dims,
        #     ignore_unlabeled_dims=True,
        #     pca_component_count=pca_component_count,
        #     skipped_components_count=skipped_components_count
        # )
        #
        # t_sne_plot_one_hot_weighted_data_ignore_labels = get_tsne_with_dimension_weighted_metric(
        #     weighted_dims=labeled_dims,
        #     ignore_unlabeled_dims=True,
        #     pca_component_count=pca_component_count,
        #     skipped_components_count=skipped_components_count,
        #     ignore_labels=True
        # )

        return ValidationResultsModel(
            roc_auc_plot_one_hot=roc_auc,
            # roc_auc_inverted=roc_auc_inverted,
            roc_auc_plot_ignore_labels=roc_auc_ignore_labels,
            # roc_auc_plot_one_hot_plain_mahalanobis=roc_auc_plain_mahalanobis,
            # t_sne_plot_original_input_data=get_tsne_for_original_data(),
            # t_sne_plot_one_hot_weighted_data=t_sne_plot_one_hot_weighted_data,
            # t_sne_plot_one_hot_weighted_data_ignore_labels=t_sne_plot_one_hot_weighted_data_ignore_labels,
        )

    @staticmethod
    def save_session_labels_to_db(session_labels: SessionLabelsModel):
        save_session_labels_to_db(session_labels=session_labels)

    def get_random_noise(self, z_dim):
        return generate_noise(batch_size=1, z_dim=z_dim, device=self.device)
