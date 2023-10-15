import os
import torch
from fastapi import status, HTTPException

from src.backend.db import save_session_labels_to_db
from src.backend.models.SessionLabelsModel import SessionLabelsModel
from src.backend.models.ValidationResultsModel import ValidationResultsModel
from src.ml.latent_direction_visualizer import LatentDirectionVisualizer
from src.ml.models.celebA.celeb_generator import CelebGenerator
from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.ml.models.mvtec128.mvtec_generator import MvTecGenerator
from src.ml.models.stl10.stl10_generator import Stl10Generator
from src.ml.tools.utils import generate_noise, apply_pca_to_matrix_a, generate_base64_images_from_tensor_list, \
    generate_base64_image_from_tensor
from src.ml.validation import load_data_points, get_lof_roc_auc_for_given_dims, \
    get_roc_auc_for_average_distance_metric, get_lof_roc_auc_for_image_data


class MainController:
    def __init__(self, base_path, z_dim, bias=False):
        self.dataset_names = os.listdir(base_path)
        self.z_dim = z_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.datasets = {}

        for dataset_name in self.dataset_names:
            matrix_a_path = os.path.join(base_path, dataset_name, 'direction_matrices')
            generator_path = os.path.join(base_path, dataset_name, 'generator.pkl')
            direction_matrices = {}
            for f in os.listdir(matrix_a_path):
                if os.path.isfile(os.path.join(matrix_a_path, f)):
                    matrix_state = torch.load(os.path.join(matrix_a_path, f), map_location=torch.device(self.device))
                    input_dim = matrix_state.get('linear.weight').shape[1]
                    output_dim = matrix_state.get('linear.weight').shape[0]
                    uses_bias = 'linear.bias' in matrix_state
                    matrix_a_linear: MatrixALinear = MatrixALinear(input_dim=input_dim, output_dim=output_dim,
                                                                   bias=uses_bias)
                    matrix_a_linear.load_state_dict(matrix_state)
                    direction_matrices.update({f: {'matrix_a': matrix_a_linear, 'direction_count': input_dim}})

            g = self.get_generator_by_dataset_name(dataset_name)
            g.load_state_dict(torch.load(generator_path, map_location=torch.device(self.device)))
            self.datasets.update({
                dataset_name: {
                    'direction_matrices': direction_matrices,
                    'generator': g,
                    'data': load_data_points(os.path.join(base_path, dataset_name, 'dataset'))
                }
            })

    def get_single_image(self, dataset_name, z):
        z = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(z), 0), -1), 2).to(self.device)
        generator = self.datasets.get(dataset_name).get('generator').to(self.device)
        img = generator(z)
        return generate_base64_image_from_tensor(img[0])

    def get_shifted_images(self, dataset_name, direction_matrix_name, z, shifts_range, shifts_count, dim, direction,
                           pca_component_count=0, pca_skipped_components_count=0):
        z = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(z), 0), -1), 2).to(self.device)
        a = self.__get_direction_matrix(dataset_name, direction_matrix_name)
        a = apply_pca_to_matrix_a(a, pca_component_count,
                                  pca_skipped_components_count) if pca_component_count > 0 else a
        a.to(self.device)
        visualizer = LatentDirectionVisualizer(matrix_a_linear=a,
                                               generator=self.datasets.get(dataset_name).get('generator').to(
                                                   self.device),
                                               device=self.device)
        shifted_images = visualizer.create_shifted_images(z, shifts_range, shifts_count, dim, direction)
        return generate_base64_images_from_tensor_list(shifted_images)

    def list_available_datasets(self):
        return [(key, [key for key, m in value.get('direction_matrices').items()]) for key, value in
                self.datasets.items()]

    def get_direction_count(self, dataset_name: str, direction_matrix_name: str):
        return self.datasets.get(dataset_name).get('direction_matrices').get(direction_matrix_name).get(
            'direction_count')

    def get_validation_results(self, dataset_name, direction_matrix_name, anomalous_directions):

        direction_matrix = self.__get_direction_matrix(dataset_name=dataset_name,
                                                       direction_matrix_name=direction_matrix_name)
        latent_space_data_points = self.datasets.get(dataset_name).get('data')[0]
        latent_space_data_labels = self.datasets.get(dataset_name).get('data')[1]

        roc_auc, _ = get_roc_auc_for_average_distance_metric(
            direction_matrix=direction_matrix,
            anomalous_directions=anomalous_directions,
            latent_space_data_points=latent_space_data_points,
            latent_space_data_labels=latent_space_data_labels
        )

        roc_auc_plot_ignore_labels, _ = get_lof_roc_auc_for_given_dims(
            direction_matrix=direction_matrix,
            anomalous_directions=anomalous_directions,
            latent_space_data_points=latent_space_data_points,
            latent_space_data_labels=latent_space_data_labels,
            n_neighbours=20,
            use_default_distance_metric=True
        )

        # roc_auc_plot_image_data = get_lof_roc_auc_for_image_data(dataset_name=dataset_name, n_neighbours=20)

        if not roc_auc:
            raise HTTPException(
                status_code=status.HTTP_406_NOT_ACCEPTABLE,
                detail="Only ano to ano directions were labeled"
            )

        return ValidationResultsModel(
            roc_auc_plot_one_hot=roc_auc,
            roc_auc_plot_ignore_labels=roc_auc_plot_ignore_labels,
            # roc_auc_plot_image_data=roc_auc_plot_image_data
        )

    @staticmethod
    def save_session_labels_to_db(session_labels: SessionLabelsModel):
        save_session_labels_to_db(session_labels=session_labels)

    def get_random_noise(self, z_dim):
        return generate_noise(batch_size=1, z_dim=z_dim, device=self.device)

    def __get_direction_matrix(self, dataset_name, direction_matrix_name):
        return self.datasets.get(dataset_name) \
            .get('direction_matrices') \
            .get(direction_matrix_name) \
            .get('matrix_a')

    def get_generator_by_dataset_name(self, dataset_name):
        match dataset_name:
            case "DS5_celebA_bald":
                return CelebGenerator(size_z=self.z_dim, num_feature_maps=64)
            case "DS9_mvtec_hazelnut":
                return MvTecGenerator(size_z=self.z_dim, num_feature_maps=64)
            case "DS7_stl10_plane_horse":
                return Stl10Generator(size_z=self.z_dim, num_feature_maps=32)
            case _:
                return Generator(size_z=self.z_dim, num_feature_maps=64, num_color_channels=1)
