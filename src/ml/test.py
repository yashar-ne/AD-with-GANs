import base64
import io

import matplotlib
from PIL import Image

import torch

from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.ml.validation import load_latent_space_data_points, get_roc_auc_for_given_dims
from itertools import chain, combinations

generator_path = '/home/yashar/git/python/AD-with-GANs/saved_models/generator.pkl'
matrix_a_path = '/home/yashar/git/python/AD-with-GANs/saved_models/matrix_a.pkl'
z_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g: Generator = Generator(size_z=z_dim, num_feature_maps=64, num_color_channels=1)
g.load_state_dict(torch.load(generator_path, map_location=torch.device(device)))
matrix_a_linear: MatrixALinear = MatrixALinear(input_dim=100, output_dim=100, bias=False)
matrix_a_linear.load_state_dict(torch.load(matrix_a_path, map_location=torch.device(device)))
latent_space_data_points, latent_space_data_labels = load_latent_space_data_points(
    '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')


def display_base64_png(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    img.show()


_, auc_lof = get_roc_auc_for_given_dims(direction_matrix=matrix_a_linear,
                                        # anomalous_directions=[0, 1, 2],
                                        anomalous_directions=range(0, 100),
                                        # anomalous_directions=[2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,
                                        #                       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
                                        #                       33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 47, 50,
                                        #                       51, 52, 53, 54, 57, 59, 60, 61, 62, 64, 65, 67, 69, 71,
                                        #                       72, 73, 74, 76, 77, 78, 80, 81, 84, 85, 86, 88, 89, 90,
                                        #                       92, 93, 94, 95, 96, 97, 98, 99],
                                        latent_space_data_points=latent_space_data_points,
                                        latent_space_data_labels=latent_space_data_labels,
                                        pca_component_count=0,
                                        pca_skipped_components_count=2,
                                        n_neighbours=20,
                                        pca_apply_standard_scaler=True,
                                        use_default_distance_metric=False)
print(auc_lof)

# auc_plain_mahalanobis = get_auc_value_plain_mahalanobis_distance(matrix_a_linear=matrix_a_linear,
#                                                                  # anomalous_directions=[9, 16, 17, 18, 20, 22, 23, 27,
#                                                                  #                       28, 33, 34, 35, 36, 47, 50, 51,
#                                                                  #                       57, 59, 60, 70, 75, 76, 80, 84,
#                                                                  #                       85, 88, 98, 99],
#                                                                  # anomalous_directions=range(0, 100),
#                                                                  anomalous_directions=[0, 1, 7, 8, 18, 19, 20, 21, 23, 26, 28],
#                                                                  # anomalous_directions=[0, 1, 2, 5, 6],
#                                                                  # anomalous_directions=range(0, 100),
#                                                                  pca_component_count=30,
#                                                                  pca_skipped_components_count=2,
#                                                                  pca_apply_standard_scaler=False)
# print(auc_plain_mahalanobis)
# display_base64_png(plot1)
