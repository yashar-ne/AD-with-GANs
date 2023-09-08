import base64
import io

import matplotlib
from PIL import Image

import torch

from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.ml.validation import load_data_points, get_lof_roc_auc_for_given_dims, \
    get_roc_auc_for_average_distance_metric
from itertools import chain, combinations

generator_path = '/home/yashar/git/AD-with-GANs/data/DS3_mnist_9_6_10_percent/generator.pkl'
matrix_a_path = '/home/yashar/git/AD-with-GANs/data/DS3_mnist_9_6_10_percent/direction_matrices/direction_matrix_steps_1000_bias_k_20.pkl'
z_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g: Generator = Generator(size_z=z_dim, num_feature_maps=64, num_color_channels=1)
g.load_state_dict(torch.load(generator_path, map_location=torch.device(device)))
matrix_a_linear: MatrixALinear = MatrixALinear(input_dim=20, output_dim=100, bias=True)
matrix_a_linear.load_state_dict(torch.load(matrix_a_path, map_location=torch.device(device)))
latent_space_data_points, latent_space_data_labels = load_data_points(
    '/home/yashar/git/AD-with-GANs/data/DS3_mnist_9_6_10_percent/dataset')


def display_base64_png(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    img.show()


_, auc = get_roc_auc_for_average_distance_metric(direction_matrix=matrix_a_linear,
                                                 anomalous_directions=[(2, 1), (16, 1)],
                                                 latent_space_data_points=latent_space_data_points,
                                                 latent_space_data_labels=latent_space_data_labels)
print(auc)