import base64
import io
import os
from operator import itemgetter

from PIL import Image

import torch
from matplotlib import pyplot as plt

from src.ml.models.celebA.celeb_generator import CelebGenerator
from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.ml.validation import load_data_points, get_roc_auc_for_average_distance_metric

base_path = '../data/DS5_celebA_bald'
generator_path = os.path.join(base_path, 'generator.pkl')
matrix_a_path = os.path.join(base_path,
                             'direction_matrices/direction_matrix_steps_2000_bias_k_30.pkl')  # '../data/DS3_mnist_9_6_10_percent/direction_matrices/direction_matrix_steps_1000_bias_k_20.pkl'
z_dim = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g: CelebGenerator = CelebGenerator(size_z=z_dim, num_feature_maps=64)
g.load_state_dict(torch.load(generator_path, map_location=torch.device(device)))
matrix_a_linear: MatrixALinear = MatrixALinear(input_dim=30, output_dim=100, bias=True)
matrix_a_linear.load_state_dict(torch.load(matrix_a_path, map_location=torch.device(device)))
latent_space_data_points, latent_space_data_labels = load_data_points(
    os.path.join(base_path, 'dataset'))  # '../data/DS3_mnist_9_6_10_percent/dataset'


def display_base64_png(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    img.show()


# [[1, 1], [14, 1], [17, 1], [25, 1], [26, 1]]
aucs = []
for i in range(30):
    _, auc = get_roc_auc_for_average_distance_metric(direction_matrix=matrix_a_linear,
                                                     anomalous_directions=[(i, 1)],
                                                     latent_space_data_points=latent_space_data_points,
                                                     latent_space_data_labels=latent_space_data_labels)
    plt.close()
    aucs.append([i, 1, auc])
    _, auc = get_roc_auc_for_average_distance_metric(direction_matrix=matrix_a_linear,
                                                     anomalous_directions=[(i, -1)],
                                                     latent_space_data_points=latent_space_data_points,
                                                     latent_space_data_labels=latent_space_data_labels)
    plt.close()
    aucs.append([i, -1, auc])

sorted_aucs = sorted(aucs, key=itemgetter(2))
print(sorted_aucs)
print(f'Highest performing direction {sorted_aucs[-1]}')
