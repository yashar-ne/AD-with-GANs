import base64
import io
import os
from operator import itemgetter

from PIL import Image

import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import imshow

from src.ml.models.base.matrix_a_linear import MatrixALinear
from src.ml.models.mvtec128.mvtec_generator import MvTecGenerator
from src.ml.tools.utils import one_hot
from src.ml.validation import load_data_points, get_roc_auc_for_average_distance_metric

base_path = '../data/DS9_mvtec_hazelnut'
generator_path = os.path.join(base_path, 'generator.pkl')
direction_matrix_path = 'direction_matrices/direction_matrix_steps_2000_bias_k_10.pkl'
full_direction_matrix_path = os.path.join(base_path, direction_matrix_path)

num_directions = int(direction_matrix_path.split(".")[0].split("_")[-1])
z_dim = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
g = MvTecGenerator(size_z=z_dim, num_feature_maps=64)
g.load_state_dict(torch.load(generator_path, map_location=torch.device(device)))
direction_matrix: MatrixALinear = MatrixALinear(input_dim=num_directions, output_dim=100, bias=True)
direction_matrix.load_state_dict(torch.load(full_direction_matrix_path, map_location=torch.device(device)))
latent_space_data_points, latent_space_data_labels = load_data_points(
    os.path.join(base_path, 'dataset'))  # '../data/DS3_mnist_9_6_10_percent/dataset'


def display_base64_png(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    img = Image.open(io.BytesIO(imgdata))
    img.show()


aucs = []
for i in range(num_directions):
    _, auc = get_roc_auc_for_average_distance_metric(direction_matrix=direction_matrix,
                                                     anomalous_directions=[(i, 1)],
                                                     latent_space_data_points=latent_space_data_points,
                                                     latent_space_data_labels=latent_space_data_labels)
    plt.close()
    aucs.append([i, 1, auc])
    _, auc = get_roc_auc_for_average_distance_metric(direction_matrix=direction_matrix,
                                                     anomalous_directions=[(i, -1)],
                                                     latent_space_data_points=latent_space_data_points,
                                                     latent_space_data_labels=latent_space_data_labels)
    plt.close()
    aucs.append([i, -1, auc])

sorted_aucs = sorted(aucs, key=itemgetter(2), reverse=True)
print(sorted_aucs)
print(f'\n Highest performing direction {sorted_aucs[0]}')

combine_n_highest = 3
anomalous_directions = []
for i in range(combine_n_highest):
    anomalous_directions.append((sorted_aucs[i][0], sorted_aucs[i][1]))

_, auc = get_roc_auc_for_average_distance_metric(direction_matrix=direction_matrix,
                                                 anomalous_directions=anomalous_directions,
                                                 latent_space_data_points=latent_space_data_points,
                                                 latent_space_data_labels=latent_space_data_labels)

print(f'Combined anomalous directions AUC: {auc}')


def print_shifted_image(noise, shift):
    direction_matrix.to(device)
    noise.to(device)
    g.to(device)

    img = g(noise).cpu()[0]
    img = (img * 0.5) + 0.5
    img = img.permute(1, 2, 0)
    imshow(img.detach().numpy())
    plt.show()

    for idx in range(combine_n_highest):
        shift_vector = one_hot(dims=num_directions, value=shift, index=sorted_aucs[idx][0]).to(device)
        latent_shift = direction_matrix(shift_vector)
        latent_shift.to(device)
        img = g.gen_shifted(noise, latent_shift).cpu()[0]
        img = (img * 0.5) + 0.5
        img = img.permute(1, 2, 0)
        imshow(img.detach().numpy())
        plt.show()


z = torch.randn(1, z_dim, 1, 1, device=device)
print_shifted_image(z, 3)
