import base64
import io
import math
import os
import numpy as np
import seaborn
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import csv
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

from src.ml.tools.utils import extract_weights_from_model_and_apply_pca
from src.ml.weighted_local_outlier_factor import WeightedLocalOutlierFactor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Reasoning
# x = Inlier
# Klasse für d wird gesucht
# Richtungsvektoren alpha_0, ..., alpha_n

# min_(lambda) | d + lambda * alpha_0 - x |

# lambda ist groß = outlier
# lambda ist klein = inlier

# min_(lambda) (d + lambda * alpha_0 - x)^2
# d / d_lambda
#    --> 2 * (d + lambda * alpha_0 - x) * alpha_0 = 0

# lambda * alpha_0 * alpha_0 = (x - d) * alpha_0

# lambda = (x - d) * alpha_0^(-1)
# lambda = (x - d) * alpha_0

# x = sum d_i * 1/N
# N = 50

# max(lambda_0, lambda_1, lambda_2)
def get_roc_auc_for_average_distance_metric(latent_space_data_points, latent_space_data_labels, direction_matrix,
                                            anomalous_directions, pca_component_count=3,
                                            pca_skipped_components_count=0, invert_labels=False):

    if invert_labels:
        inverted_anomalous_directions = []
        for i in range(pca_skipped_components_count if pca_component_count > 0 else 100):
            if (i, 1) not in anomalous_directions: inverted_anomalous_directions.append((i, 1))
            # if (i, -1) not in anomalous_directions: inverted_anomalous_directions.append((i, -1))
        anomalous_directions = inverted_anomalous_directions

    consider_ano_to_ano_directions = True
    inlier = []

    latent_space_data_points = normalize(latent_space_data_points, axis=1, norm='l2')
    for idx, p in enumerate(latent_space_data_points):
        if latent_space_data_labels[idx] is False:
            inlier.append(p / np.linalg.norm(p) if np.linalg.norm(p) != 0 else p)
    average_inlier_vector = np.mean(inlier, axis=0)

    a = extract_weights_from_model_and_apply_pca(direction_matrix, pca_component_count, pca_skipped_components_count)
    directions = [a[d[0]] for d in anomalous_directions] if consider_ano_to_ano_directions \
        else [a[d[0]] for d in anomalous_directions if (d[0], d[1]*-1) not in anomalous_directions]

    scores = []
    for data_point in latent_space_data_points:
        direction_scores = []
        for d in directions:
            direction_scores.append((average_inlier_vector - data_point) @ d)
        scores.append(max(direction_scores))

    y = np.array([-1 if d is False else 1 for d in latent_space_data_labels])
    return get_roc_curve_as_base64(y, scores)


def get_lof_roc_auc_for_given_dims(direction_matrix,
                                   anomalous_directions,
                                   latent_space_data_points,
                                   latent_space_data_labels,
                                   pca_component_count,
                                   pca_skipped_components_count,
                                   n_neighbours,
                                   use_default_distance_metric=False):
    a = extract_weights_from_model_and_apply_pca(direction_matrix, pca_component_count, pca_skipped_components_count)
    weighted_lof = WeightedLocalOutlierFactor(direction_matrix=a,
                                              anomalous_directions=anomalous_directions,
                                              n_neighbours=n_neighbours,
                                              pca_component_count=pca_component_count,
                                              pca_skipped_components_count=pca_skipped_components_count,
                                              use_default_distance_metric=use_default_distance_metric)

    weighted_lof.load_latent_space_datapoints(data=latent_space_data_points)
    weighted_lof.fit()

    y = np.array([1 if d is False else -1 for d in latent_space_data_labels])
    # get_3d_plot(local_outlier_factor=weighted_lof)
    # get_2d_plot(local_outlier_factor=weighted_lof)
    return get_roc_curve_as_base64(y, weighted_lof.get_negative_outlier_factor())


def plot_to_base64(plot):
    io_bytes = io.BytesIO()
    plot.savefig(io_bytes, format='jpg')
    io_bytes.seek(0)
    return base64.b64encode(io_bytes.read()).decode()


def get_2d_plot(local_outlier_factor):
    seaborn.set_style("darkgrid")
    plt.figure(figsize=(10, 9))
    axes = plt.axes()

    data_points, data_label = load_latent_space_data_points(
        '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')

    data = []
    for idx, p in enumerate(data_points):
        p_star = p.T @ local_outlier_factor.get_labeled_directions_matrix().T
        data.append((p_star, data_label[idx]))

    x1 = [p[0][0] for p in data if p[1] == 'True']
    y1 = [p[0][1] for p in data if p[1] == 'True']
    x2 = [p[0][0] for p in data if p[1] == 'False']
    y2 = [p[0][1] for p in data if p[1] == 'False']
    axes.scatter(x1, y1, color="green")
    # axes.scatter(x2, y2, color="red")

    axes.arrow(data[0][0], data[1][0], data[1][0], data[1][1], head_width=0.5, head_length=1)

    plt.xlim(-20000, 20000)
    plt.ylim(-20000, 20000)

    axes.set_xlabel('x')
    axes.set_ylabel('y')
    plt.show()


def get_3d_plot(local_outlier_factor):
    seaborn.set_style("darkgrid")
    plt.figure(figsize=(10, 9))
    axes = plt.axes(projection='3d')
    axes.set_xlim3d(-20000, 20000)
    axes.set_ylim3d(-20000, 20000)
    axes.set_zlim3d(-20000, 20000)

    data_points, data_label = load_latent_space_data_points(
        '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')

    data = []
    for idx, p in enumerate(data_points):
        p_star = p.T @ local_outlier_factor.get_labeled_directions_matrix().T
        data.append((p_star, data_label[idx]))

    x1 = [p[0][0] for p in data if p[1] == 'True']
    y1 = [p[0][1] for p in data if p[1] == 'True']
    z1 = [p[0][2] for p in data if p[1] == 'True']
    x2 = [p[0][0] for p in data if p[1] == 'False']
    y2 = [p[0][1] for p in data if p[1] == 'False']
    z2 = [p[0][2] for p in data if p[1] == 'False']
    # axes.scatter(x1, y1, z1, color="green")
    axes.scatter(x2, y2, z2, color="red")
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    plt.show()


def get_roc_curve_as_base64(label, values):
    plt.clf()
    values = [0 if math.isnan(x) else x for x in values]
    fpr, tpr, thresholds = metrics.roc_curve(label, values)
    auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
                                      estimator_name='LOF')
    display.plot()
    return plot_to_base64(plt), auc


def load_latent_space_data_points(base_url):
    path = os.path.join(base_url, "latent_space_mappings.csv")
    data_points = []
    data_labels = []
    with open(path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            latent_space_point_path = os.path.join(base_url, row[0])
            latent_space_point_pt = torch.squeeze(
                torch.load(latent_space_point_path, map_location=torch.device(device)).detach())
            latent_space_point = latent_space_point_pt.numpy()
            data_points.append(latent_space_point)
            data_labels.append(True if int(row[1]) == 6 else False)

    return data_points, data_labels


def get_tsne_for_original_data():
    plt.clf()
    data_points, data_label = load_latent_space_data_points(
        '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')
    tsne = TSNE(n_components=2, random_state=0)
    tsne_res = tsne.fit_transform(np.array(data_points))
    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=data_label, palette=sns.hls_palette(2), legend='full')
    return plot_to_base64(plt)
