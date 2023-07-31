import base64
import io
import math
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy as np
import scipy.spatial.distance
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import csv
from scipy.spatial import distance

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchvision.transforms import transforms

from src.ml.models.generator import Generator
from src.ml.tools.ano_mnist_dataset_generator import get_ano_mnist_dataset
from src.ml.tools.utils import apply_pca_to_matrix_a, extract_weights_from_model_and_apply_pca
from src.ml.weighted_local_outlier_factor import WeightedLocalOutlierFactor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
global_weights = [1, 1, 0, 1]


def create_roc_curve(label, lofs_in):
    plt.clf()
    fpr, tpr, thresholds = metrics.roc_curve(label, lofs_in)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name='LOF')
    display.plot()
    plt.show()


def plot_to_base64(plot):
    io_bytes = io.BytesIO()
    plot.savefig(io_bytes, format='jpg')
    io_bytes.seek(0)
    return base64.b64encode(io_bytes.read()).decode()


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
            data_labels.append(row[2])

    return data_points, data_labels


def get_ano_mnist_data(base_url, num=1308):
    saved_path = os.path.join(base_url, f'ano_mnist_data_raw_{num}.pt')
    if not os.path.exists(saved_path):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(.5,), std=(.5,))
        ])
        dat, _ = get_ano_mnist_dataset(root_dir=base_url, transform=transform)
        torch.save(dat, saved_path)
    else:
        dat = torch.load(saved_path, map_location=torch.device(device))

    counter = 0
    X = []
    y = []
    for d in dat:
        if counter > num: break
        X.append(d[0].numpy().reshape(784))
        y.append(d[1]['anomaly'])
        counter += 1

    return X, y


# TODO: implement pca_apply_standard_scaler in API/controller/frontend
def get_roc_auc_for_given_dims(direction_matrix, anomalous_directions, latent_space_data_points,
                               latent_space_data_labels,
                               pca_component_count,
                               pca_skipped_components_count, n_neighbours, pca_apply_standard_scaler=True,
                               weight_factor=10, one_hot_weighing=True, use_default_distance_metric=False):
    a = extract_weights_from_model_and_apply_pca(direction_matrix, pca_component_count, pca_skipped_components_count,
                                                 pca_apply_standard_scaler)
    weighted_lof = WeightedLocalOutlierFactor(direction_matrix=a,
                                              anomalous_directions=anomalous_directions,
                                              n_neighbours=n_neighbours,
                                              pca_component_count=pca_component_count,
                                              pca_skipped_components_count=pca_skipped_components_count,
                                              use_default_distance_metric=use_default_distance_metric)

    weighted_lof.load_latent_space_datapoints(data=latent_space_data_points)
    weighted_lof.fit()

    y = np.array([1 if d == "False" else -1 for d in latent_space_data_labels])
    return get_roc_curve_as_base64(y, weighted_lof.get_negative_outlier_factor())


def get_data_for_plain_mahalanobis_distance(matrix_a_linear, anomalous_directions, pca_component_count,
                                            pca_skipped_components_count, pca_apply_standard_scaler=False):

    a = extract_weights_from_model_and_apply_pca(matrix_a_linear,
                                                 pca_component_count,
                                                 pca_skipped_components_count,
                                                 pca_apply_standard_scaler)

    labeled_directions = []

    # Remove directions that were not labeled
    for idx, direction in enumerate(a):
        if idx in anomalous_directions:
            labeled_directions.append(direction)

    # Weigh down directions that were not labeled
    # for idx, direction in enumerate(a):
    #     if idx not in anomalous_directions:
    #         labeled_directions.append(direction * 0.1)
    #     else:
    #         labeled_directions.append(direction * 0.5)

    # Replace normal directions with zero-vectors
    # for idx, direction in enumerate(a):
    #     if idx not in anomalous_directions:
    #         labeled_directions.append(np.zeros_like(direction))
    #     else:
    #         labeled_directions.append(direction)

    labeled_directions = np.array(labeled_directions)
    test_data_points, test_data_label = load_latent_space_data_points(
        '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')

    # Scale Data
    # labeled_directions = StandardScaler().fit_transform(labeled_directions)
    # test_data_points = StandardScaler().fit_transform(test_data_points)

    # Apply PCA on data_points
    # if pca_component_count > 0:
    #     pca = PCA(n_components=pca_component_count + pca_skipped_components_count)
    #     principal_components = pca.fit_transform(test_data_points)
    #     test_data_points = principal_components[:, pca_skipped_components_count:]
    #     print(pca.explained_variance_ratio_)

    # Use labeled directions as reference
    cov = np.cov(labeled_directions.T)
    vi = np.linalg.inv(cov)
    mean_vector = np.mean(labeled_directions, axis=0)

    # Use data distribution as reference
    # data = np.array(test_data_points)
    # v = np.cov(data.T)
    # vi = np.linalg.inv(v)
    # mean_vector = np.mean(data, axis=0)

    distance_list = []
    label_list = []
    for idx, point in enumerate(test_data_points):
        dist = mahalanobis_distance(u=point, mean=mean_vector, vi=vi)
        if not np.isnan(dist):
            distance_list.append(dist)
            label_list.append(1 if test_data_label[idx] == "True" else -1)

    # y = np.array([1 if d == "True" else -1 for d in test_data_label])
    return label_list, distance_list


def get_roc_auc_for_plain_mahalanobis_distance(direction_matrix, anomalous_directions, pca_component_count,
                                               pca_skipped_components_count, pca_apply_standard_scaler=True):
    label_list, distance_list = get_data_for_plain_mahalanobis_distance(matrix_a_linear=direction_matrix,
                                                                        anomalous_directions=anomalous_directions,
                                                                        pca_component_count=pca_component_count,
                                                                        pca_skipped_components_count=pca_skipped_components_count,
                                                                        pca_apply_standard_scaler=True)
    return get_roc_curve_as_base64(label_list, distance_list)


def get_auc_value_plain_mahalanobis_distance(matrix_a_linear, anomalous_directions, pca_component_count,
                                             pca_skipped_components_count, pca_apply_standard_scaler=False):
    label_list, distance_list = get_data_for_plain_mahalanobis_distance(matrix_a_linear=matrix_a_linear,
                                                                        anomalous_directions=anomalous_directions,
                                                                        pca_component_count=pca_component_count,
                                                                        pca_skipped_components_count=pca_skipped_components_count,
                                                                        pca_apply_standard_scaler=pca_apply_standard_scaler)
    fpr, tpr, thresholds = metrics.roc_curve(label_list, distance_list)
    return metrics.auc(fpr, tpr)


def get_tsne_for_original_data():
    plt.clf()
    data_points, data_label = load_latent_space_data_points(
        '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')
    tsne = TSNE(n_components=2, random_state=0)
    tsne_res = tsne.fit_transform(np.array(data_points))
    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=data_label, palette=sns.hls_palette(2), legend='full')
    return plot_to_base64(plt)


def get_tsne_with_dimension_weighted_metric(weighted_dims, ignore_unlabeled_dims, pca_component_count=0,
                                            skipped_components_count=0, weight_factor=10, ignore_labels=False):
    plt.clf()

    if pca_component_count > 0:
        weights = np.ones(pca_component_count) if not ignore_unlabeled_dims else np.zeros(pca_component_count)
        pca = PCA(n_components=pca_component_count + skipped_components_count)
    else:
        weights = np.ones(100) if not ignore_unlabeled_dims else np.zeros(100)

    for dim in weighted_dims:
        weights[dim] = weight_factor if not ignore_unlabeled_dims else 1

    global global_weights
    global_weights = weights

    data_points, data_label = load_latent_space_data_points(
        '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')

    data = np.array(data_points)
    if pca_component_count > 0:
        assert pca_component_count + skipped_components_count < data.shape[1], \
            "pca_component_count+skipped_components_count must be smaller then total number of columns"

        data = StandardScaler().fit_transform(data)
        principal_components = pca.fit_transform(data)
        data = principal_components[:, skipped_components_count:]

    tsne = TSNE(n_components=2, random_state=0,
                metric=element_weighted_euclidean_distance if not ignore_labels else "euclidean")
    tsne_res = tsne.fit_transform(data)
    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=data_label, palette=sns.hls_palette(2), legend='full')

    return plot_to_base64(plt)


def element_weighted_euclidean_distance(u, v):
    return np.linalg.norm((np.multiply(u - v, global_weights)))


def mahalanobis_distance(u, mean, vi):
    return scipy.spatial.distance.mahalanobis(u, mean, vi)
