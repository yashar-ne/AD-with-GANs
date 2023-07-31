import base64
import io
import math
import os
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


# TODO: implement pca_apply_standard_scaler in API/controller/frontend
def get_roc_auc_for_given_dims(direction_matrix, anomalous_directions,
                               latent_space_data_points,
                               latent_space_data_labels,
                               pca_component_count,
                               pca_skipped_components_count, n_neighbours,
                               pca_apply_standard_scaler=True,
                               use_default_distance_metric=False):
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


def get_tsne_for_original_data():
    plt.clf()
    data_points, data_label = load_latent_space_data_points(
        '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')
    tsne = TSNE(n_components=2, random_state=0)
    tsne_res = tsne.fit_transform(np.array(data_points))
    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=data_label, palette=sns.hls_palette(2), legend='full')
    return plot_to_base64(plt)


def element_weighted_euclidean_distance(u, v):
    return np.linalg.norm((np.multiply(u - v, global_weights)))


def mahalanobis_distance(u, mean, vi):
    return scipy.spatial.distance.mahalanobis(u, mean, vi)
