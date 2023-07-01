import base64
import io
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn import metrics
import csv

from sklearn.neighbors import LocalOutlierFactor
from torchvision.transforms import transforms

from src.ml.tools.ano_mnist_dataset_generator import get_ano_mnist_dataset
from src.ml.weighted_local_outlier_factor import WeightedLocalOutlierFactor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def create_roc_curve(label, lofs_in):
    fpr, tpr, thresholds = metrics.roc_curve(label, lofs_in)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name='LOF')
    display.plot()
    plt.show()


def get_roc_curve_as_base64(label, lofs_in):
    fpr, tpr, thresholds = metrics.roc_curve(label, lofs_in)
    auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc,
                                      estimator_name='LOF')
    display.plot()
    io_bytes = io.BytesIO()
    plt.savefig(io_bytes, format='jpg')
    io_bytes.seek(0)
    return base64.b64encode(io_bytes.read()).decode(), auc


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


def get_roc_auc_for_given_dims(weighted_dims, latent_space_data_points, latent_space_data_labels, pca_component_count,
                               skipped_components_count, n_neighbours=20):
    weighted_lof = WeightedLocalOutlierFactor(weighted_dims=weighted_dims,
                                              n_neighbours=n_neighbours,
                                              pca_component_count=pca_component_count,
                                              skipped_components_count=skipped_components_count)

    weighted_lof.load_latent_space_datapoints(data=latent_space_data_points)
    weighted_lof.fit()

    y = np.array([-1 if d == "False" else 1 for d in latent_space_data_labels])
    return get_roc_curve_as_base64(y, weighted_lof.get_negative_outlier_factor())

# np.set_printoptions(threshold=np.inf)
#
# latent_space_data_points, latent_space_data_labels = load_latent_space_data_points('../../data/LatentSpaceMNIST')
# weighted_lof = WeightedLocalOutlierFactor(weighted_dims=[9, 17], pca_component_count=20, skipped_components_count=3)
# weighted_lof.load_latent_space_datapoints(data=latent_space_data_points)
# weighted_lof.fit()
#
# y = np.array([-1 if d == "False" else 1 for d in latent_space_data_labels])
# create_roc_curve(y, weighted_lof.get_negative_outlier_factor())

# lof = LocalOutlierFactor(n_neighbors=3, contamination="auto")
# lof_ano_mnist = LocalOutlierFactor(n_neighbors=3, contamination="auto")


#
# y_pred = lof.fit_predict(X)
# lofs = lof.negative_outlier_factor_
# n_err = (y_pred != y).sum()
#
# outlier_count = len(y)
# print(f"Outlier Count (real): {len(y[y == -1])}")
# print(f"Outlier Count (pred): {len(y_pred[y_pred == -1])}")
# print(f"Error Count: {n_err}")
#
# print("---------------")
#
# X_ano_mnist, y_ano_mnist = get_ano_mnist_data('../../data')
# y_ano_mnist = np.array([1 if d == False else -1 for d in y_ano_mnist])
# y_ano_mnist_pred = lof_ano_mnist.fit_predict(X_ano_mnist)
# lofs_ano_mnist = lof_ano_mnist.negative_outlier_factor_
# n_err = (y_ano_mnist_pred != y_ano_mnist).sum()
#
# print(f"Outlier Count (real): {len(y_ano_mnist[y_ano_mnist == -1])}")
# print(f"Outlier Count (pred): {len(y_ano_mnist_pred[y_ano_mnist_pred == -1])}")
# print(f"Error Count: {n_err}")
#
#
# create_roc_curve(y, lofs)
# create_roc_curve(y_ano_mnist, lofs_ano_mnist)
