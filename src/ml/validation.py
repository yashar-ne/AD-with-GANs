import base64
import io
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import metrics
import csv

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import transforms

from src.ml.models.generator import Generator
from src.ml.tools.ano_mnist_dataset_generator import get_ano_mnist_dataset
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


def get_roc_curve_as_base64(label, lofs_in):
    plt.clf()
    fpr, tpr, thresholds = metrics.roc_curve(label, lofs_in)
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


def get_roc_auc_for_given_dims(weighted_dims, latent_space_data_points, latent_space_data_labels, pca_component_count,
                               skipped_components_count, n_neighbours, weight_factor=10, one_hot_weighing=True, ignore_labels=False):

    weighted_lof = WeightedLocalOutlierFactor(weighted_dims=weighted_dims,
                                              weight_factor=weight_factor,
                                              n_neighbours=n_neighbours,
                                              pca_component_count=pca_component_count,
                                              skipped_components_count=skipped_components_count,
                                              one_hot_weighing=one_hot_weighing,
                                              ignore_labels=ignore_labels)

    weighted_lof.load_latent_space_datapoints(data=latent_space_data_points)
    weighted_lof.fit()

    y = np.array([1 if d == "False" else -1 for d in latent_space_data_labels])
    return get_roc_curve_as_base64(y, weighted_lof.get_negative_outlier_factor())


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

    tsne = TSNE(n_components=2, random_state=0, metric=element_weighted_euclidean_distance if not ignore_labels else "euclidean")
    tsne_res = tsne.fit_transform(data)
    sns.scatterplot(x=tsne_res[:, 0], y=tsne_res[:, 1], hue=data_label, palette=sns.hls_palette(2), legend='full')

    return plot_to_base64(plt)


def element_weighted_euclidean_distance(u, v):
    return np.linalg.norm((np.multiply(u - v, global_weights)))


# Just to test the implementations for LOF and ROC-AUC
def test_roc_auc_and_lof():
    X = [[-1.1, 2.1, 4.2, -862.4], [0.2, 2.1, 4.2, -8.4], [101.1, 88.1, 4.2, -8.4], [-81.2, 105.1, 4.2, -8.4],
         [0.3, 2.1, 4.2, -8.4], [1.5, 2.1, 4.2, -8.4], [2.1, 2.1, 4.2, -8.4], [-1.6, 2.1, 4.2, -8.4],
         [-3.6, 2.1, 4.2, -8.4]]
    y = [-1, 1, -1, -1, 1, 1, 1, 1, 1]
    global global_weights
    global_weights = [1, 1, 0, 1]
    clf = LocalOutlierFactor(n_neighbors=1,
                             metric=element_weighted_euclidean_distance)
    y_pred = clf.fit_predict(X)
    print(y)
    print(y_pred)
    print(clf.negative_outlier_factor_)
    print(y == y_pred)

    fpr, tpr, thresholds = metrics.roc_curve(y, clf.negative_outlier_factor_)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                      estimator_name='LOF')
    display.plot()
    plt.show()


def test_latent_space_points():
    data_points, data_labels = load_latent_space_data_points(
        '/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST')
    generator: Generator = Generator(size_z=100, num_feature_maps=64, num_color_channels=1)
    generator.load_state_dict(torch.load('/home/yashar/git/python/AD-with-GANs/saved_models/generator.pkl',
                                         map_location=torch.device(device)))
    to_pil_image = transforms.ToPILImage()
    generator.eval()
    for i in range(len(data_points)):
        if data_labels[i] == 'True':
            z = torch.from_numpy(data_points[i].reshape(1, 100, 1, 1))
            original_img = generator(z).cpu()
            plt.imshow(to_pil_image(original_img[0]))
            plt.show()
