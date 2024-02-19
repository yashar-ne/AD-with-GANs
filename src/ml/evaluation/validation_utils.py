import base64
import csv
import io
import math
import os

import seaborn
import torch
from matplotlib import pyplot as plt
from sklearn import metrics

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_to_base64(plot):
    io_bytes = io.BytesIO()
    plot.savefig(io_bytes, format='jpg')
    io_bytes.seek(0)
    return base64.b64encode(io_bytes.read()).decode()


def get_2d_plot(local_outlier_factor):
    seaborn.set_style("darkgrid")
    plt.figure(figsize=(10, 9))
    axes = plt.axes()

    data_points, data_label = load_data_points(
        '../data/LatentSpaceMNIST')

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

    data_points, data_label = load_data_points(
        '../data/LatentSpaceMNIST')

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
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc)
    display.plot()
    return plot_to_base64(plt), auc


def load_data_points(base_url):
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
            latent_space_point = latent_space_point_pt.cpu().numpy()
            data_points.append(latent_space_point)
            data_labels.append(True if row[1] == 'True' else False)

    return data_points, data_labels
