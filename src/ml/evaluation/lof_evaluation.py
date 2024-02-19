import csv
import os

import numpy as np
from PIL import Image
from sklearn.neighbors import LocalOutlierFactor

from src.ml.evaluation.validation_utils import get_roc_curve_as_base64


def get_roc_auc_lof(dataset_name, n_neighbours):
    X = []
    y = []
    image_folder = os.path.join('../data', dataset_name, 'dataset_raw')
    csv_file_path = os.path.join('../data', dataset_name, 'dataset_raw', 'ano_dataset.csv')
    with open(csv_file_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            image_path = os.path.join(image_folder, row[0])
            img = Image.open(image_path)

            X.append(np.asarray(img).flatten())
            y.append(1 if row[1] == 'True' else -1)

    lof = LocalOutlierFactor(n_neighbors=n_neighbours)
    lof.fit_predict(X)

    result = get_roc_curve_as_base64(y, lof.negative_outlier_factor_)

    return result
