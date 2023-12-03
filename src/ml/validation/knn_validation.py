import csv
import os

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from src.ml.validation.validation_utils import get_roc_curve_as_base64


def get_knn_validation(dataset_name, k):
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
            y.append(1 if row[1] == 'True' else 0)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.95)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    return get_roc_curve_as_base64(y_test, knn.predict(X_test))
