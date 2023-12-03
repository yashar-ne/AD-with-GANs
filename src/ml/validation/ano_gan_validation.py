import csv
import os

from src.ml.validation.validation_utils import get_roc_curve_as_base64


def get_roc_auc_for_ano_gan_validation(dataset_name):
    X = []
    y = []
    csv_file_path = os.path.join('../data', dataset_name, 'dataset', 'latent_space_mappings.csv')
    with open(csv_file_path, 'r') as csvfile:
        datareader = csv.reader(csvfile)
        next(datareader)
        for row in datareader:
            if len(row) < 4:
                return None, None

            X.append(float(row[3]))
            y.append(1 if row[1] == 'True' else 0)

    return get_roc_curve_as_base64(y, X)
