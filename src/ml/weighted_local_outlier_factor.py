import os
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import torch


class WeightedLocalOutlierFactor:
    def __init__(self, direction_matrix, anomalous_directions, n_neighbours, use_default_distance_metric=False):
        self.data = []
        self.direction_matrix = direction_matrix
        self.labeled_directions_matrix = []

        for d in anomalous_directions:
            if (d[0], d[1]*-1) not in anomalous_directions:
                self.labeled_directions_matrix.append(direction_matrix[d[0]]*d[1])
        self.labeled_directions_matrix = np.array(self.labeled_directions_matrix)

        outlier_weight = 1
        normal_weight = 0
        self.label_vector = np.ones(direction_matrix.shape[0])
        for idx, d in enumerate(self.label_vector):
            if idx in anomalous_directions:
                self.label_vector[idx] = d * outlier_weight
            else:
                self.label_vector[idx] = d * normal_weight

        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbours,
            metric=self.__get_distance if not use_default_distance_metric else "minkowski",
        )

    def fit(self):
        data_as_array = np.array(self.data)
        self.lof.fit_predict(data_as_array)

    def load_latent_space_datapoints(self, data=[], root_dir=''):
        if len(data) > 0:
            self.data = np.array(data)
        else:
            directory = os.fsencode(root_dir)
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".pt"):
                    path = os.path.join(root_dir, filename)
                    self.data.append(torch.load(path, map_location=torch.device('cpu')).detach().numpy().reshape(100))
            self.data = np.array(self.data)

    def get_negative_outlier_factor(self):
        return self.lof.negative_outlier_factor_

    def get_labeled_directions_matrix(self):
        return self.labeled_directions_matrix

    def __get_distance(self, u, v):
        u_star = u.T @ self.labeled_directions_matrix.T
        v_star = v.T @ self.labeled_directions_matrix.T
        u_norm = u/np.linalg.norm(u)
        v_norm = v/np.linalg.norm(v)

        left = u_star @ v_star
        right = u_norm @ v_norm

        d = left / right
        return d
