import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import numpy
import scipy.spatial.distance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


class WeightedLocalOutlierFactor:
    def __init__(self, direction_matrix, anomalous_directions, n_neighbours, pca_component_count=0,
                 pca_skipped_components_count=0, use_default_distance_metric=False):
        self.data = []
        self.direction_matrix = direction_matrix
        self.pca = PCA(n_components=pca_component_count + pca_skipped_components_count)
        # self.labeled_directions = []
        #
        # for idx, direction in enumerate(direction_matrix):
        #     if idx in anomalous_directions:
        #         self.labeled_directions.append(direction)
        #     else:
        #         self.labeled_directions.append(direction*0.1)

        # for idx, direction in enumerate(direction_matrix):
        #     if idx in anomalous_directions:
        #         self.labeled_directions.append(direction)

        # self.labeled_directions = np.array(self.labeled_directions)

        outlier_weight = 1
        normal_weight = 0
        self.label_vector = np.ones(pca_component_count if pca_component_count > 0 else 100)
        for idx, d in enumerate(self.label_vector):
            if idx in anomalous_directions:
                self.label_vector[idx] = d * outlier_weight
            else:
                self.label_vector[idx] = d * normal_weight

        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbours,
            metric=self.__get_distance if not use_default_distance_metric else "minkowski",
        )

        self.pca_component_count = pca_component_count
        self.skipped_components_count = pca_skipped_components_count

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

        # if self.pca_component_count > 0:
        #     assert self.pca_component_count + self.skipped_components_count < self.data.shape[1], \
        #         "pca_component_count+skipped_components_count must be smaller then total number of columns"
        #
        #     self.pca_component_count = self.pca_component_count + self.skipped_components_count
        #     data = self.data
        #     data = StandardScaler().fit_transform(data)
        #     principal_components = self.pca.fit_transform(data)
        #     self.data = principal_components[:, self.skipped_components_count:]

    def get_negative_outlier_factor(self):
        return self.lof.negative_outlier_factor_

    def __get_distance(self, u, v):
        diff = u/np.linalg.norm(u) - v/np.linalg.norm(v)
        # diff = (u-v)
        left = np.dot(diff.T, self.direction_matrix.T)
        d = np.dot(left, self.label_vector)
        return d
        # return (u - v).T @ self.direction_matrix.T @ self.label_vector
