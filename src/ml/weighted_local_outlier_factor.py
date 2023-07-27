import os

import numpy
import scipy.spatial.distance
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance


class WeightedLocalOutlierFactor:
    def __init__(self,
                 direction_matrix, anomalous_directions, n_neighbours, pca_component_count=0,
                 skipped_components_count=0, ignore_labels=False):
        self.data = []
        self.direction_matrix = direction_matrix.linear.weight.data.numpy()
        self.pca = PCA(n_components=pca_component_count + skipped_components_count)

        self.labeled_directions = []
        for idx, val in enumerate(anomalous_directions):
            self.labeled_directions.append(self.direction_matrix.T[val])
        self.labeled_directions = np.array(self.labeled_directions)

        # mean_array = np.matrix(labeled_directions).mean(0).A1
        #
        # for idx, col in enumerate(self.direction_matrix):
        #     if idx not in labeled_dims:
        #         self.direction_matrix[idx] = mean_array

        # for idx, col in enumerate(self.direction_matrix):
        #     if idx not in labeled_dims:
        #         self.direction_matrix[idx] = col * 0.01
        #
        # cov = np.cov(self.direction_matrix)
        # self.vi = np.linalg.inv(cov)

        cov = np.cov(self.labeled_directions.T)
        self.vi = np.linalg.inv(cov)

        # for idx, col in enumerate(self.direction_matrix):
        #     if idx not in labeled_dims:
        #         self.direction_matrix[idx] = col*0.01

        self.lof = LocalOutlierFactor(
            n_neighbors=n_neighbours,
            metric=self.__get_mahalanobis_distance if not ignore_labels else "minkowski",
        )

        self.pca_component_count = pca_component_count
        self.skipped_components_count = skipped_components_count

    def fit(self):
        data_as_array = np.array(self.data)
        self.lof.fit_predict(data_as_array)

    def load_latent_space_datapoints(self, data=[], root_dir=''):
        if len(data) > 0:
            self.data = numpy.array(data)
        else:
            directory = os.fsencode(root_dir)
            for file in os.listdir(directory):
                filename = os.fsdecode(file)
                if filename.endswith(".pt"):
                    path = os.path.join(root_dir, filename)
                    self.data.append(torch.load(path, map_location=torch.device('cpu')).detach().numpy().reshape(100))
            self.data = numpy.array(self.data)

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

    # def __get_mahalanobis_distance(self, u, v):
    #     return distance.mahalanobis(u, v, self.direction_matrix.T)

    def __get_mahalanobis_distance(self, u, v):
        diff = (u - v)
        return np.sqrt(diff.T @ self.vi @ diff)
        # return scipy.spatial.distance.mahalanobis(u, v, self.iv)
