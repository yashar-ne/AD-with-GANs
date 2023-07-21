import os

import numpy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class WeightedLocalOutlierFactor:
    def __init__(self, weighted_dims, n_neighbours, weight_factor=10, pca_component_count=0,
                 skipped_components_count=0, ignore_unlabeled_dims=False, ignore_labels=False):
        self.data = []
        if pca_component_count > 0:
            self.weights = np.ones(pca_component_count) if not ignore_unlabeled_dims else np.zeros(pca_component_count)
            self.pca = PCA(n_components=pca_component_count + skipped_components_count)
        else:
            self.weights = np.ones(100) if not ignore_unlabeled_dims else np.zeros(100)

        for dim in weighted_dims:
            self.weights[dim] = weight_factor if not ignore_unlabeled_dims else 1

        self.lof = LocalOutlierFactor(n_neighbors=n_neighbours,
                                      metric=self.__element_weighted_euclidean_distance if not ignore_labels else "minkowski")
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

        if self.pca_component_count > 0:
            assert self.pca_component_count + self.skipped_components_count < self.data.shape[1], \
                "pca_component_count+skipped_components_count must be smaller then total number of columns"

            self.pca_component_count = self.pca_component_count + self.skipped_components_count
            data = self.data
            data = StandardScaler().fit_transform(data)
            principal_components = self.pca.fit_transform(data)
            self.data = principal_components[:, self.skipped_components_count:]

    def get_negative_outlier_factor(self):
        return self.lof.negative_outlier_factor_

    def __element_weighted_euclidean_distance(self, u, v):
        return np.linalg.norm((np.multiply(u - v, self.weights)))
