import os

import numpy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler


class WeightedLocalOutlierFactor:
    def __init__(self, weighted_dims, contamination="auto", weight_factor=1, n_neighbours=20, pca_component_count=0,
                 skipped_components_count=0):
        self.data = []
        if pca_component_count > 0:
            self.weights = np.zeros(pca_component_count)
            self.pca = PCA(n_components=pca_component_count + skipped_components_count)
        else:
            self.weights = np.zeros(100)

        for dim in weighted_dims:
            self.weights[dim] = weight_factor

        self.lof = LocalOutlierFactor(n_neighbors=n_neighbours,
                                      metric=self.__element_weighted_euclidean_distance,
                                      novelty=True,
                                      contamination=contamination)
        self.pca_component_count = pca_component_count
        self.skipped_components_count = skipped_components_count

    def predict(self, x):
        if self.pca_component_count > 0:
            x = self.__pca_transform(x)

        return self.lof.predict(x)

    def fit(self):
        data_as_array = np.array(self.data)
        self.lof.fit(data_as_array)

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

    def __pca_transform(self, data_point):
        return self.pca.transform(data_point)[:, self.skipped_components_count:]

    def __element_weighted_euclidean_distance(self, u, v):
        return np.linalg.norm((np.multiply(u - v, self.weights)))

# USAGE Example:
# lof = WeightedLocalOutlierFactor([4, 8, 11], pca_component_count=20, skipped_components_count=4)
# lof.load_datapoints("../../data/LatentSpaceMNIST")
# lof.fit()
#
# dat = torch.load("/home/yashar/git/python/AD-with-GANs/data/LatentSpaceMNIST/mapped_z_5506.pt", map_location=torch.device('cpu')).detach().numpy().reshape(1, 100)
# print(lof.predict(dat))
