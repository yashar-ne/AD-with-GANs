import os

import numpy
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from src.ml.latent_space_mapper import LatentSpaceMapper


class WeightedLocalOutlierFactor:
    def __init__(self, weighted_dims, weight_factor=10, n_neighbours=50, pca_component_count=0, skipped_components_count=0):
        self.data = []
        if pca_component_count > 0:
            self.weights = np.ones(pca_component_count)
            self.pca = PCA(n_components=pca_component_count+skipped_components_count)
        else:
            self.weights = np.ones(100)

        for dim in weighted_dims:
            self.weights[dim] = weight_factor
        self.lof = LocalOutlierFactor(n_neighbors=n_neighbours,
                                      metric=self.__element_weighted_euclidean_distance,
                                      novelty=True)
        self.pca_component_count = pca_component_count
        self.skipped_components_count = skipped_components_count

    def predict(self, x):
        if self.pca_component_count > 0:
            x = self.__pca_transform(x)

        return self.lof.predict(x)

    def fit(self):
        data_as_array = np.array(self.data)
        self.lof.fit(data_as_array)

    def load_datapoints(self, root_dir):
        directory = os.fsencode(root_dir)
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            if filename.endswith(".pt"):
                path = os.path.join(root_dir, filename)
                self.data.append(torch.load(path, map_location=torch.device('cpu')).detach().numpy().reshape(100))

        self.data = numpy.array(self.data)

        if self.pca_component_count > 0:
            assert self.pca_component_count+self.skipped_components_count < self.data.shape[1],\
                "pca_component_count+skipped_components_count must be smaller then total number of columns"

            self.pca_component_count = self.pca_component_count+self.skipped_components_count
            data = self.data
            data = StandardScaler().fit_transform(data)
            principal_components = self.pca.fit_transform(data)
            self.data = principal_components[:, self.skipped_components_count:]

    def __pca_transform(self, data_point):
        return self.pca.transform(data_point)[:, self.skipped_components_count:]

    def __element_weighted_euclidean_distance(self, u, v):
        return np.linalg.norm(u - (np.multiply(v, self.weights)))


# USAGE Example:
# lof = WeightedLocalOutlierFactor([4, 8, 11], pca_component_count=20, skipped_components_count=4)
# lof.load_datapoints("../../data/latent_space_mappings")
# lof.fit()
#
# dat = torch.load("/home/yashar/git/python/AD-with-GANs/data/latent_space_mappings/mapped_z_5506.pt", map_location=torch.device('cpu')).detach().numpy().reshape(1, 100)
# print(lof.predict(dat))





