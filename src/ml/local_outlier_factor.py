import sklearn
import numpy as np


class LocalOutlierFactor:
    def __init__(self, weighted_dims, weight_factor=10, n_neighbours=20):
        self.weights = np.ones(100)
        for dim in weighted_dims:
            self.weights[dim] = weight_factor
        self.lof = sklearn.neighbors.LocalOutlierFactor(n_neighbors=n_neighbours,
                                                        metric=self.element_weighted_euclidean_distance)

    def fit(self, X):
        return self.lof.fit_predict(X)

    def element_weighted_euclidean_distance(self, u, v):
        return np.linalg.norm(u - (np.multiply(v, self.weights)))
