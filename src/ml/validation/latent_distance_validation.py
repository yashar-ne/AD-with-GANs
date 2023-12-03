import numpy as np
from sklearn.preprocessing import normalize

from src.ml.tools.utils import extract_weights_from_model
from src.ml.validation.validation_utils import get_roc_curve_as_base64


# Reasoning
# x = Inlier
# Klasse für d wird gesucht
# Richtungsvektoren alpha_0, ..., alpha_n

# min_(lambda) | d + lambda * alpha_0 - x |

# lambda ist groß = outlier
# lambda ist klein = inlier

# min_(lambda) (d + lambda * alpha_0 - x)^2
# d / d_lambda
#    --> 2 * (d + lambda * alpha_0 - x) * alpha_0 = 0

# lambda * alpha_0 * alpha_0 = (x - d) * alpha_0

# lambda = (x - d) * alpha_0^(-1)
# lambda = (x - d) * alpha_0

# x = sum d_i * 1/N
# N = 50

# max(lambda_0, lambda_1, lambda_2)
def get_roc_auc_for_average_distance_metric(latent_space_data_points,
                                            latent_space_data_labels,
                                            direction_matrix,
                                            anomalous_directions):
    inlier = []
    latent_space_data_points = normalize(latent_space_data_points, axis=1, norm='l2')

    for idx, p in enumerate(latent_space_data_points):
        if latent_space_data_labels[idx] is False:
            inlier.append(p / np.linalg.norm(p) if np.linalg.norm(p) != 0 else p)
    average_inlier_vector = np.mean(inlier, axis=0)

    direction_matrix = extract_weights_from_model(direction_matrix)
    directions = [direction_matrix[d[0]] * d[1] for d in anomalous_directions if
                  (d[0], d[1] * -1) not in anomalous_directions]

    if len(directions) == 0:
        return None, None

    scores = []
    for data_point in latent_space_data_points:
        direction_scores = []
        for d in directions:
            # direction_scores.append((average_inlier_vector - data_point) @ d)
            cos_angle = data_point @ d
            direction_scores.append(cos_angle)
            # direction_scores.append(data_point @ d)
        scores.append(max(direction_scores))

    y = np.array([0 if d is False else 1 for d in latent_space_data_labels])
    return get_roc_curve_as_base64(y, scores)
