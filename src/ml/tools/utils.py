import torch
import numpy as np
import base64
import io
from PIL import Image

from torchvision.transforms import ToPILImage
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from src.backend.models.ImageStripModel import ImageStripModel
from src.ml.models.matrix_a_linear import MatrixALinear


def generate_noise(batch_size, z_dim, device):
    return torch.randn(batch_size, z_dim, 1, 1, device=device)


def to_image(tensor):
    if len(tensor.shape) == 4:
        tensor = tensor[0]

    tensor = (tensor + 1) / 2
    tensor.clamp(0, 1)
    return ToPILImage()((255 * tensor.cpu().detach()).to(torch.uint8))


def one_hot(dims, value, index):
    vec = torch.zeros(dims)
    vec[index] = value
    return vec


def apply_pca_to_matrix_a(matrix_a_linear, component_count, skipped_components_count):
    matrix_a_np = matrix_a_linear.linear.weight.data.numpy()
    matrix_a_np = normalize(matrix_a_np, axis=1, norm='l2')
    pca = PCA(n_components=component_count + skipped_components_count)
    principal_components = pca.fit_transform(matrix_a_np)
    principal_components = principal_components[:, skipped_components_count:]

    new_weights = torch.from_numpy(principal_components)
    matrix_a_linear_after_pca = MatrixALinear(input_dim=component_count, output_dim=100, bias=False)
    new_dict = {
        'linear.weight': new_weights
    }
    matrix_a_linear_after_pca.load_state_dict(new_dict)

    return matrix_a_linear_after_pca


def extract_weights_from_model(matrix_a_linear):
    matrix_a_np = matrix_a_linear.linear.weight.data.cpu().numpy()
    matrix_a_np = normalize(matrix_a_np, axis=1, norm='l2')

    return matrix_a_np.T


def extract_weights_from_model_and_apply_pca(matrix_a_linear, pca_component_count, pca_skipped_components_count):
    matrix_a_np = matrix_a_linear.linear.weight.data.cpu().numpy()
    matrix_a_np = normalize(matrix_a_np, axis=1, norm='l2')

    if pca_component_count == 0:
        return matrix_a_np.T

    pca = PCA(n_components=pca_component_count + pca_skipped_components_count)
    principal_components = pca.fit_transform(matrix_a_np)
    return principal_components[:, pca_skipped_components_count:].T


def generate_base64_image_from_tensor(images_tensor):
    transform = ToPILImage()
    images_tensor = (images_tensor * 0.5) + 0.5
    img = transform(images_tensor)

    with io.BytesIO() as buf:
        img.save(buf, format='PNG')
        img_str = base64.b64encode(buf.getvalue())

    return img_str


def generate_base64_images_from_tensor_list(images_tensor_list):
    image_list = []
    for idx, i in enumerate(images_tensor_list):
        image_list.append(
            ImageStripModel(
                position=idx,
                image=generate_base64_image_from_tensor(i),
                direction_position=1,
                total_directions=1
            )
        )

    return image_list
