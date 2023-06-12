import torch
from torchvision.transforms import ToPILImage
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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


def apply_pca(matrix_a_linear, component_count, skipped_components_count, apply_standard_scaler):
    matrix_a_np = matrix_a_linear.linear.weight.data.numpy()

    if apply_standard_scaler:
        matrix_a_np = StandardScaler().fit_transform(matrix_a_np)
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
