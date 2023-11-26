import os

from torchvision.datasets import MNIST

from src.ml.datasets.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.datasets.generate_dataset import add_line_to_csv


class DatasetGeneratorMnist_9_6(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            dataset_name='DS12_mnist_9_6',
            num_color_channels=1,
            num_epochs=50,
            n_latent_space_search_iterations=1000,
            draw_images=False,
            num_imgs=0,
        )

    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        mnist_dataset = MNIST(
            root=temp_directory,
            train=True,
            download=True,
        )

        norm_class = 9
        norms = [d for d in mnist_dataset if (d[1] == norm_class)]
        for i, img in enumerate(norms):
            file_name = f"img_{norm_class}_{i}.png"
            img[0].save(os.path.join(dataset_folder, file_name))
            add_line_to_csv(csv_path, [file_name, "False"])

    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        mnist_dataset = MNIST(
            root=temp_directory,
            train=True,
            download=True,
        )

        ano_class = 6
        anos = [d for d in mnist_dataset if (d[1] == ano_class)]
        anos = anos[:round(len(anos) * ano_fraction)]
        for i, img in enumerate(anos):
            file_name = f"img_{ano_class}_{i}.png"
            img[0].save(os.path.join(dataset_folder, file_name))
            add_line_to_csv(csv_path, [file_name, "True"])


DatasetGeneratorMnist_9_6().run(ano_fraction=0.1)
