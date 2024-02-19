import os

import torchvision
from torchvision.transforms import transforms

from src.ml.dataset_generation.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.dataset_generation.generate_dataset import add_line_to_csv


class DatasetGeneratorMnistFashionMnist(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            dataset_name='DS14_fashionmnist_shirt_sneaker',
            num_channels=1,
            num_epochs=100,
            n_latent_space_search_iterations=2000,
            draw_images=False,
            directions_count=30,
            direction_train_steps=1500
        )

    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        fashion_mnist_dataset = torchvision.datasets.FashionMNIST(root=temp_directory,
                                                                  download=True,
                                                                  transform=self.transform)

        norm_class = 0
        norms = []
        for d in fashion_mnist_dataset:
            if d[1] == norm_class:
                norms.append(d)

        for i, img in enumerate(norms):
            img = (img[0] * 0.5) + 0.5
            img = transforms.ToPILImage()(img)
            file_name = f"img_{norm_class}_{i}.png"
            img.save(os.path.join(dataset_folder, file_name))
            add_line_to_csv(csv_path, [file_name, "False"])

    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        fashion_mnist_dataset = torchvision.datasets.FashionMNIST(root=temp_directory,
                                                                  download=True,
                                                                  transform=self.transform)

        ano_class = 7
        anos = []
        for d in fashion_mnist_dataset:
            if d[1] == ano_class:
                anos.append(d)

        anos = anos[:round(len(anos) * ano_fraction)]
        for i, img in enumerate(anos):
            img = (img[0] * 0.5) + 0.5
            img = transforms.ToPILImage()(img)
            file_name = f"img_{ano_class}_{i}.png"
            img.save(os.path.join(dataset_folder, file_name))
            add_line_to_csv(csv_path, [file_name, "True"])


if __name__ == '__main__':
    ds_generator = DatasetGeneratorMnistFashionMnist()

    # ds_generator.run_generate_dataset()
    # ds_generator.run_equalize_image_sizes()
    # ds_generator.run_train_and_save_gan()
    # ds_generator.run_train_direction_matrix()
    # ds_generator.run_train_beta_vae()
    ds_generator.run_create_latent_space_dataset()
