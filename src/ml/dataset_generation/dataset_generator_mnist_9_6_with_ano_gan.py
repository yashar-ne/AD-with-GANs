import os

from torchvision.datasets import MNIST

from src.ml.dataset_generation.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.dataset_generation.generate_dataset import add_line_to_csv


class DatasetGeneratorMnist_9_6(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            dataset_name='mnist_9_6_with_ano_gan',
            num_channels=1,
            num_epochs=100,
            n_latent_space_search_iterations=1500,
            draw_images=False,
            num_imgs=0,
            directions_count=30,
            direction_train_steps=1500
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


if __name__ == '__main__':
    ds_generator = DatasetGeneratorMnist_9_6()

    # ds_generator.run_generate_dataset()
    # ds_generator.run_equalize_image_sizes()
    # ds_generator.run_train_and_save_gan()
    # ds_generator.run_train_direction_matrix()
    # ds_generator.run_train_beta_vae()
    # ds_generator.run_create_latent_space_dataset()
