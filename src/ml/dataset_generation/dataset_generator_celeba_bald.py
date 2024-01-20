import csv
import itertools
import os

from PIL import Image

from src.ml.dataset_generation.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.dataset_generation.generate_dataset import add_line_to_csv


class DatasetGeneratorCelebA_Bald(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            dataset_name='DS13_celeba_bald',
            num_color_channels=3,
            num_epochs=20,
            n_latent_space_search_iterations=2000,
            draw_images=False
        )

    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        class_id = "-1"
        files = []
        dataset_directory = os.path.join(temp_directory, "celebA")
        dataset_directory_images = os.path.join(dataset_directory, "imgs")
        dataset_directory_csv = os.path.join(dataset_directory, "list_attr_celeba.csv")

        with open(dataset_directory_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in itertools.islice(reader, 10000):
                if row[5] == class_id:
                    files.append((row[0]))

        for i, filename in enumerate(files):
            new_filename = f"img_norm_{i}.png"
            im = Image.open(os.path.join(dataset_directory_images, filename))
            im.save(os.path.join(dataset_folder, new_filename))
            add_line_to_csv(csv_path, [new_filename, "False"])

    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        class_id = "1"
        files = []
        dataset_directory = os.path.join(temp_directory, "celebA")
        dataset_directory_images = os.path.join(dataset_directory, "imgs")
        dataset_directory_csv = os.path.join(dataset_directory, "list_attr_celeba.csv")

        with open(dataset_directory_csv, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)
            for row in itertools.islice(reader, 100000):
                if row[5] == class_id:
                    files.append((row[0]))

        for i, filename in enumerate(files):
            new_filename = f"img_ano_{i}.png"
            im = Image.open(os.path.join(dataset_directory_images, filename))
            im.save(os.path.join(dataset_folder, new_filename))
            add_line_to_csv(csv_path, [new_filename, "True"])


if __name__ == '__main__':
    ds_generator = DatasetGeneratorCelebA_Bald()
    # ds_generator.run_generate_dataset()
    # ds_generator.run_equalize_image_sizes()
    # ds_generator.run_train_and_save_gan(display_generator_test=True)
    # ds_generator.run_train_direction_matrix()
    ds_generator.run_train_beta_vae()
    # ds_generator.run_create_latent_space_dataset()
