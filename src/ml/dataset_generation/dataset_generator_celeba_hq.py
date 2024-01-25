import csv
import itertools
import os

from PIL import Image

from src.ml.dataset_generation.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.dataset_generation.generate_dataset import add_line_to_csv


class DatasetGeneratorCelebaHQ(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            dataset_name='DS17_celeba_hq_glasses',
            num_color_channels=3,
            num_epochs=1500,
            n_latent_space_search_iterations=2500,
            draw_images=True,
            num_imgs=0,
            directions_count=20,
            direction_train_steps=1000,
            stylegan=True,
            only_consider_anos=True,
        )

    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        files = []
        class_id = "-1"
        raw_dataset_folder = os.path.join(temp_directory, "CelebAMask-HQ")
        images_folder = os.path.join(raw_dataset_folder, "CelebA-HQ-img")
        dataset_directory_csv = os.path.join(raw_dataset_folder, "CelebAMask-HQ-attribute-anno.txt")

        with open(dataset_directory_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            next(reader)
            next(reader)
            for row in itertools.islice(reader, 200):
                if row[6] == class_id:
                    files.append((row[0]))

        for i, filename in enumerate(files):
            new_filename = f"img_norm_{i}.png"
            img = Image.open(os.path.join(images_folder, filename))
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            img.save(os.path.join(dataset_folder, new_filename))
            add_line_to_csv(csv_path, [new_filename, "False"])

    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        files = []
        class_id = "1"
        raw_dataset_folder = os.path.join(temp_directory, "CelebAMask-HQ")
        images_folder = os.path.join(raw_dataset_folder, "CelebA-HQ-img")
        dataset_directory_csv = os.path.join(raw_dataset_folder, "CelebAMask-HQ-attribute-anno.txt")

        with open(dataset_directory_csv, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=' ')
            next(reader)
            head = next(reader)
            for row in itertools.islice(reader, 1000):
                if row[6] == class_id:
                    files.append((row[0]))

        for i, filename in enumerate(files):
            new_filename = f"img_ano_{i}.png"
            img = Image.open(os.path.join(images_folder, filename))
            img.thumbnail((256, 256), Image.Resampling.LANCZOS)
            img.save(os.path.join(dataset_folder, new_filename))
            add_line_to_csv(csv_path, [new_filename, "True"])


if __name__ == '__main__':
    ds_generator = DatasetGeneratorCelebaHQ()

    # ds_generator.run_generate_dataset(ano_fraction=0.1)
    # ds_generator.run_train_direction_matrix()
    # ds_generator.run_train_beta_vae()
    ds_generator.run_create_latent_space_dataset()
