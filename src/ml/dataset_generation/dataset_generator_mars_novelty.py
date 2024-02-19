import os

import numpy as np
from PIL import Image

from src.ml.dataset_generation.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.dataset_generation.dataset_generation_core import add_line_to_csv


class DatasetGeneratorMvTecHazelnut(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            dataset_name='DS16_mars_novelty',
            num_color_channels=3,
            num_epochs=1000,
            n_latent_space_search_iterations=4000,
            draw_images=True,
            num_imgs=0,
            directions_count=20,
            direction_train_steps=1000,
            only_consider_anos=True,
        )

    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        folder = os.path.join(temp_directory, "MarsNovelty", "train_typical")
        for counter, filename in enumerate(os.listdir(folder)):
            if filename.endswith(".npy"):
                file_name = f"img_norm_{counter}.png"
                img = np.load(os.path.join(folder, filename))
                img = np.take(img, [2, 0, 1], axis=2)
                img = np.interp(img, (img.min(), img.max()), (0, 1))
                img = Image.fromarray(np.uint8(img * 255))
                img.save(os.path.join(dataset_folder, file_name))
                add_line_to_csv(csv_path, [file_name, "False"])

    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        folder = os.path.join(temp_directory, "MarsNovelty", "anos")
        for counter, filename in enumerate(os.listdir(folder)):
            if filename.endswith(".npy"):
                file_name = f"img_ano_{counter}.png"
                img = np.load(os.path.join(folder, filename))
                img = np.take(img, [2, 0, 1], axis=2)
                img = np.interp(img, (img.min(), img.max()), (0, 1))
                img = Image.fromarray(np.uint8(img * 255))
                img.save(os.path.join(dataset_folder, file_name))
                add_line_to_csv(csv_path, [file_name, "True"])


if __name__ == '__main__':
    ds_generator = DatasetGeneratorMvTecHazelnut()

    # ds_generator.run_generate_dataset(ano_fraction=0.1)
    # ds_generator.run_equalize_image_sizes()
    # ds_generator.run_train_and_save_gan(display_generator_test=True)
    # ds_generator.run_train_direction_matrix()
    # ds_generator.run_train_beta_vae()
    ds_generator.run_create_latent_space_dataset()
