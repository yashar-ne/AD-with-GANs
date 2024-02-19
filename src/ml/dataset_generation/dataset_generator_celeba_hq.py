import csv
import itertools
import os

from PIL import Image

from src.ml.dataset_generation.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.dataset_generation.generate_dataset import add_line_to_csv
from src.ml.models.StyleGAN import dnnlib, legacy
from src.ml.models.StyleGAN.ano_detection.stylegan_generator_wrapper import StyleGANGeneratorWrapper
from src.ml.models.StyleGAN.ano_detection.stylegan_reconstructor import StyleGANReconstructor


class DatasetGeneratorCelebaHQ(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            size_z=512,
            dataset_name='DS17_celeba_hq_bald',
            num_channels=3,
            num_epochs=1500,
            n_latent_space_search_iterations=1500,
            draw_images=False,
            num_imgs=0,
            directions_count=2,
            direction_train_steps=1000,
            direction_batch_size=14,
            max_retries=3,
            retry_threshold=0.1,
            only_consider_anos=False,
            retry_check_after_iter=800,
            start_learning_rate=0.001,
            print_every_n_iters=2000,
            use_discriminator_for_latent_space_mapping=False,
            stylegan=True,
        )

        self.reconstructor = StyleGANReconstructor(directions_count=self.directions_count,
                                                   num_channels=self.num_channels,
                                                   width=2).to(self.device)

        stylegan_models_path = os.path.join(self.root_dir, self.dataset_name, 'stylegan_pretrained_models.pkl')
        with dnnlib.util.open_url(str(stylegan_models_path)) as f:
            networks = legacy.load_network_pkl(f)
            G = networks['G_ema'].to(self.device)
        self.generator = StyleGANGeneratorWrapper(G).to(self.device)

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
            for row in itertools.islice(reader, 7000):
                if row[6] == class_id:
                    files.append((row[0]))

        print(f"Number Normals: {len(files)}")
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
            next(reader)
            for row in itertools.islice(reader, 30000):
                if row[6] == class_id:
                    files.append((row[0]))

        print(f"Number Anos: {len(files)}")
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
    ds_generator.run_train_beta_vae()
    # ds_generator.run_create_latent_space_dataset()
