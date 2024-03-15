import os

from src.ml.dataset_generation.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.models.StyleGAN import dnnlib, legacy
from src.ml.models.StyleGAN.ano_detection.stylegan_generator_wrapper import StyleGANGeneratorWrapper
from src.ml.models.StyleGAN.ano_detection.stylegan_reconstructor import StyleGANReconstructor


class DatasetGeneratorHazelnutStylegan(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            size_z=512,
            dataset_name='DS18_mvtec_hazelnut_stylegan',
            num_channels=3,
            num_epochs=1500,
            n_latent_space_search_iterations=5000,
            draw_images=True,
            num_imgs=0,
            directions_count=30,
            direction_train_steps=5000,
            direction_batch_size=14,
            direction_train_shift_scale=3.0,
            max_retries=3,
            retry_threshold=0.002,
            only_consider_anos=True,
            retry_check_after_iter=1000,
            start_learning_rate=0.0001,
            print_every_n_iters=2000,
            use_discriminator_for_latent_space_mapping=False,
            stylegan=True,
        )

        self.reconstructor = StyleGANReconstructor(directions_count=self.directions_count,
                                                   num_channels=self.num_channels,
                                                   width=2,
                                                   shape=[14, 6, 64, 64]).to(self.device)

        stylegan_models_path = os.path.join(self.root_dir, self.dataset_name, 'models.pkl')
        with dnnlib.util.open_url(str(stylegan_models_path)) as f:
            networks = legacy.load_network_pkl(f)
            G = networks['G'].to(self.device)
        self.generator = StyleGANGeneratorWrapper(G).to(self.device)

    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        pass

    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        pass


if __name__ == '__main__':
    ds_generator = DatasetGeneratorHazelnutStylegan()

    ds_generator.run_train_direction_matrix()
    # ds_generator.run_create_latent_space_dataset()
