import os

from PIL import Image, ImageOps

from src.ml.dataset_generation.abstract_dataset_generator import AbstractDatasetGenerator
from src.ml.dataset_generation.dataset_generation_core import add_line_to_csv


class DatasetGeneratorMvTecHazelnut(AbstractDatasetGenerator):
    def __init__(self):
        super().__init__(
            dataset_name='DS15_mvtec_hazelnut',
            num_channels=3,
            num_epochs=1500,
            n_latent_space_search_iterations=3500,
            draw_images=False,
            num_imgs=0,
            directions_count=20,
            direction_train_steps=1000
        )

    def generate_normals(self, dataset_folder, csv_path, temp_directory):
        mvtec_hazelnut_normals_folder = os.path.join(temp_directory, "mvtec_hazelnut", "normals")
        for counter, filename in enumerate(os.listdir(mvtec_hazelnut_normals_folder)):
            if filename.endswith(".png"):
                file_name = f"img_norm_{counter}_0.png"
                img = Image.open(os.path.join(mvtec_hazelnut_normals_folder, filename))
                img.thumbnail((64, 64), Image.Resampling.LANCZOS)
                img.save(os.path.join(dataset_folder, file_name))
                add_line_to_csv(csv_path, [file_name, "False"])

                file_name = f"img_norm_{counter}_0_1.png"
                img_mirror = ImageOps.mirror(img)
                img_mirror.save(os.path.join(dataset_folder, file_name))
                add_line_to_csv(csv_path, [file_name, "False"])

                file_name = f"img_norm_{counter}_1.png"
                img = img.rotate(90)
                img.save(os.path.join(dataset_folder, file_name))
                add_line_to_csv(csv_path, [file_name, "False"])

                file_name = f"img_norm_{counter}_1_1.png"
                img_mirror = ImageOps.mirror(img)
                img_mirror.save(os.path.join(dataset_folder, file_name))
                add_line_to_csv(csv_path, [file_name, "False"])

                file_name = f"img_norm_{counter}_2.png"
                img = img.rotate(90)
                img.save(os.path.join(dataset_folder, file_name))
                add_line_to_csv(csv_path, [file_name, "False"])

                file_name = f"img_norm_{counter}_2_1.png"
                img_mirror = ImageOps.mirror(img)
                img_mirror.save(os.path.join(dataset_folder, file_name))
                add_line_to_csv(csv_path, [file_name, "False"])

    def generate_anomalies(self, dataset_folder, csv_path, temp_directory, ano_fraction):
        mvtec_hazelnut_anomalies_folder = os.path.join(temp_directory, "mvtec_hazelnut", "anomalies")
        counter = 0
        for _, dirs, files in os.walk(mvtec_hazelnut_anomalies_folder):
            for directory in dirs:
                sub_folder = os.path.join(mvtec_hazelnut_anomalies_folder, directory)
                for i, filename in enumerate(os.listdir(sub_folder)):
                    if filename.endswith(".png"):
                        file_name = f"img_ano_{counter}_0.png"
                        img = Image.open(os.path.join(sub_folder, filename))
                        img.thumbnail((64, 64), Image.Resampling.LANCZOS)
                        img.save(os.path.join(dataset_folder, file_name))
                        add_line_to_csv(csv_path, [file_name, "True"])

                        file_name = f"img_ano_{counter}_0_1.png"
                        img_mirror = ImageOps.mirror(img)
                        img_mirror.save(os.path.join(dataset_folder, file_name))
                        add_line_to_csv(csv_path, [file_name, "True"])

                        file_name = f"img_ano_{counter}_1.png"
                        img = img.rotate(90)
                        img.save(os.path.join(dataset_folder, file_name))
                        add_line_to_csv(csv_path, [file_name, "True"])

                        file_name = f"img_ano_{counter}_1_1.png"
                        img_mirror = ImageOps.mirror(img)
                        img_mirror.save(os.path.join(dataset_folder, file_name))
                        add_line_to_csv(csv_path, [file_name, "True"])

                        file_name = f"img_ano_{counter}_2.png"
                        img = img.rotate(90)
                        img.save(os.path.join(dataset_folder, file_name))
                        add_line_to_csv(csv_path, [file_name, "True"])

                        file_name = f"img_ano_{counter}_2_1.png"
                        img_mirror = ImageOps.mirror(img)
                        img_mirror.save(os.path.join(dataset_folder, file_name))
                        add_line_to_csv(csv_path, [file_name, "True"])

                        file_name = f"img_ano_{counter}_3.png"
                        img = img.rotate(90)
                        img.save(os.path.join(dataset_folder, file_name))
                        add_line_to_csv(csv_path, [file_name, "True"])

                        file_name = f"img_ano_{counter}_3_1.png"
                        img_mirror = ImageOps.mirror(img)
                        img_mirror.save(os.path.join(dataset_folder, file_name))
                        add_line_to_csv(csv_path, [file_name, "True"])

                        counter += 1


if __name__ == '__main__':
    ds_generator = DatasetGeneratorMvTecHazelnut()

    # ds_generator.run_generate_dataset(ano_fraction=0.1)
    # ds_generator.run_equalize_image_sizes()
    # ds_generator.run_train_and_save_gan(display_generator_test=True)
    ds_generator.run_train_direction_matrix()
    # ds_generator.run_train_beta_vae()
    # ds_generator.run_create_latent_space_dataset()
