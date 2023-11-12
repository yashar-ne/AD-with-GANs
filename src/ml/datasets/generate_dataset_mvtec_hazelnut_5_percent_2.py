import sys

sys.path.append('/home/yashar/git/AD-with-GANs/')

from PIL import Image, ImageOps
import torch
import os

import torchvision.transforms as transforms

from src.ml.datasets.generate_dataset import add_line_to_csv, create_latent_space_dataset
from src.ml.models.mvtec128.mvtec_discriminator import MvTecDiscriminator
from src.ml.models.mvtec128.mvtec_generator import MvTecGenerator
from src.ml.models.mvtec128.mvtec_reconstructor import MvTecReconstructor

# Hyperparameter
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 512
learning_rate = 0.001
gan_num_epochs = 2500
num_color_channels = 3
num_feature_maps_g = 64
num_feature_maps_d = 64
image_size = 128
size_z = 100
num_imgs = 0
save_checkpoint_every_n_epoch = 50

directions_count = 30
direction_train_steps = 2500
useBias = True

max_opt_iterations = 7500
max_retries = 5
opt_threshold = 0.035
ignore_rules_below_threshold = 0.1
immediate_retry_threshold = 0.1
only_consider_anos = True
plateu_threshold = -1
check_every_n_iter = 2500
start_learning_rate = 0.0001
print_every_n_iters = 2500
retry_after_n_iters = 2500
draw_images = True

map_anomalies = True
map_normals = True
tmp_directory = '../data_temp'
data_root_directory = '../data'
dataset_name = 'DS11_mvtec_hazelnut_5_percent_2'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

mvtec_128_generator = MvTecGenerator(size_z=size_z, num_feature_maps=num_feature_maps_g, dropout_rate=0.1).to(device)
mvtec_128_discriminator = MvTecDiscriminator(num_feature_maps=num_feature_maps_d, dropout_rate=0.1).to(device)
mvtec_128_reconstructor = MvTecReconstructor(directions_count=directions_count, width=2).to(device)


def generate_normals(dataset_folder, csv_path, temp_directory):
    mvtec_hazelnut_normals_folder = os.path.join(temp_directory, "mvtec_hazelnut", "normals")
    for counter, filename in enumerate(os.listdir(mvtec_hazelnut_normals_folder)):
        if filename.endswith(".png"):
            file_name = f"img_norm_{counter}_0.png"
            img = Image.open(os.path.join(mvtec_hazelnut_normals_folder, filename))
            img.thumbnail((128, 128), Image.Resampling.LANCZOS)
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

            # file_name = f"img_norm_{counter}_3.png"
            # img = img.rotate(90)
            # img.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])
            #
            # file_name = f"img_norm_{counter}_3_1.png"
            # img_mirror = ImageOps.mirror(img)
            # img_mirror.save(os.path.join(dataset_folder, file_name))
            # add_line_to_csv(csv_path, [file_name, "False"])


def generate_anomalies(dataset_folder, csv_path, temp_directory, ano_fraction):
    mvtec_hazelnut_anomalies_folder = os.path.join(temp_directory, "mvtec_hazelnut", "anomalies")
    counter = 0
    for _, dirs, files in os.walk(mvtec_hazelnut_anomalies_folder):
        for directory in dirs:
            sub_folder = os.path.join(mvtec_hazelnut_anomalies_folder, directory)
            for i, filename in enumerate(os.listdir(sub_folder)):
                if filename.endswith(".png"):
                    file_name = f"img_ano_{counter}_0.png"
                    img = Image.open(os.path.join(sub_folder, filename))
                    img.thumbnail((128, 128), Image.Resampling.LANCZOS)
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


# ################## RUN ####################

# generate_dataset(root_dir=data_root_directory,
#                  temp_directory=tmp_directory,
#                  dataset_name=dataset_name,
#                  generate_normals=generate_normals,
#                  generate_anomalies=generate_anomalies,
#                  ano_fraction=0.1)

# train_and_save_gan(root_dir=data_root_directory,
#                    dataset_name=dataset_name,
#                    size_z=size_z,
#                    num_epochs=gan_num_epochs,
#                    num_feature_maps_g=num_feature_maps_g,
#                    num_feature_maps_d=num_feature_maps_d,
#                    num_color_channels=num_color_channels,
#                    batch_size=batch_size,
#                    device=device,
#                    learning_rate=learning_rate,
#                    generator=mvtec_128_generator,
#                    discriminator=mvtec_128_discriminator,
#                    transform=transform,
#                    num_imgs=num_imgs,
#                    save_checkpoint_every_n_epoch=save_checkpoint_every_n_epoch)

# train_direction_matrix(root_dir=data_root_directory,
#                        dataset_name=dataset_name,
#                        direction_count=directions_count,
#                        steps=direction_train_steps,
#                        device=device,
#                        use_bias=useBias,
#                        generator=mvtec_128_generator,
#                        reconstructor=mvtec_128_reconstructor)

create_latent_space_dataset(root_dir=data_root_directory,
                            transform=transform,
                            dataset_name=dataset_name,
                            size_z=size_z,
                            num_feature_maps_g=num_feature_maps_g,
                            num_feature_maps_d=num_feature_maps_d,
                            num_color_channels=num_color_channels,
                            device=device,
                            n_iterations=max_opt_iterations,
                            generator=mvtec_128_generator,
                            discriminator=mvtec_128_discriminator,
                            max_retries=max_retries,
                            opt_threshold=opt_threshold,
                            ignore_rules_below_threshold=ignore_rules_below_threshold,
                            retry_threshold=immediate_retry_threshold,
                            only_consider_anos=only_consider_anos,
                            plateu_threshold=plateu_threshold,
                            retry_check_after_iter=check_every_n_iter,
                            learning_rate=start_learning_rate,
                            print_every_n_iters=print_every_n_iters,
                            retry_after_n_iters=retry_after_n_iters,
                            draw_images=draw_images)

# test_generator(128, size_z, mvtec_128_generator, '/home/yashar/git/AD-with-GANs/checkpoints/DS10_mvtec_hazelnut_5_percent/generator_epoch_4600_iteration_4601.pkl', device)

# checkpoints_folder = '/home/yashar/git/AD-with-GANs/checkpoints/DS10_mvtec_hazelnut_5_percent'
# png_folder = os.path.join(checkpoints_folder, 'plots')
# os.makedirs(png_folder, exist_ok=True)
#
# for _, dirs, files in os.walk(checkpoints_folder):
#     for generator_filename in files:
#         if generator_filename.startswith("generator"):
#             # get filename but without file extension
#             filename_png = f"{generator_filename.split('.')[0]}.png"
#             test_generator_and_save_plot(128, size_z, mvtec_128_generator,
#                                          os.path.join(checkpoints_folder, generator_filename), device,
#                                          os.path.join(png_folder, filename_png))
