import os
import torch
import numpy as np
import base64
import io
from PIL import Image
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

from src.ml.latent_direction_visualizer import LatentDirectionVisualizer, get_random_strip_as_numpy_array
from src.ml.models.generator import Generator
from src.ml.models.matrix_a_linear import MatrixALinear
from src.backend.models.ImageStripModel import ImageStripModel
from src.ml.tools.utils import generate_noise


class MainController:
    def __init__(self, generator_path, z_dim):
        self.z_dim = z_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.g: Generator = Generator(size_z=self.z_dim, num_feature_maps=64, num_color_channels=1)
        self.g.load_state_dict(torch.load(generator_path, map_location=torch.device(self.device)))
        self.matrix_a: MatrixALinear = MatrixALinear(input_dim=self.z_dim, output_dim=self.z_dim)

    def get_image_strip(self):
        image_list = []
        img_arr = get_random_strip_as_numpy_array(os.path.abspath("../out_dir/data.npy"))
        for idx, i in enumerate(img_arr):
            two_d = (np.reshape(i, (28, 28)) * 255).astype(np.uint8)
            img = Image.fromarray(two_d, 'L')

            with io.BytesIO() as buf:
                img.save(buf, format='PNG')
                img_str = base64.b64encode(buf.getvalue())

            image_list.append(ImageStripModel(position=idx, image=img_str))

        return image_list

    def get_shifted_images(self, z, shifts_range, shifts_count, dim):
        image_list = []
        z = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(torch.FloatTensor(z), 0), -1), 2)
        visualizer = LatentDirectionVisualizer(matrix_a=self.matrix_a, generator=self.g, device=self.device)
        shifted_images = visualizer.create_shifted_images(z, shifts_range, shifts_count, dim)

        for idx, i in enumerate(shifted_images):
            two_d = (np.reshape(i.numpy(), (28, 28)) * 255).astype(np.uint8)
            img = Image.fromarray(two_d, 'L')

            with io.BytesIO() as buf:
                img.save(buf, format='PNG')
                img_str = base64.b64encode(buf.getvalue())

            image_list.append(ImageStripModel(position=idx, image=img_str))

        return image_list

    def save_to_db(self):
        uri = "mongodb+srv://admin_lse:5bKIwnZbTM7sGDjh@cluster0.6gyi8ct.mongodb.net/?retryWrites=true&w=majority"
        client = MongoClient(uri, server_api=ServerApi('1'))
        try:
            client.admin.command('ping')
            print("Pinged your deployment. You successfully connected to MongoDB!")
            return True
        except Exception as e:
            print(e)
            return False

    def get_random_noise(self, z_dim):
        return generate_noise(batch_size=1, z_dim=z_dim, device=self.device)
