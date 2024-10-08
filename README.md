# AD-with-GANs
Repository for Master Thesis "Weakly Supervised Anomaly Detection by Utilizing Human Feedback on Latent Space Representations of a Generative Model" by Yashar Nehls

## Code Repository
The code repository is structured as follows:

### Folder "src" contains all source files and is subdivided in "backend", "frontend" and "ml"
  - "backend" and "frontend" contain a python web-api (based on FastAPI) and an angular app
    - to start the backend, run main.py inside the backend folder
    - to start the frontend run "npm i" and "ng serve" inside frontend/latent-space-discoverer
      - once the frontend is running, open http://localhost:4200/ inside a browser


### Folder "data" contains the different datasets that can be chosen for labeling/experiments
  - All folder/files, including image-datasets, latent-space representations of the image data, generator-/discriminator-/reconstructor-model-dump,
  inside the data folder are generated by the dataset generation pipeline.
    - "./src/ml/datasets/generate_dataset.py" holds all the necessary functions. They include
      - generating the image-dataset
      - training of the GAN
      - training of the direction-matrix
      - creation of the latent space representations
    - Running one of the dataset-specific scripts (i.e. "generate_dataset_cifar10_plane_horse.py") will create a dataset inside the data folder
      - Inside the scripts some of the hyperparameter for the pipeline can be adjusted. Those are usually at the top of the script. Also, two functions are defined that
      will be called by the pipe to generate the normal and the anomalous image files
      - Please have a look at one of the newer pipes (like "generate_dataset_stl10_plane_horse.py") since they already contain all the functionalities can be used as a blueprint for other dataset-creation-pipes
    - For every dataset, the corresponding generator must be specified in the controller in order for the webservice be able to work properly. This is done in the "main_controller.py" inside the "get_generator_by_dataset_name" function

### The datasets in the data folder are automatically loaded in the app and can be used for labeling. In order to work properly, the structure must look as follows

- PARENT FOLDER
  - DATASET -> contains latent space representations 
  - DATASET_RAW -> contains image data
  - DIRECTION_MATRICES -> contains model-dumps of one or multiple direction-matrices
  - discriminator.pkl
  - generator.pkl -> just for backup, not needed during labeling/testing
  - reconstructor.pkl -> just for backup, not needed during labeling/testing