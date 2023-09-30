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
    - Running one of the dataset-specific scripts (i.e. "generate_dataset_cifar10_plane_horse.py") will create a dataset inside of the data folder
      - Inside the scripts some of the hyperparameter for the pipeline can be adjusted. Those are usually at the top of the script. Also two functions are defined that
      will be called by the pipe to generate the normal and the anomalous image files
      - Please have a look at one of the newer pipes (like "generate_dataset_stl10_plane_horse.py") since they already contain all of the functionalities an can be used as a blueprint for other dataset-creation-pipes
      - Some of the functionalities are not yet parametrized inside the specific pipe scripts. That especially includes the functions that create the latent-space-representations.
      To adjust those, change the values in "generate_dataset.py", starting line 279

```
max_retries = 5
opt_threshold = 0.05
ignore_rules_below_threshold = 0.1
immediate_retry_threshold = 0.15

mapped_z, reconstruction_loss, retry = lsm.map_image_to_point_in_latent_space(image=data_point,
                                                                              max_opt_iterations=max_opt_iterations,
                                                                              plateu_threshold=0,
                                                                              check_every_n_iter=5000,
                                                                              learning_rate=0.001,
                                                                              print_every_n_iters=5000,
                                                                              retry_after_n_iters=100000,
                                                                              ignore_rules_below_threshold=ignore_rules_below_threshold,
                                                                              opt_threshold=opt_threshold,
                                                                              immediate_retry_threshold=immediate_retry_threshold)
```

### The datasets in the data folder are automatically loaded in the app and can be used for labeling. In order to work properly, the structure must look as follows

- PARENT FOLDER
  - DATASET -> contains latent space representations 
  - DATASET_RAW -> contains image data
  - DIRECTION_MATRICES -> contains model-dumps of one or multiple direction-matrices
  - discriminator.pkl
  - generator.pkl -> just for backup, not needed during labeling/testing
  - reconstructor.pkl -> just for backup, not needed during labeling/testing