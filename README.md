# CongestionSim

## Aalto University project for course CS-E4875 - Research Project in Machine Learning, Data Science and Artificial Intelligence, 5 ECTS.

The goal of the project was to estimate traffic congestion using sensors of vehicles. This was done by simulating a city with vehicles and pedestrians in CARLA and storing the results in HDF5 format. The results were then processed by a discrete-event simulation (DES) that detects objects using the YOLOv5 model and estimates the position of detected objects in a 2D world relative to the camera. The 2D world is then used for analysis, such as congestion detection and very simple collision warning detection. A separate CNN-based deep learning model was also trained to classify the congestion status of an area from images produced by CARLA.

<p align="center">
  <img src="https://github.com/EliasHaaralahti/congestion-sim/blob/main/graphics/CongestionSimGif2.gif" alt="graphics/CongestionSimGif2.gif" />
</p>

## High-level architecture of our system

<p align="center">
  <img src="https://github.com/EliasHaaralahti/congestion-sim/blob/main/graphics/architecture.png" alt="graphics/architecture.png" />
</p>

## The repository has the following components:

- Folder `carla`, which is our CARLA integration and allows running simulations of varying sizes and storing data to HDF5 format.
- Folder `congestion_detection`, which is the CNN-based model to classify congestion status from a single image.
- Folder `simulation`, which contains a DES simulation to process the CARLA simulation data, stored in HDF5 format. It also contains utilities for visualizing the DES simulation results and analyzing the data.

## Prerequisites

- Python 3.8 or 3.9
  - Python 3.8 is required for simulating the traffic because CARLA does not support newer versions.
  - If you only want to run the discrete-event simulation, you can also use Python 3.9, which is supported by PyTorch.
- [CARLA](https://carla.org/)

## Installation

- Install the required dependencies by running `pip install -r requirements.txt`.
  - The requirement file has Torch CUDA version, which may or may not cause problems if CUDA or GPU is not available. Alternatively, you can install the CPU version of PyTorch.

## Running the project

### Simulating traffic and collecting sensor data using CARLA

#### Collecting image data for the discrete-event simulation

- To generate the HDF5 file required for the DES, run `python client.py` in the `carla` directory. The resulting HDF5 file will be saved to `carla/runs`.
  - If you want to modify the vehicle locations, you can run `python visualize_spawn_points.py` to visualize available spawn points.

#### Collecting image data from RSUs to train and evaluate the congestion detection model

- Run `python training_data_generator.py`. The resulting HDF5 file will be saved to `carla/runs`.

### Training the congestion detection model

- Once you have created the HDF5 file using the instructions above, you can transform it into an image dataset by running `python make_dataset.py` in the `congestion_detection` directory. The resulting dataset will be saved to `congestion_detection/data`.
- Now you can train the model by running `python train.py`. The model weights will be saved to `congestion_detection/models`.
- Once you have trained the model, you can evaluate it using a separate test set by running `python evaluate.py`.
- To detect congestion in the simulated scenarios, run `python classify_intersections.py`.

### Running the discrete-event simulation

- Generate a HDF5 datafile with CARLA and place the file under `simulation/runs/`.
- Run the DES simulation by executing main.py with proper command line arguments. For more information run the command `python main.py -h`. When the simulation is done, the output will be placed as json files under `simulation/results/<run_id>/`. The file `results.json` contains the simulation output, while the file `yolo_results.json` contains information about YOLO bounding boxes for visualization purposes.
- To visualize the results of the DES simulation, run the file `visualize.py` with the proper command line arguments. See `python visualize.py -h` for more information. The visualization is either interactive or saves a video. To proceed in the interactive visualization, click any key.

## Division of work

### Aleksi

- Simulating traffic and collecting sensor data using CARLA.
- Fine-tuning AlexNet-based congestion detection model.

### Elias

- Building the discrete-event simulation, object detection and congestion analysis.
- Building visualization tools for the DES results.