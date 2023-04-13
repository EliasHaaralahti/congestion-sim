# EdgeML

## Prerequisites

- Python 3.8 or 3.9
  - Python 3.8 is required for simulating the traffic because CARLA does not support newer versions.
  - If you only want to run the discrete-event simulation, you can also use Python 3.9, which PyTorch supports.
- [CARLA](https://carla.org/)

## Installation

- Install the required dependencies by running `pip install -r requirements.txt`.
  - The requirement file has Torch CUDA version, which may or may not cause problems if CUDA/GPU is not available. Alternatively, you can install the CPU version of PyTorch.

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