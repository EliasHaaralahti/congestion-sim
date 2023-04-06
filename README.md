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

### Running the discrete-event simulation

- Add a HDF5 datafile generated with CARLA under folder simulation/runs/. The filename can be provided in main.py or in data.py (change default parameter value). Currently main.py does not accept arguments.
- Run python main.py. The simulation will run and create a json file under simulation/results. The results file contains the output of the simulation.
- Run python visualize.py to visualize the results/results.json file. By changing the main function parameters the visualization will create a video or it will be interactive, which means the simulation is shown one step at a time and any button click progresses the simulation by SKIP_STEPS, which is defined in the visualize.py file. No button to close the window currently, recommended approach is to close the python process with ctrl+c. Might require sometimes clicking the window again after that.

## Division of work

### Aleksi

- Simulating traffic and collecting sensor data using CARLA
- Fine-tuning AlexNet-based congestion detection model

### Elias

- Discrete-event simulation