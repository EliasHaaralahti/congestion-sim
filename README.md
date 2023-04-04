# EdgeML

To run the simulation:

- Python 3.9 is currently required, due to requiring Pytorch, which only supports up to 3.9.
- Use pip to install requirements from requirements.txt (pip install -r requirements.txt). The requirement file has torch cuda version, which may or may not cause problems if cuda/gpu not available. Alternatively install cpu version of pytorch.
- Add a hdf5 datafile generated with Carla under folder simulation/runs/. The filename can be provided in main.py or in data.py (change default parameter value). Currently main.py does not accept arguments.
- Run python main.py. The simulation will run and create a json file under simulation/results. The results file contains the output of the simulation.
- Run python visualize.py to visualize the results/results.json file. By changing the main function parameters the visualization will create a video or it will be interactive, which means the simulation is shown one step at a time and any button click progresses the simulation by SKIP_STEPS, which is defined in the visualize.py file. No button to close the window currently, recommended approach is to close the python process with ctrl+c. Might require sometimes clicking the window again after that.