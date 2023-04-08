import os
import json
import matplotlib.pyplot as plt
from data import DataLoader
from utils.visualizations import *


# Change depending in your needs
CARLA_RUN_NAME = "intersection_5_vehicles.hdf5"
RUN_FOLDER = "medium-intersection_5_vehicles.hdf5-rsu_used_True-1680983400"
IS_INTERACTIVE = True # False creates a video
SKIP_TIMESTEPS = 50


def read_data(results_path, yolo_path):
    with open(results_path) as json_file:
        data_results = json.load(json_file)

    with open(yolo_path) as yolo_file:
        data_yolo = json.load(yolo_file)
        # Convert the items in the list back to dicts (from str)
        for i in range(len(data_yolo)):
            data_yolo[i] = json.loads(data_yolo[i])
    
    return data_results, data_yolo


results_path = os.path.join("results", RUN_FOLDER)
simulation_results_path = os.path.join(results_path, "results.json")
yolo_results_path = os.path.join(results_path, "yolo_results.json")
data_results, data_yolo = read_data(simulation_results_path, yolo_results_path)

dataloader = DataLoader(CARLA_RUN_NAME)
max_timesteps = dataloader.get_simulation_length()
#data_results, data_yolo = convert_data(data_results, data_yolo, max_timesteps)
metadata_summary = dataloader.get_metadata_summary()
agents = dataloader.get_entity_ids()
x_waypoints, y_waypoints = zip(*dataloader.get_map()) # List of (x,y).
waypoints = (x_waypoints, y_waypoints)
x_waypoint_min, x_waypoint_max = min(x_waypoints), max(x_waypoints)
y_waypoint_min, y_waypoint_max = min(x_waypoints), max(x_waypoints)

agent_count = metadata_summary['n_vehicles']
# TODO Currently only two cars supported!
view_car_indexes = [3, 5]


def render_visualization(interactive=False, skip_timesteps=0):
    if interactive:
        plt.ion()
    else:
        # Ensure figures folder is created if interactive = False
        folder = "visualization/figures"
        if not os.path.exists(folder):
            os.makedirs(folder)

    timestep = 0
    max_timesteps = dataloader.get_simulation_length()
    while timestep <= max_timesteps - 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f"Simulation timestep {timestep}")
    
        # TODO: Optimize and clean! Same loops essentially ran twice.
        # Though it might be useful to keep it like that to ensure modularity.
        draw_car_views(view_car_indexes, axes, agents, data_results, data_yolo, 
                       timestep, dataloader)
        draw_information_view(axes[1,0], agent_count, data_results, timestep)
        draw_map(axes[1,1], waypoints, agents, data_results, timestep)
        
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        if interactive:
            plt.waitforbuttonpress()
        else:
            # save figure if not interactive
            fig.savefig(f'{folder}/fig-{timestep}.png')
            # Specify no new line ending to replace this line constantly
            print(f"Saved figure {timestep+1} / {max_timesteps}", end='\r')
        plt.close()
        timestep += 1 + skip_timesteps
    print("done")

render_visualization(interactive=IS_INTERACTIVE, skip_timesteps=SKIP_TIMESTEPS)
