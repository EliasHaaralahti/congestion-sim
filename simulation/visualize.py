import os
import json
import matplotlib.pyplot as plt
from data import DataLoader
from utils.visualizations import *


# Change depending in your needs
CARLA_RUN_NAME = "intersection_5_vehicles.hdf5"
RUN_FOLDER = "medium-intersection_5_vehicles.hdf5-rsu_used_True-1681038881"
IS_INTERACTIVE = False # False creates a video
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
# NOTE: Currently data searches to find corresponding timestep data is done 
# many many times inside the visualization loop. If faster required, 
# pre-process the data to the form data['agents'][timestep].
data_results, data_yolo = read_data(simulation_results_path, yolo_results_path)

dataloader = DataLoader(CARLA_RUN_NAME)
max_timesteps = dataloader.get_simulation_length()
metadata_summary = dataloader.get_metadata_summary()
agents = dataloader.get_entity_ids()
x_waypoints, y_waypoints = zip(*dataloader.get_map()) # List of (x,y).
waypoints = (x_waypoints, y_waypoints)
x_waypoint_min, x_waypoint_max = min(x_waypoints), max(x_waypoints)
y_waypoint_min, y_waypoint_max = min(x_waypoints), max(x_waypoints)

agent_count = metadata_summary['n_vehicles']
# NOTE Currently only two cars supported!
view_car_indexes = [3, 6]


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
        #fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        #fig, axes = plt.subplot_mosaic("ABC;DDD")
        fig, axes = plt.subplot_mosaic(
            [["top left", "top centre", "top right"],
            ["bottom row", "bottom row", "bottom row"]], 
            height_ratios=[1, 1.5], width_ratios=[1, 1, 1]
        )

        fig.suptitle(f"Simulation timestep {timestep}")

        #axes[1,1].legend()
        view_axes = ["top left", "top right"]
        draw_car_views(view_car_indexes, view_axes, axes, agents, data_results,
                       data_yolo, timestep, dataloader)
        draw_information_view(axes["top centre"], agent_count, data_results, 
                              timestep, dataloader)
        draw_map(axes["bottom row"], waypoints, agents, data_results, timestep)
        
        mng = plt.get_current_fig_manager()
        mng.full_screen_toggle()
        if interactive:
            plt.waitforbuttonpress()
        else:
            # save figure if not interactive
            fig.set_size_inches(18, 11) # w, h
            fig.savefig(f'{folder}/fig-{timestep}.png', dpi = 100)
            # Specify no new line ending to replace this line constantly
            print(f"Saved figure {timestep + 1} / {max_timesteps}", end='\r')
        plt.close()
        timestep += 1
        if interactive: # If recording video, timestep skips not applied
            timestep += skip_timesteps
    print("done")

render_visualization(interactive=IS_INTERACTIVE, skip_timesteps=SKIP_TIMESTEPS)
