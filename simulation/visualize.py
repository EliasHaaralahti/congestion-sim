import os
import json
import matplotlib.pyplot as plt
from data import DataLoader
from utils.visualizations import *

IS_INTERACTIVE = True
SKIP_TIMESTEPS = 30

RESULTS_FILE = "results.json"
path = os.path.join('results', RESULTS_FILE)

with open(path) as json_file:
    data = json.load(json_file)

dataloader = DataLoader()
metadata_summary = dataloader.get_metadata_summary()
agents = dataloader.get_entity_ids()
x_waypoints, y_waypoints = zip(*dataloader.get_map()) # List of (x,y).
waypoints = (x_waypoints, y_waypoints)
x_waypoint_min, x_waypoint_max = min(x_waypoints), max(x_waypoints)
y_waypoint_min, y_waypoint_max = min(x_waypoints), max(x_waypoints)

agent_count = metadata_summary['n_vehicles']


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
        draw_car_views(axes, agents, data, timestep, dataloader)
        draw_information_view(axes[1,0], agent_count, data, timestep)
        draw_map(axes[1,1], waypoints, agents, data, timestep)
        
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
