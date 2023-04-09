import os
import sys
import json
import getopt
import matplotlib.pyplot as plt
from data import DataLoader
from utils.visualizations import *


folder = "visualization/figures"


def read_data(results_path, yolo_path):
    with open(results_path) as json_file:
        data_results = json.load(json_file)

    with open(yolo_path) as yolo_file:
        data_yolo = json.load(yolo_file)
        # Convert the items in the list back to dicts (from str)
        for i in range(len(data_yolo)):
            data_yolo[i] = json.loads(data_yolo[i])
    
    return data_results, data_yolo


def render_visualization(run_folder, carla_environment, interactive=False,
                         skip_timesteps=0):
    
    results_path = os.path.join("results", run_folder)
    simulation_results_path = os.path.join(results_path, "results.json")
    yolo_results_path = os.path.join(results_path, "yolo_results.json")
    # NOTE: Currently data searches to find corresponding timestep data is done 
    # many many times inside the visualization loop. If faster required, 
    # pre-process the data to the form data['agents'][timestep].
    data_results, data_yolo = read_data(simulation_results_path, yolo_results_path)

    dataloader = DataLoader(carla_environment)
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
        
    if interactive:
        plt.ion()
    else:
        # Ensure figures folder is created if interactive = False
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
        if interactive: # If recording video, timestep skips not applied!
            timestep += skip_timesteps
    print("done")


def print_help():
    help_text = "Ensure main.py has been run before this. If not, read the README.\n" \
        "Usage: python main.py. Possible arguments:\n" \
        "\t-h - Help\n" \
        "\t--save_video - Disables interactive mode and saves each figure.\n" \
        "\t\tUse utils/video.py to convert them to a video.\n" \
        "\t--skip <integer> - In interactive mode when advancing skip <integer> frames.\n" \
        "\t--environment <string> - The CARLA environment file name, stored under runs/.\n" \
        "\t--run_folder <string> - Path of simulation results\n" \
        "\n" \
        "Example:\n" \
        "python visualize.py --environment intersection_5_vehicles.hdf5 --run_folder " \
        "medium-intersection_5_vehicles.hdf5-rsu_used_True-1681047827 --skip 20\n" \
        "python visualize.py --environment intersection_5_vehicles.hdf5 --run_folder " \
        "medium-intersection_5_vehicles.hdf5-rsu_used_True-1681047827 --save_video"
        
    print(help_text)


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["save_video", "skip=",
                                                   "environment=", "run_folder="])
    RUN = True
    IS_INTERACTIVE = True
    SKIP_TIMESTEPS = 0
    RUN_FOLDER = ""
    CARLA_DATA_NAME = ""
    for opt, arg in opts:
        if opt == "-h":
            print_help()
            RUN = False
        if opt == "--save_video":
            IS_INTERACTIVE = False
        if opt == "--skip":
            SKIP_TIMESTEPS = int(arg)
        if opt == "--environment":
            CARLA_DATA_NAME = arg
        if opt == "--run_folder":
            RUN_FOLDER = arg

    if RUN:
        if RUN_FOLDER == "" or CARLA_DATA_NAME == "":
            print("Missing arguments. See -h for help")
            sys.exit(2)

        if not IS_INTERACTIVE and SKIP_TIMESTEPS > 0:
            print("Cannot add skip argument if not interactive!")
            sys.exit(2)

        render_visualization(run_folder=RUN_FOLDER,
                             carla_environment=CARLA_DATA_NAME,
                             interactive=IS_INTERACTIVE, 
                             skip_timesteps=SKIP_TIMESTEPS)
        
        if not IS_INTERACTIVE:
            print(f"Video saved as figures to folder {folder}.\n" \
                "Use utils/video.py to convert them to a video if necessary.")
        print("Done.")
