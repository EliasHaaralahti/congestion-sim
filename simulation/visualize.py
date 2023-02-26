import os
import json
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.markers import MarkerStyle
from data import DataLoader


def get_relevant_coordinates(agents, timestep, processing_results):
    """
    Find min and max values for both x and y coordinates for each agent
    and detected entity. Used for scaling the visualization.
    """   
    coordinates = []
    for a in agents:
        a_state = processing_results[timestep][a]
        coordinates.append( (a_state['x'], a_state['y']) )
        for d in a_state['detected']:
            coordinates.append( (d[0], d[1]) )

    x, y = zip(*coordinates)
    return (min(x), max(x), min(y), max(y))


RESULTS_FILE = "results.json"
path = os.path.join('results', RESULTS_FILE)

with open(path) as json_file:
    data = json.load(json_file)

print(data.keys())

dataloader = DataLoader()
agents = dataloader.read_agent_ids()
x_waypoints, y_waypoints = zip(*dataloader.get_map()) # List of (x,y).
x_waypoint_min, x_waypoint_max = min(x_waypoints), max(x_waypoints)
y_waypoint_min, y_waypoint_max = min(x_waypoints), max(x_waypoints)

# TODO: Use cmap or something to automatically pick colors.
colors = ['red', 'blue', 'yellow', 'black', 'orange', 'green']

plt.ion()
timestep = 0
max_timesteps = dataloader.get_simulation_length()
while timestep <= max_timesteps:
    f, axes = plt.subplots(2, 2)
    min_x, max_x, min_y, max_y = get_relevant_coordinates(
        agents, timestep, data['processing_results'])

    print(f"Result: minx: {min_x}, maxx: {max_x}, miny: {min_y}, maxy: {max_y}")
    # Full size window for displaying map
    axes[1,0].scatter(x_waypoints, y_waypoints, c='k', s=0.05, zorder=0)
    axes[1,0].set_xlim([x_waypoint_min-10, x_waypoint_max+10])
    axes[1,0].set_ylim([y_waypoint_min+10, y_waypoint_max+50])
    # Zoomed window for displaying map
    axes[1,1].scatter(x_waypoints, y_waypoints, c='k', s=0.1, zorder=0)
    axes[1,1].set_xlim([min_x-40, max_x+40])
    axes[1,1].set_ylim([min_y-40, max_y+40])


    for i, agent in enumerate(agents):
        agent_data = data['processing_results'][timestep][agent]
        image = dataloader.read_images(agent, timestep)

        # Draw the yolo detection
        yolo_bounds = data['yolo_images'][agent][timestep]

        axes[0, i].set_title(f"Agent {agent} timestep {timestep}, \ncolor {colors[i]}")
        axes[0, i].imshow(image)

        for bounds in yolo_bounds:
            # -1 -> No detection
            if not -1 in bounds:
                # Draw the border
                x_min = bounds[0]
                x_max = bounds[1]
                y_min = bounds[2]
                y_max = bounds[3]
                height = y_max - y_min
                width = x_max - x_min

                # Rectangle(xy, width, height, ...
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                axes[0, i].add_patch(rect)

        # Rotation difference between CARLA (unit circle -> right = angle 0)
        # Matplotlib 0 angle = up. -90 to compensate.
        rotation = agent_data['direction'] - 90
        m = MarkerStyle(6)
        m._transform.rotate_deg(rotation)

        # Draw full sized version of map
        axes[1,0].scatter(
            agent_data['x'], agent_data['y'], c=colors[i], marker=m, s=100)

        # Draw scaled version of map
        axes[1,1].scatter(
            agent_data['x'], agent_data['y'], c=colors[i], marker=m, s=100)
        
        for detection in agent_data['detected']:
            axes[1,0].scatter(detection[0], detection[1], c=colors[i], edgecolors="black")
            axes[1,1].scatter(detection[0], detection[1], c=colors[i], edgecolors="black")
    
    mng = plt.get_current_fig_manager()
    mng.full_screen_toggle()
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    timestep += 30