import os
import json
from typing import List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as pe
from matplotlib.markers import MarkerStyle
from data import DataLoader


def calculate_angle_y_flip(rotation: float) -> float:
    """
    Flipping Matplotlib y-axis is required due to CARLA handling it differently.
    Need to also flip markers with orientation, but only on y axis direction.
    This returns the matching angle, but x axis direction remains untouched.
    Angles are considered similar to the unit circle, where 0 degrees = right.
    """
    # Convert any angle to range 0-360
    angle = rotation % 360
    if 0 <= angle < 90:
        return angle - (angle * 2)
    elif 90 <= angle <= 180:
        return angle + ((180 - angle) * 2)
    elif 180 < angle <= 270:
        return angle - ((angle - 180) * 2)
    elif 270 < angle <= 360:
        return angle + ((360 - angle) * 2)
    else:
        return -1


def get_relevant_coordinates(agents: List[str], step: int, processed: object):
    """
    Find min and max values for both x and y coordinates for each agent
    and detected entity. Used for scaling the visualization.
    """

    # TEMP: ignore step. Check all timesteps
    sim_length = 700 - 1

    x_coords = []
    y_coords = []
    for sim_step in range(sim_length):
        coordinates = []
        for a in agents:
            a_state = processed[sim_step][a]
            coordinates.append( (a_state['x'], a_state['y']) )
            for d in a_state['detected']:
                coordinates.append( (d[0], d[1]) )

        x, y = zip(*coordinates)
        x_coords.append(min(x))
        x_coords.append(max(x))
        y_coords.append(min(y))
        y_coords.append(max(y))

    return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))


RESULTS_FILE = "results.json"
path = os.path.join('results', RESULTS_FILE)

with open(path) as json_file:
    data = json.load(json_file)

dataloader = DataLoader()
metadata_summary = dataloader.get_metadata_summary()
agents = dataloader.read_agent_ids()
x_waypoints, y_waypoints = zip(*dataloader.get_map()) # List of (x,y).
x_waypoint_min, x_waypoint_max = min(x_waypoints), max(x_waypoints)
y_waypoint_min, y_waypoint_max = min(x_waypoints), max(x_waypoints)

# TODO: Use cmap or something to automatically pick colors.
colors = ['blue', 'red', 'yellow', 'black', 'orange', 'green']

def render_visualization(interactive=False, skip_timesteps=0):
    if interactive:
        plt.ion()
    else:
        # Ensure figures folder is created if interactive = False
        folder = "figures"
        if not os.path.exists(folder):
            os.makedirs(folder)


    timestep = 0
    max_timesteps = dataloader.get_simulation_length()
    while timestep <= max_timesteps - 1:
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle(f"Simulation timestep {timestep}")
        min_x, max_x, min_y, max_y = get_relevant_coordinates(
            agents, timestep, data['processing_results'])

        # Show metadata information in this window.
        axes[1, 0].set_title("Scene information")
        for i, field in enumerate(metadata_summary):
            y_position = 1 - 0.1 - (i * 0.1)
            field_upp = field[0].upper() + field[1:]
            text = f"{field_upp}: {metadata_summary[field]}"
            axes[1, 0].text(0.01, y_position, text, size=15, color='black')
        
        # Zoomed window for displaying map
        axes[1, 1].set_title("2D scene map")
        axes[1, 1].scatter(x_waypoints, y_waypoints, c='k', s=0.1, zorder=0)
        axes[1, 1].set_xlim([min_x-60, max_x+60])
        axes[1, 1].set_ylim([min_y-60, max_y+60])

        for i, agent in enumerate(agents):
            agent_data = data['processing_results'][timestep][agent]

            image = dataloader.read_images(agent, timestep)

            # Draw the yolo detection
            yolo_bounds = data['yolo_images'][agent][timestep]

            velocity = agent_data['velocity']
            axes[0, i].set_title(
                f"Agent ID: {agent}, Color: {colors[i]}, Velocity: {velocity:.1f}m/s")
            axes[0, i].imshow(image)

            for bounds in yolo_bounds:
                # -1 -> No detection
                if not -1 in bounds:
                    # Draw the border
                    id = bounds[0]
                    type = bounds[1]
                    x_min = bounds[2]
                    x_max = bounds[3]
                    y_min = bounds[4]
                    y_max = bounds[5]
                    height = y_max - y_min
                    width = x_max - x_min

                    # Rectangle(xy, width, height, ...
                    rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, 
                                            edgecolor='r', facecolor='none')
                    axes[0, i].add_patch(rect)

                    # Find distance to current agent using bounds id and detection id
                    matching_detection = [
                        i for i in agent_data['detected'] if i[3]==id
                        ][0]
                    
                    text = f"C: {type}, D: {matching_detection[4]:.1f}m"
                    # if crashing
                    if matching_detection[2]: 
                        text += " - PROXIMITY WARNING"
                    axes[0, i].text(
                        x_min, y_min-10, text, size=13, color='white',
                        path_effects=[pe.withStroke(linewidth=2, foreground="black")])

            # Rotation difference between CARLA (unit circle -> right = angle 0)
            # Matplotlib 0 angle = up. -90 to compensate. Additionally since the
            # y-axis is flipped, need to account for that too with calculate_angle_y_flip.
            rotation_adjusted = calculate_angle_y_flip(agent_data['direction']) - 90
            m = MarkerStyle(6)
            m._transform.rotate_deg(rotation_adjusted)

            # Draw scaled version of map
            axes[1,1].scatter(
                agent_data['x'], agent_data['y'], c=colors[i], marker=m, s=50)
            
            for detection in agent_data['detected']:
                x_det, y_det, crashing_det, id, distance = detection
                color = "red" if crashing_det else "black"
                axes[1,1].scatter(x_det, y_det, c=colors[i], edgecolors=color)

        # Rotate Y axis since CARLA Y axis is "upside down".
        axes[1,1].set_ylim(axes[1,1].get_ylim()[::-1])
        
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
render_visualization(interactive=False, skip_timesteps=0)
