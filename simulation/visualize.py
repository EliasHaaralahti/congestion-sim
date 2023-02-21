import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.markers import MarkerStyle
from data import DataLoader

RESULTS_FILE = "result.json"
path = os.path.join('results', RESULTS_FILE)

with open(path) as json_file:
    data = json.load(json_file)

print(data.keys())

dataloader = DataLoader()
agents = dataloader.read_agent_ids()

# TODO: Use cmap or something to automatically pick colors.
colors = ['red', 'blue', 'yellow', 'black', 'orange', 'green']

plt.ion()
timestep = 0
while timestep < 2000:
    f, axes = plt.subplots(2, 2)
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
        axes[1,1].scatter(
            agent_data['x'], agent_data['y'], c=colors[i], 
            marker=m, s=100)
        axes[1,1].set_xlim([-100, 200])
        axes[1,1].set_ylim([-100, 200])
        
        for detection in agent_data['detected']:
            plt.scatter(detection[0], detection[1], c=colors[i], edgecolors="black")
    
    plt.show()
    plt.waitforbuttonpress()
    plt.close()
    timestep += 30