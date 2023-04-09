import math
from matplotlib.path import Path
from matplotlib.pyplot import Axes
from typing import List, Tuple
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from data_models.world import World
from data import DataLoader


COLOR_MAP = plt.get_cmap('viridis')


def draw_information_view(ax: Axes, agent_count, data: World, timestep: int, 
                          dataloader: DataLoader):
    intersections = [
        d for d in data['intersection_statuses'] if d['timestep'] == timestep]

    ax.set_title("Analysis results")

    # Variables to keep track where text is placed currently
    text_x, text_y = 0.01, 0.93
    text_y_decrease = 0.08
    text_color = "black"

    # Print ground truth information
    metadata = dataloader.get_metadata_summary()
    total_cars = metadata['n_vehicles']
    total_pedestrians = metadata['n_pedestrians']
    total_rsus = metadata['n_rsus']
    ax.text(text_x, text_y, "Ground truth:", size=13, color=text_color)
    text_y -= text_y_decrease
    ax.text(text_x, text_y, f"Total cars {total_cars}:", size=10, color=text_color)
    text_y -= text_y_decrease
    ax.text(text_x, text_y, f"Total pedestrians: {total_pedestrians}", size=10, color=text_color)
    text_y -= text_y_decrease
    ax.text(text_x, text_y, f"Total RSUs: {total_rsus}", size=10, color=text_color)
    text_y -= text_y_decrease

    text_x += 0.4
    text_y = 0.93

    # Loop over intersections
    for intersection in intersections:
        intersection_id = intersection['id']
        status = intersection['status']
        car_count = intersection['car_count']
        human_count = intersection['human_count']
        speeds = intersection['speeds']
        # TODO: Same calculation performed in processor.py
        # just store result there?
        if len(speeds) == 0:
            average_speed = 0
        else:
            average_speed = (sum(speeds) / len(speeds)) * 3.6 # km/h
        
        # ax.text(x,y) where 0,0 is 0,1 is top left corner.
        intersection_id_number = intersection_id.split('_')[1]
        ax.text(text_x, text_y, f"Intersection {intersection_id_number} result:", 
                size=13, color=text_color)
        ax.text(text_x, text_y - text_y_decrease, f"Total cars: {car_count}", 
                size=10, color=text_color)
        ax.text(text_x, text_y - (text_y_decrease * 2), 
                f"Total humans: {human_count}", 
                size=10, color=text_color)
        ax.text(text_x, text_y - (text_y_decrease * 3), 
                f"Average speed: {round(average_speed, 1)} km/h", 
                size=10, color=text_color)
        ax.text(text_x, text_y - (text_y_decrease * 4),
                f"Congestion status: {status}", 
                size=10, color=text_color)

        text_y -= (text_y_decrease * 6)


def draw_car_views(car_indexes, ax_names, axes: Axes, agent_names, 
                   data_results, data_yolo, timestep, dataloader):
    
    for i, car_index in enumerate(car_indexes):
        # List to keep count of label y coordinates.
        # This is an attempt to have less overlapping of labels.
        labels_y_coords = []

        agent_name = agent_names[car_index]
    
        all_agents_timestep = [
            d for d in data_results['agents'] if d['timestep'] == timestep]
        #all_agent_ids = [
        #    d['id'] for d in data_results['agents'] if d['timestep'] == timestep]
        agent_data = next((d for d in all_agents_timestep if d['id'] == 
                      agent_name and d.get('timestep') == timestep), None)
        images = dataloader.read_images(agent_name, timestep)
        # TODO: Error: Sometimes the agent data seems to disappear??
        # thus velocity is not available only in this part. The other parts 
        # work. This is a silent fix.
        # Later noted: This may be for RSUs only, in which case velocity=0
        if agent_data is None:
            velocity = -1
        else:
            velocity = agent_data['velocity']
        total_agents = len(agent_names)
        colors = [COLOR_MAP(1.*i/total_agents) for i in range(total_agents)]
        axes[ax_names[i]].set_title(
            f"Car ID: {agent_name}\nColor: {colors[i]}\nVelocity: {velocity:.1f}m/s")
        axes[ax_names[i]].imshow(images)

        # Draw the yolo detections
        yolo_bounds = [
            d for d in data_yolo if d['parent_id'] == agent_name and
            d['timestep'] == timestep]

        for bound in yolo_bounds:                
            # Draw the border
            detection_id = bound["detection_id"]
            parent_id = bound['parent_id']
            type = bound["type"]
            x_min = bound["xmin"]
            x_max = bound["xmax"]
            y_min = bound["ymin"]
            y_max = bound["ymax"]
            height = y_max - y_min
            width = x_max - x_min

            # Rectangle(xy, width, height, ...
            rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1,
                                    edgecolor='r', facecolor='none')
            axes[ax_names[i]].add_patch(rect)

            # Find the parent of the detection (ie the agent itself)
            detection_agent = None          
            for agent in all_agents_timestep:
                # Only process detected agents.
                if not agent['detected']:
                    continue

                det_id = agent['id']
                detection_parent_id = f"{parent_id}-{detection_id}"
                if det_id == detection_parent_id:
                    detection_agent = agent
                    break

            parent_position = (detection_agent['x'], 
                               detection_agent['y'])
            agent_position = (agent_data['x'], agent_data['y'])
            distance = math.dist(agent_position, parent_position) # meters
            text = f"C: {type}, D: {distance:.1f}m"
            # if crashing
            if detection_agent['crashing']: 
                text += " - COLLISION WARNING"

            label_y = y_min - 10
            # If label with y coordinate close enough to current label_y
            tolerance = 20
            while True:
                if any(abs(x - label_y) <= tolerance for x in labels_y_coords):
                    label_y -= 40
                    if label_y <= 0:
                        label_y = 0
                        break
                else:
                    break
            axes[ax_names[i]].text(
                x_min, label_y, text, size=12, color='white',
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])
            
            labels_y_coords.append(label_y)

def draw_map(ax: Axes, waypoints: List[Tuple], agents, data, timestep):
    """
    Draw the top-view map visualization and markers for cars and humans.
    """
    x_waypoints, y_waypoints = waypoints
    human_marker, car_marker, rsu_marker = get_human_marker(), get_car_marker(), get_rsu_marker()
    min_x, max_x, min_y, max_y = get_relevant_coordinates(data)

    ax.set_title("2D scene map")
    ax.scatter(x_waypoints, y_waypoints, c='k', s=1, zorder=0)
    ax.set_xlim([min_x-60, max_x+60])
    ax.set_ylim([min_y-60, max_y+60])

    timestep_agents = [
        d for d in data['agents'] if d['timestep'] == timestep]

    total_agents = len(timestep_agents)
    for i, agent in enumerate(timestep_agents):
        # Rotation difference between CARLA (unit circle -> right = angle 0)
        # Matplotlib 0 angle = up. -90 to compensate. Additionally since the
        # y-axis is flipped, need to account for that too with calculate_angle_y_flip.
        rotation_adjusted = calculate_angle_y_flip(agent['direction'])
        agent_marker = car_marker.transformed(
            mpl.transforms.Affine2D().rotate_deg(rotation_adjusted))
        
        # + rsus are rotated + 90, as the marker points down by default
        rsu_marker_2 = rsu_marker.transformed(
            mpl.transforms.Affine2D().rotate_deg(rotation_adjusted + 90))

        # Draw markers on the map
        colors = [COLOR_MAP(1.*i/total_agents) for i in range(total_agents)]

        if agent["type"] == 'is_rsu': # Agent is RSU
            ax.scatter(
                agent['x'], agent['y'], color=colors[i], marker=rsu_marker_2, s=500)
        if agent["type"] == 'person': # Agent is pedestrian
             ax.scatter(agent['x'], agent['y'], marker=human_marker,
                                color=colors[i], edgecolors="red", s=200)
        else: # Agent is vehicle
            if agent['detected']: # The vehicle was detected, draw circle.
                ax.scatter(agent['x'], agent['y'], color=colors[i], 
                           edgecolors="red")
            else: # Agent is "known".
                ax.scatter(agent['x'], agent['y'], color=colors[i], 
                           marker=agent_marker, s=100)
    
    # Rotate Y axis since CARLA Y axis is "upside down".
    ax.set_ylim(ax.get_ylim()[::-1])


def get_relevant_coordinates(processed: object):
    """
    Find min and max values for both x and y coordinates for each agent
    and detected entity. Used for scaling the visualization.
    """
    agent_states = processed['agents']
    x_coords = []
    y_coords = []
    for agent in agent_states: 
        x_coords.append(agent['x'])
        y_coords.append(agent['y'])

    return (min(x_coords), max(x_coords), min(y_coords), max(y_coords))


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
    

def get_car_marker():
    # Car center is about 0,0 to ensure markers are placed 
    # properly at center of x,y.
    car_verts = [
        # x,y (0,0) = left bottom
        (-0.75, -0.5), # driver door corner.
        (-0.75, 0.5), # forward-right corner
        (0.5, 0.5), # back-right corner 
        (1, 0.2), # bumper1
        (1, -0.2), #bumper2
        (0.5, -0.5) # back-left corner.
    ]

    car_codes = [Path.MOVETO]
    for _ in range(len(car_verts) - 1):
        car_codes.append(Path.LINETO)
    car_marker = Path(car_verts, car_codes)
    return car_marker


def get_rsu_marker():
    # RSU center is approx around 0,0 to ensure markers are placed
    # properly at center of x,y 
    rsu_verts = [
        # x,y (0,0) = left bottom
        (0, 0.5),
        (-0.25, 0.5),
        (-0.25, 1),
        (0.25, 1),
        (0.25, 0.5),
        (0, 0.5),
        (0, -0.25),
    ]
    rsu_codes = [Path.MOVETO]
    for _ in range(len(rsu_verts) - 1):
        rsu_codes.append(Path.LINETO)
    rsu_marker = Path(rsu_verts, rsu_codes)
    return rsu_marker


def get_human_marker():
    # Human center is around 0,0 to ensure markers are placed
    # properly at center of x,y 
    human_verts = [
        # x,y (0,0) = left bottom

        # start at the bottom of the head center, draw head
        (0, 0.5),
        (-0.25, 0.5),
        (-0.25, 1),
        (0.25, 1),
        (0.25, 0.5),
        (0, 0.5),

        # left arm
        (-0.25, 0),
        (0, 0.5),

        # right arm
        (0.25, 0),
        (0, 0.5),

        # spine
        (0, -0.25),

        # left leg
        (-0.25, -1),
        (0, -0.25),

        # right leg
        (0.25, -1),
        (0, -0.25),
    ]
    human_codes = [Path.MOVETO]
    for _ in range(len(human_verts) - 1):
        human_codes.append(Path.LINETO)
    human_marker = Path(human_verts, human_codes)
    return human_marker