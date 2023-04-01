from matplotlib.path import Path
from matplotlib.pyplot import Axes
from typing import List, Tuple
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.patches as patches


# TODO: Use cmap or something to automatically pick colors?
colors = ['blue', 'red', 'yellow', 'black', 'orange', 'green']


def draw_information_view(ax: Axes, agent_count, data, timestep):
    status = data['processing_results'][timestep]['intersection_status']
    car_detection_count = status['car_detection_count']
    human_detection_count = status['human_detection_count']
    total_cars = status['total_cars']
    intersection_status = status['status']

    ax.set_title("Scene information")
    ax.text(0.01, 0.93, f"Agents in intersection (from metadata): {agent_count}", 
            size=15, color='black')
    ax.text(0.01, 0.85, 
            f"Total cars (agents + detections): {agent_count + car_detection_count}", 
            size=15, color='black')
    ax.text(0.01, 0.77, f"Total humans: {human_detection_count}", 
            size=15, color='black')
    ax.text(0.01, 0.69, f"Total cars in intersection: {total_cars}", 
            size=15, color='black')
    ax.text(0.01, 0.61-0.08, f"Intersection status: {intersection_status}", 
            size=15, color='black')


def draw_car_views(axes: Axes, agents, data, timestep, dataloader):
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


def draw_map(ax: Axes, waypoints: Tuple, agents, data, timestep):
    """
    Draw the top-view map visualization and markers for cars and humans.
    """
    x_waypoints, y_waypoints = waypoints
    human_marker, car_marker = get_human_marker(), get_car_marker()
    min_x, max_x, min_y, max_y = get_relevant_coordinates(
            agents, timestep, data['processing_results'])

    ax.set_title("2D scene map")
    ax.scatter(x_waypoints, y_waypoints, c='k', s=5, zorder=0)
    ax.set_xlim([min_x-60, max_x+60])
    ax.set_ylim([min_y-60, max_y+60])

    for i, agent in enumerate(agents):
        agent_data = data['processing_results'][timestep][agent]
        # Rotation difference between CARLA (unit circle -> right = angle 0)
        # Matplotlib 0 angle = up. -90 to compensate. Additionally since the
        # y-axis is flipped, need to account for that too with calculate_angle_y_flip.
        # TODO: The current marker is set to be at default 0 degrees = right. No need to compensate.
        #rotation_adjusted = calculate_angle_y_flip(agent_data['direction']) - 90
        rotation_adjusted = calculate_angle_y_flip(agent_data['direction'])
        #m = MarkerStyle(6)
        #m._transform.rotate_deg(rotation_adjusted)
        agent_marker = car_marker.transformed(
            mpl.transforms.Affine2D().rotate_deg(rotation_adjusted))

        # Draw markers on the map
        ax.scatter(
            agent_data['x'], agent_data['y'], c=colors[i], marker=agent_marker, s=100)
        
        for detection in agent_data['detected']:
            x_det, y_det, crashing_det, id, distance, type = detection
            color = "red" if crashing_det else "black"

            if type == "car": # car
                ax.scatter(x_det, y_det, c=colors[i], edgecolors=color)
            else: # human
                ax.scatter(x_det, y_det, marker=human_marker,
                                c=colors[i], edgecolors=color, s=200)
            
    # Rotate Y axis since CARLA Y axis is "upside down".
    ax.set_ylim(ax.get_ylim()[::-1])


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