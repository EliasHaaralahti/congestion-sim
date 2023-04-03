from matplotlib.path import Path
from matplotlib.pyplot import Axes
from typing import List, Tuple
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from data_models.world import World


COLOR_MAP = plt.get_cmap('viridis')


def draw_information_view(ax: Axes, agent_count, data: World, timestep):
    timestep_data = data['processing_results'][timestep]

    ax.set_title("Intersections")

    # Loop over intersections
    for id, intersection in timestep_data['intersection_statuses'].items():
        status = intersection['status']
        car_count = intersection['car_count']
        human_count = intersection['human_count']
        rsu_count = intersection['rsu_count']

        ax.text(0.01, 0.93, f"Agents in intersection {id}: {agent_count}", 
                size=15, color='black')
        ax.text(0.01, 0.85, 
                f"Total cars (agents + detections): {car_count}", 
                size=15, color='black')
        ax.text(0.01, 0.77, f"Total humans: {human_count}", 
                size=15, color='black')
        ax.text(0.01, 0.69, f"RSUs in intersection: {rsu_count}", 
                size=15, color='black')
        ax.text(0.01, 0.61-0.08, f"Intersection status: {status}", 
                size=15, color='black')
        
        break # TODO Temp loop only once, later need to find places for labels...


def draw_car_views(axes: Axes, agents, data, timestep, dataloader):
    for i, agent in enumerate(agents):
        # limit to two first results for now TODO
        if i == 2:
            break

        agent_data = data['processing_results'][timestep]['agents'][agent]
        image = dataloader.read_images(agent, timestep)
        # Draw the yolo detection
        yolo_bounds = data['yolo_images'][agent][timestep]
        velocity = agent_data['velocity']
        total_agents = len(agents)
        colors = [COLOR_MAP(1.*i/total_agents) for i in range(total_agents)]
        axes[0, i].set_title(
            f"Agent ID: {agent}, Color: {colors[i]}, Velocity: {velocity:.1f}m/s")
        axes[0, i].imshow(image)

        for bounds in yolo_bounds:
            # -1 -> No detection
            if -1 in bounds:
                continue
                
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
            print("HMM")
            print(agent_data['detected'])
            print("with id: " + str(id))

            # Find the detected agent state, such as distance
            matching_detection = None
            print("All detections: ")
            print(agent_data['detected'])
            
            """
            for detection in agent_data['detected']:
                # id is in form <number>-camera_<number>. Match only camera_<number>.
                det_id = detection[4]
                actual_det_id = det_id.partition("-")[2]
                actual_id = id.partition("-")[2]
                print(f"Comparing ids {actual_id} and {actual_det_id}")
                if actual_det_id == actual_id:
                    matching_detection = detection
            """

            matching_detection = [
                i for i in agent_data['detected'] if i[4]==id
                ][0]

            text = f"C: {type}, D: {matching_detection[5]:.1f}m"
            # if crashing
            if matching_detection[2]: 
                text += " - COLLISION WARNING"
            axes[0, i].text(
                x_min, y_min-10, text, size=13, color='white',
                path_effects=[pe.withStroke(linewidth=2, foreground="black")])


def draw_map(ax: Axes, waypoints: Tuple, agents, data, timestep):
    """
    Draw the top-view map visualization and markers for cars and humans.
    """
    x_waypoints, y_waypoints = waypoints
    human_marker, car_marker, rsu_marker = get_human_marker(), get_car_marker(), get_rsu_marker()
    min_x, max_x, min_y, max_y = get_relevant_coordinates(
            agents, timestep, data['processing_results'])

    ax.set_title("2D scene map")
    ax.scatter(x_waypoints, y_waypoints, c='k', s=1, zorder=0)
    ax.set_xlim([min_x-60, max_x+60])
    ax.set_ylim([min_y-60, max_y+60])

    total_agents = len(agents)
    for i, agent in enumerate(agents):
        agent_data = data['processing_results'][timestep]['agents'][agent]
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
        
        # + rsus are rotated + 90, as the marker points down by default
        rsu_marker_2 = rsu_marker.transformed(
            mpl.transforms.Affine2D().rotate_deg(rotation_adjusted + 90))

        # Draw markers on the map
        colors = [COLOR_MAP(1.*i/total_agents) for i in range(total_agents)]

        if agent_data['is_rsu']: # Agent is RSU
            ax.scatter(
                agent_data['x'], agent_data['y'], color=colors[i], marker=rsu_marker_2, s=500)
        else: # Agent is vehicle
            ax.scatter(
                agent_data['x'], agent_data['y'], color=colors[i], marker=agent_marker, s=100)
        
        for detection in agent_data['detected']:
            # If matches, the car matches to an existing agent.
            x_det, y_det, crashing_det, distance_to_agent, id, distance, type, matches = detection
            color = "red" if crashing_det else "black"

            if type == "car": # car
                ax.scatter(x_det, y_det, color=colors[i], edgecolors=color)
            else: # human
                ax.scatter(x_det, y_det, marker=human_marker,
                                color=colors[i], edgecolors=color, s=200)
            
    # Rotate Y axis since CARLA Y axis is "upside down".
    ax.set_ylim(ax.get_ylim()[::-1])


def get_relevant_coordinates(agents: List[str], step: int, processed: object):
    """
    Find min and max values for both x and y coordinates for each agent
    and detected entity. Used for scaling the visualization.
    """

    # TODO: What is this. Clean and use actual sim length.
    #   -> This was a trick to instead get min and max coordinates over
    #       all steps. Constant jumping in a video looks bad. Need to decide
    #       how this should be in the future.
    # TEMP: ignore step. Check all timesteps
    sim_length = 700 - 1

    x_coords = []
    y_coords = []
    for sim_step in range(sim_length):
        coordinates = []
        for a in agents:
            a_state = processed[sim_step]['agents'][a]
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