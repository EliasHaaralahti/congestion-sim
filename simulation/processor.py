import math
from typing import List, Tuple
from data_models.output_summary import OutputSummary, DetectionData
from data_models.agent_state import DetectedEntityState
from data_models.world import IntersectionStatus, World, Intersection
import numpy as np


class Processor():
    """
    Class that represent a single external processing unit in the simulated network.
    Each processor is responsible for processing the data of multiple vehicles and 
    RSUs. The processor will create the '3D world' and then analyze it, creating 
    the final object that will contain all data and analysis results for visualizing.
    """
    def __init__(self, env, data_pipe: dict, 
                 result_storage_pipe: list, dataloader):
        self.env = env
        self.data_pipe = data_pipe
        self.result_storage_pipe = result_storage_pipe
        self.dataloader = dataloader

        image_width, image_height = dataloader.get_image_dimensions()
        self.image_width = image_width
        self.image_height = image_height

        # Meters, how far detections are assumed to be existing agents.
        self.threshold_detection_radius = 2
        self.threshold_congestigation_speed = 12 # km/h
        # meters, how far away from intersection to be counted as part of intersection
        # Binary classification (congested or not) is performed based on average speed.
        self.threshold_within_intersection_range = 50
        

    def get_distance(self, agent: OutputSummary, target: Tuple[int, int]) -> float:
        """
        Calculate distance between agent and a target point (usually detection).
        Can be used in analysis to for example, calculate if potential for crash.
        """
        agent_position = (agent.agent_x, agent.agent_y)
        distance = math.dist(agent_position, target) # meters
        return distance

    def get_closest_intersection(self, agent: OutputSummary) -> object:
        """
        Returns the closest intersection distance with keys id and location.
        Location has keys x and y.
        """
        closest_intersection = None
        closest_distance = math.inf

        for intersection in self.dataloader.get_intersections():
            x = intersection['location']['x']
            y = intersection['location']['y']
            distance = self.get_distance(agent, (x,y))
            if distance < closest_distance:
                closest_intersection = intersection
                closest_distance = distance

        # Car is not close to any intersection currently.
        if closest_distance > self.threshold_within_intersection_range:
            return None
        
        return closest_intersection

    def process_detection(self, data: DetectionData) -> Tuple[float]:
        focal_length = 30
        height_in_frame = data.ymax - data.ymin
        image_width = self.image_width
        box_width_center = data.xmin + ((data.xmax - data.xmin) / 2)
        fov_angle = 90 # Camera fov should be 90.
        real_height = 0

        if data.type == "car":
            real_height = 1500
        elif data.type == "person":
            real_height = 1700

        # (mm * mm) / px. Distance is in cm!
        distance = (real_height * focal_length) / height_in_frame
        distance = distance / 100 # cm to m. In Carla one coordinate unit = 1m
        width_offset = (box_width_center/image_width * fov_angle) - (fov_angle / 2)
        id = f"{data.parent_id}-{data.detection_id}" 
        return DetectedEntityState(id, data.type, distance, width_offset)

    def process_detections(self, data: OutputSummary) -> dict:
        detections: List[DetectionData] = data.detections
        processed_detections = []
        for detection in detections:
            detection_result = self.process_detection(detection)
            processed_detections.append(detection_result)
        return processed_detections
    
    def is_detection_another_agent(self, current_agent_name, agent_names, target_position):
        for agent_name in agent_names:
            # Assume car does not detect itself.
            if agent_name == current_agent_name:
                continue

            agent_state: OutputSummary = self.data_pipe[agent_name]
            distance = self.get_distance(agent_state, target_position)
            if distance <= self.threshold_detection_radius:
                # Within threshold range, assume agent and detection are same.
                return False
        return True
    
    def is_collision_warning(self, agent_state: OutputSummary, 
                    target: Tuple[float, float]) -> Tuple[float, float]:
        distance = self.get_distance(agent_state, target) # meters

        # source for equation: https://www.ikorkort.nu/en/vk_korkortsfraga_en_23.php
        # Braking distance equation used: ((v/10)^2)/2,
        # where v is in km/h and result in meters.
        velocity_ms = agent_state.velocity # in m/s, convert to km/h
        velocity_kmh = velocity_ms * 3.6 # km/h
        braking_distance = ((velocity_kmh / 10) ** 2) / 2 # meters

        # Add one meter to braking distance to account for unknown target velocity
        # and give some time to brake. The simulation is not capable of detecting 
        # target velocity over steps.
        if distance < braking_distance + 1:
            return distance, True

        return distance, False

    def process_all(self, detections: dict) -> World:
        """
        Visualize the 'agents' found while processing the image date
        Color indicates which node was behind the detection.
        Nodes have a black circle around them

        :param detections: Processed detections, which indicate new agent positions
                            relative to the node position and direction
        :param node_data:  Data from the data_pipe, including EntityState inside a dict
                            for each agent. 
        Returns a list of processed nodes
        """
        processed_agents = {'agents': []}
        # Detections is a dict of agents with value list of detections.
        for agent_name in detections:
            state: OutputSummary = self.data_pipe[agent_name]
            node_direction = state.direction
            intersection = self.get_closest_intersection(state)
            
            # First add all the detections by the agent as new agent
            any_detection_crashing = False
            detection: DetectionData
            for detection in detections[agent_name]:
                #detected_agent_offset = detection.offset

                # Calculate new agent position based on distance and width offset.
                # CARLA seems to handle direction according to unit circle.
                # Therefore x=cos(angle), y=sin(angle)
                target_angle = node_direction + detection.width_offset
                target_x = state.agent_x + (detection.distance *
                                            np.cos(np.deg2rad(target_angle)))
                target_y = state.agent_y + (detection.distance *
                                            np.sin(np.deg2rad(target_angle)))
        
                distance_to_agent, crashing = self.is_collision_warning(state, (target_x, target_y))

                # Decide if detected target is a new target or an another agent.
                matches_agent = self.is_detection_another_agent(agent_name, detections, (target_x, target_y))
                # The processed agents must still be kept, as needed for bounding box drawing 
                # in visualization! Added to label to target matches_agent to allow filtering.
                if crashing:
                    any_detection_crashing = True
                # TODO: Combine these dicts to a new datatype. Or use existing.
                # Try to combine existing data types. Less is better.
                processed_agents['agents'].append({
                    'id': detection.id,
                    'x': target_x,
                    'y': target_y,
                    'type': detection.type,
                    'crashing': crashing,
                    'intersection': intersection,
                    'direction': 0,
                    'velocity': 0,
                    'detected': True, # Agent is detected. Properties not known.
                    'matches': matches_agent, # Does the detection match actual agent
                    'timestep': self.env.now
                })

                # Then add the original agent too.
                # NOTE: Using same intersection for node and detections. In many cases 
                # this may be not desired.
                agent_type = "rsu" if state.is_rsu else "vehicle"
                processed_agents['agents'].append({
                    'id': state.node_id,
                    'x': state.agent_x,
                    'y': state.agent_y,
                    'type': agent_type,
                    'crashing': any_detection_crashing,
                    'intersection': intersection,
                    'direction': node_direction,
                    'velocity': state.velocity,
                    'detected': False, # Is the agent detected or "known" from data.
                    'matches': False,
                    'timestep': self.env.now
                })
        return processed_agents

    def get_intersection_statuses(self, world: World) -> List[IntersectionStatus]:
        """
        Analyze intersection and get intersection status data object.
        """
        statuses: List[IntersectionStatus] = []

        # Initialize the dict with all intersections.
        intersections = self.dataloader.get_intersections()
        for i, intersection in enumerate(intersections):
            statuses.append({
                'id': intersection['id'],
                'car_count': 0,
                'human_count': 0,
                'rsu_count': 0,
                'speeds': [], # in m/s
                'status': "low",
                "timestep": self.env.now
            })

            # Loop each agent in the scene (cars, RSUs...). Note agents also
            # include detections, which have been converted to agents.
            for agent_data in world['agents']:
                agent_intersection: Intersection = agent_data['intersection']

                # Agent is not currently part of any intersection
                if agent_intersection is None:
                    continue
                    
                # Add agent if not rsu. Rsus do not add
                if agent_data['type'] == "rsu":
                    rsu_count = statuses[i]['rsu_count']
                    rsu_count += 1
                    statuses[i]['rsu_count'] = rsu_count
                elif agent_data['type'] == "vehicle":
                    vehicle_count = statuses[i]['car_count']
                    vehicle_count += 1
                    statuses[i]['car_count'] = vehicle_count
                    # add speed to speeds for congestion analysis.
                    velocity = agent_data['velocity']
                    statuses[i]['speeds'].append(velocity)
                else: # pedestrian
                    human_count = statuses[i]['human_count']
                    human_count += 1
                    statuses[i]['human_count'] = human_count


        # Revise the status.
        for intersection in statuses:
            car_count = intersection['car_count']

            # If car count is 0, the intersection status does not need to be updated.
            if car_count == 0:
                continue

            # Calculate average speed and convert m/s to km/h
            average_speed = (sum(intersection['speeds']) / len(intersection['speeds'])) * 3.6
            # Congestion is low by default. See if condition for high met.
            if average_speed < self.threshold_congestigation_speed:
                intersection['status'] = "congested"

        return statuses

    def analyze(self, world: World):
        """
        Perform analysis on the processed "world"
        """
        world['intersection_statuses'] = self.get_intersection_statuses(world)
        return world

    def run(self):
        """
        Dictionary to hold new detected 'agents' for each actual node.
        Dictionary key corresponds to node that detected the agent.
        """
        processed_detections = {}
        while True:
            # Convert all yolo detections to DetectedAgent. These are used 
            # later to find possible new detected vehicles and convert them 
            # into "agents".
            for agent in self.data_pipe:
                results = self.process_detections(self.data_pipe[agent])
                processed_detections[agent] = results

            # Processed_agents contains all agents and detections processed.
            # This is essentially the "3D" world. This object is what will be
            # used for final analysis and processing.
            processed_agents = self.process_all(processed_detections)

            # Most of the actual analysis happens here to understand the 3D world
            world: World = self.analyze(processed_agents)

            # Store the results, which will be saved under results/results.json
            self.result_storage_pipe['processing_results'][
                'agents'].extend(world['agents'])
            self.result_storage_pipe['processing_results'][
                'intersection_statuses'].extend(world['intersection_statuses'])

            # Process the simulation by one step for this processor.
            yield self.env.timeout(1)
        