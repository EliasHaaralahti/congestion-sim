import math
from typing import List, Tuple
from data_models.output_summary import OutputSummary, DetectionData
from data_models.agent_state import DetectedAgentState
import numpy as np


def check_if_crashing(
        agent: OutputSummary, target: Tuple[int, int]) -> bool:
    """ TODO
    Function for checking if the agent is about to crash with other other 
    agents or detected. Or just detected?
    """
    # Basic logic for now: If agent detected, is close enough and velocity > 0
    # -> crashing.

    agent_position = (agent.agent_x, agent.agent_y)
    distance = math.dist(agent_position, target) # meters

    crashing = False
    if distance < 2:
        crashing = True
    return crashing

def process_detection(data: DetectionData) -> Tuple[float]:
    focal_length = 30
    height_in_frame = data.ymax - data.ymin
    # TEMPORARY: Whole picture is 640x640. Get this data from somewhere.
    image_width = 640
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
    return DetectedAgentState(data.id, distance, width_offset)


def process_agent_data(data: OutputSummary) -> dict:
    detections: List[DetectionData] = data.detections

    # Process detections
    processed_detections = []
    for detection in detections:
        result = process_detection(detection)
        processed_detections.append(result)
    return processed_detections


def process(detections: dict, data_pipe: dict, result_storage_pipe: list):
    """
    Visualize the 'agents' found while processing the image date
    Color indicates which node was behind the detection.
    Nodes have a black circle around them

    :param detections: Processed detections, which indicate new agent positions
                        relative to the node position and direction
    :param node_data:  Data from the data_pipe, including AgentState inside a dict
                        for each agent. 
    """
    # Dict of agents with fields x,y,direction and detected,
    # which is array of arrays [[x,y]]
    processed_agents = {}
    for agent in detections:
        state: OutputSummary = data_pipe[agent]
        node_direction = state.direction
        processed_agents[agent] = {
            'x': state.agent_x,
            'y': state.agent_y,
            'direction': node_direction,
            'velocity': state.velocity,
            'detected': []
        }

        for detection in detections[agent]:
            #detected_agent_offset = detection.offset

            # Calculate new agent position based on distance and width offset.
            # TODO: width offset not implemented. Distance not used
            # CARLA seems to handle direction according to unit circle.
            # Therefore x=cos(angle), y=sin(angle)
            target_angle = node_direction + detection.width_offset
            #print(f"Node direction: {node_direction}, target angle: {detection.width_offset}")
            target_x = state.agent_x + (detection.distance *
                                        np.cos(np.deg2rad(target_angle)))
            target_y = state.agent_y + (detection.distance *
                                        np.sin(np.deg2rad(target_angle)))
    
            crashing = check_if_crashing(state, (target_x, target_y))
            target = [target_x, target_y, crashing, detection.id, detection.distance]
            processed_agents[agent]['detected'].append( target )

    result_storage_pipe['processing_results'].append(processed_agents)


def processor(env, data_pipe: dict, result_storage_pipe: list):
    """
    Dictionary to hold new detected 'agents' for each actual node.
    Dictionary key corresponds to node that detected the agent.
    """
    detected_agents = {}
    while True:
        for agent in data_pipe:
            results = process_agent_data(data_pipe[agent])
            detected_agents[agent] = results

        process(detected_agents, data_pipe, result_storage_pipe)
        yield env.timeout(1)
        