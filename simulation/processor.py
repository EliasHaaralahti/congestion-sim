from typing import List, Tuple
from data_models.output_summary import OutputSummary, DetectionData
from data_models.agent_state import DetectedAgentState
import numpy as np


def process_detection(data: DetectionData) -> Tuple[float]:
    focal_length = 50
    height_in_frame = data.ymax - data.ymin
    real_height = 0
    if data.type == "car":
        real_height = 1500
    elif data.type == "person":
        real_height = 1700
    
    # (mm * mm) / px. Distance is in cm!
    distance = (real_height * focal_length) / height_in_frame
    distance = distance / 100 # cm to m. In Carla one coordinate unit = 1m
    width_offset = 0
    return DetectedAgentState(distance, width_offset)


def process_agent_data(data: OutputSummary) -> dict:
    #print(f"Agent {data.node_id} data {data}")
    detections: List[DetectionData] = data.detections

    # Process detections into new 'agents'
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
        agentState: OutputSummary = data_pipe[agent]
        node_x = agentState.agent_x
        node_y = agentState.agent_y
        node_direction = agentState.direction
        processed_agents[agent] = {
            'x': node_x,
            'y': node_y,
            'direction': node_direction,
            'detected': []
        }

        for detection in detections[agent]:

            detected_agent_distance = detection.distance
            detected_agent_offset = detection.offset

            # Calculate new agent position based on distance and width offset.
            # TODO: width offset not implemented. Distance not used
            # CARLA seems to handle direction according to unit circle.
            # Therefore x=cos(angle), y=sin(angle)
            target_x = node_x + (detected_agent_distance * np.cos(np.deg2rad(node_direction)))
            target_y = node_y + (detected_agent_distance * np.sin(np.deg2rad(node_direction)))
    
            processed_agents[agent]['detected'].append( [target_x, target_y] )

    result_storage_pipe['processing_results'].append(processed_agents)


def processor(env, data_pipe: dict, result_storage_pipe: list):
    # Dictionary to hold new detected 'agents' for each actual node.
    # Dictionary key corresponds to node that detected the agent.
    detected_agents = {}
    while True:
        for agent in data_pipe:
            results = process_agent_data(data_pipe[agent])
            detected_agents[agent] = results

        process(detected_agents, data_pipe, result_storage_pipe)
        yield env.timeout(1)
        