from dataclasses import dataclass


@dataclass
class DetectionData:
    """
    Data class for describing a bounding box detection
    produced by the model.
    """
    type: str # Class/type of detection
    xmin: float # Width of the bounding box
    xmax: float # Height of the bounding box
    ymin: float # box x coordinate in the image
    ymax: float # Box y coordinate in the image


@dataclass
class OutputSummary:
    """
    Data class describing the node specific output of the yolo
    model in a more summarized way for final processing.
    """
    node_id: str
    agent_x: float # X position of agent in the world
    agent_y: float # Y position of agent in the world
    direction: float # TODO: Is this direction needed if AgentState also has direction?
    detections: list[DetectionData] # Cars detected by yolo