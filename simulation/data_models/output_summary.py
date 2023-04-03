from dataclasses import dataclass


@dataclass
class DetectionData:
    """
    Data class for describing a bounding box detection
    produced by the model.
    """
    id: str # Identifies separate detectors at a single timestep.
    type: str # Class/type of detection
    xmin: float # Width of the bounding box
    xmax: float # Height of the bounding box
    ymin: float # box x coordinate in the image
    ymax: float # Box y coordinate in the image


@dataclass
class OutputSummary:
    """
    Data class for describing data sent by nodes to the processor
    """
    node_id: str
    is_rsu: bool
    agent_x: float # X position of agent in the world
    agent_y: float # Y position of agent in the world
    direction: float
    velocity: float
    detections: list[DetectionData] # Cars detected by yolo