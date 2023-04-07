from dataclasses import dataclass
import json


@dataclass
class DetectionData:
    """
    Data class for describing a bounding box detection
    produced by the model.
    """
    # Combination of parent and detection id at a single timestep 
    # create a unique id for searching.
    parent_id: str # Who detected the detection
    detection_id: str # Identifies detections at a single timestep.
    type: str # Class/type of detection
    xmin: float # Width of the bounding box
    xmax: float # Height of the bounding box
    ymin: float # box x coordinate in the image
    ymax: float # Box y coordinate in the image

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=1)


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