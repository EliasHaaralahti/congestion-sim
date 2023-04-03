from dataclasses import dataclass


@dataclass
class EntityState:
    """
    Data class for storing agent GPS state in a 2D world
    """
    is_rsu: bool # is the entity a RSU or a vehicle.
    x: float
    y: float
    direction: float # Direction of the agent defined as angle
    velocity: float


@dataclass
class DetectedEntityState:
    """
    Data class for new 'agents' that are detected from node processing
    """
    id: str
    type: str
    distance: float
    width_offset: float # Width offset