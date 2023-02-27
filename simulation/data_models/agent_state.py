from dataclasses import dataclass


@dataclass
class AgentState:
    """
    Data class for storing agent GPS state in a 2D world
    """
    x: float
    y: float
    direction: float # Direction of the agent defined as angle
    velocity: float


@dataclass
class DetectedAgentState:
    """
    Data class for new 'agents' that are detected from node processing
    """
    id: str
    distance: float
    width_offset: float # Width offset