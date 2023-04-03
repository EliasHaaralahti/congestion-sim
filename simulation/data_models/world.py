from dataclasses import dataclass

@dataclass
class CongestigationStatus:
    """
    Data class for describing congestigation status.
    Later this could be turned into an enum or something.
    """
    status: str # Values should be either low, medium, high


@dataclass
class Intersection:
    """
    Data class for defining a single intersection
    """
    id: str
    x: float
    y: float


@dataclass
class IntersectionStatus:
    """
    Describe a intersection
    """
    id: str
    car_count: int
    human_count: int
    rsu_count: int
    speeds: list[float]
    status: CongestigationStatus


@dataclass
class World:
    """
    Data for describing the final 3D world, which will be visualized.
    """
    agents: list
    intersection_statuses: dict[str, Intersection]
