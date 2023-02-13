from model import Model
from data_models.output_summary import OutputSummary, DetectionData
from data_models.agent_state import AgentState
from data import DataLoader
from typing import Tuple
from numpy import ndarray

import time


class Node():
    """
    Class that represents a single simulation node in the simulated network of agents.
    Each node has the responsibility of processing the data of a single agent.
    The processed data will be then delivered to an external processing entity.
    """
    # Initialize with sympy environment
    def __init__(self, env: object, node_id: str, dataloader: DataLoader,
                model: Model, data_pipe: dict):
        self.env: object = env
        self.node_id: str = node_id
        self.dataloader: DataLoader = dataloader
        self.model: Model = model
        # For now store model output for later processing here
        self.data_pipe = data_pipe

    def read_data(self) -> Tuple[AgentState, ndarray] :
        """
        Read data for current node at simulation tick
        """
        print(f"Reading data for node {self.node_id}")
        agent_state = self.dataloader.read_agent_state(self.node_id, self.env.now)
        camera_image = self.dataloader.read_images(self.node_id, self.env.now)
        return agent_state, camera_image

    def summarize_output(self, raw_output: object, state: AgentState) -> OutputSummary:
        """ TODO
        Convert raw model results into OutputSummary data class.
        Only accept results with class 2 (car)
        """
        # Remove the .pandas() to get the data as a tensor. Might boost performance
        # Pandas offers columns xmin, ymin, xmax, ymax, confidence, class, name
        pandas_yolo_results = raw_output.pandas().xyxy[0]

        # Convert yolo detections to OutputSummary
        # and exclude results for classes other than car.
        # Note, for optimization instead of using iterrows
        # alternative panda methods could possibly be used.
        detections = []        
        for _, row in pandas_yolo_results.iterrows():
            if row['name'] == 'car':
                detection = DetectionData(
                    xmin=row['xmin'],
                    xmax=row['xmax'],
                    ymin=row['ymin'],
                    ymax=row['ymax']
                )
                detections.append(detection)

        output = OutputSummary(
            node_id=self.node_id,
            agent_x=state.x,
            agent_y=state.y,
            direction=state.direction,
            detections=detections
        )
        return output

    def run(self):
        """
        Generator for nodes where at each simulation tick data from
        camera is processed and sent to a centralized computer.
        """
        while True:
            state, image = self.read_data()
            raw_output = self.model.forward(image)
            # TODO: Communicate output to other nodes/central computer?
            output = self.summarize_output(raw_output, state)
            self.data_pipe[self.node_id] = output
            yield self.env.timeout(1)
            