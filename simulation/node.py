from typing import Tuple
from model import Model
from data_models.output_summary import OutputSummary, DetectionData
from data_models.agent_state import EntityState
from data import DataLoader
from numpy import ndarray


class Node():
    """
    Class that represents a single simulation node in the simulated network of agents.
    Each node has the responsibility of processing the data of a single agent.
    The processed data will be then delivered to an external processing entity.
    """
    def __init__(self, env: object, node_id: str, dataloader: DataLoader,
                model: Model, data_pipe: dict, result_storage_pipe: dict):
        self.env: object = env
        self.node_id: str = node_id
        self.dataloader: DataLoader = dataloader
        self.model: Model = model
        # For now store model output for later processing here
        self.data_pipe = data_pipe
        self.result_storage_pipe = result_storage_pipe

    def read_data(self) -> Tuple[EntityState, ndarray] :
        """
        Read data for current node at simulation tick
        """
        agent_state = self.dataloader.read_entity_state(self.node_id, self.env.now)
        camera_image = self.dataloader.read_images(self.node_id, self.env.now)
        return agent_state, camera_image

    def summarize_output(self, raw_output: object, 
                         im_shape, state: EntityState) -> OutputSummary:
        """
        Convert raw model results into OutputSummary data class.
        Only accept results with class car or person.
        """
        # Remove the .pandas() to get the data as a tensor. Might boost performance
        # Pandas offers columns xmin, ymin, xmax, ymax, confidence, class, name
        pandas_yolo_results = raw_output.pandas().xyxy[0]

        # Convert yolo detections to OutputSummary
        # and exclude results for classes other than car and human.
        # Note, for optimization instead of using iterrows
        # alternative panda methods could possibly be used.
        detections = []
        for i, row in pandas_yolo_results.iterrows():
            if row['name'] == 'car' or row['name'] == 'person':

                # Filter out matches where the car detects itself by not taking
                # Get about 1/3 of image bottom size
                ymin_max = im_shape[0] / 1.6
                # Get about 1/6 of the bottom of the image size
                ymax_max = im_shape[0] / 16

                if row['name'] == 'car':
                    if row['ymin'] > ymin_max and row['ymax'] > ymax_max:
                        continue

                detection = DetectionData(
                    id=f"{i}-{self.node_id}",
                    type=row['name'],
                    xmin=row['xmin'],
                    xmax=row['xmax'],
                    ymin=row['ymin'],
                    ymax=row['ymax']
                )
                detections.append(detection)

        output = OutputSummary(
            node_id=self.node_id,
            is_rsu=state.is_rsu,
            agent_x=state.x,
            agent_y=state.y,
            direction=state.direction,
            velocity=state.velocity,
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
            # Debugging / visualization
            output = self.summarize_output(raw_output, image.shape, state)

            json_friendly_list = []
            if len(output.detections) > 0:
                for detection in output.detections:
                    json_friendly = (
                        detection.id,
                        detection.type,
                        detection.xmin,
                        detection.xmax,
                        detection.ymin,
                        detection.ymax,
                    )
                    json_friendly_list.append(json_friendly)
            self.result_storage_pipe['yolo_images'][self.node_id].append(json_friendly_list)
            
            self.data_pipe[self.node_id] = output
            yield self.env.timeout(1)
            