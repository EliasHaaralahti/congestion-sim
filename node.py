from model import Model
from data_models.output_summary import OutputSummary
from data import DataLoader

class Node():
    # Initialize with sympy environment
    def __init__(self, env: object, node_id: str, dataloader: DataLoader, model: Model):
        self.env: object = env
        self.node_id: str = node_id
        self.dataloader: DataLoader = dataloader
        self.model: Model = model
        # For now store model output for later processing here
        self.output: OutputSummary = None

    def read_data_at_tick(self, node_id: str) -> list[int]:
        """ TODO
        Read data for specified node at simulation tick using self.node_id
        """
        print(f"Reading data for node {node_id}")
        return self.dataloader.read_images()

    def summarize_output(self, raw_output: object) -> OutputSummary:
        """ TODO
        Convert raw model results into OutputSummary data class
        """
        return raw_output

    def run(self):
        """
        Generator for nodes where at tick t matching simulator
        data is read for the specific node.
        """
        while True:
            print(f"Processing node {self.node_id}")
            data = self.read_data_at_tick(self.node_id)
            raw_output = self.model.forward(data)
            print(f"Model output: {raw_output}")
            # TODO: Communicate output to other nodes/central computer?
            # At least temporarily use self.output to store the information.
            self.output = self.summarize_output(raw_output)
            yield self.env.timeout(1)
            