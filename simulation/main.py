import simpy
from node import Node
from processor import processor
from data import DataLoader
from model import Model
import json


def write_data(processed_data):
    with open('results/result.json', 'w') as file:
        json.dump(processed_data, file)


def main():
    """
    Simpy simulation for agents to combine data from sensors to create
    an overview of the situation by sharing data. The agents are represented
    as network nodes, while the processor represents a centralized processing unit
    that creates the overview. Simulation ticks correspond to ticks, at which data
    was collected from the Carla simulator.
    """
    env = simpy.Environment()
    dataloader = DataLoader()
    model = Model()

    agent_ids = dataloader.read_agent_ids()
    # Use a dictionary entry for all agents and the value will be the latest output.
    # This will be a temporary communication pipe
    # Later this could use actual simpy store as done in
    # https://simpy.readthedocs.io/en/latest/examples/process_communication.html
    data_pipe = {}
    # yolo_images is dict with key agent value detections for each timestep
    result_storage_pipe = {'processing_results': [], 'yolo_images': {}}
    # Add yolo_images key and value array
    for agent in agent_ids:
        result_storage_pipe['yolo_images'][agent] = []

    # Create nodes and add to simulation as processes
    for node_id in agent_ids:
        #nodes.append( Node(env, node_id, dataloader, model) )
        node = Node(env, node_id, dataloader, model, data_pipe, result_storage_pipe)
        env.process( node.run() )

    # Create the 'central processor' process.
    env.process( processor(env, data_pipe, result_storage_pipe) )

    print("Starting simulation")
    env.run(until=800)
    print("Simulation done")
    # Write processed results for visualization
    write_data(result_storage_pipe)


if __name__ == "__main__":
    main()