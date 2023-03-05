import json
import os
import time
import simpy
from node import Node
from processor import processor
from data import DataLoader
from model import Model



def write_data(processed_data):
    folder = "results/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    path = folder + "results.json"
    with open(path, 'w', encoding="utf-8") as file:
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

    sim_length = dataloader.get_simulation_length()
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
        node = Node(env, node_id, dataloader, model, data_pipe, result_storage_pipe)
        env.process( node.run() )

    # Create the 'central processor' process.
    env.process( processor(env, data_pipe, result_storage_pipe) )

    print(f"\n\nStarting simulation with {sim_length} timesteps")
    start_time = time.time()
    env.run(until=sim_length)
    final_time = time.time() - start_time
    loop_time = final_time / sim_length
    
    print(f"Simulation done in {final_time:.1f} seconds.")
    print(f"Simulation for each timestep took approximitely {loop_time:.3f} seconds.")
    # Write processed results for visualization
    write_data(result_storage_pipe)


if __name__ == "__main__":
    main()