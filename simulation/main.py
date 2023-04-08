import json
import os
import sys
import getopt
import time
import simpy
from node import Node
from processor import Processor
from data import DataLoader
from model import Model


def print_progress(env, max_steps):
    while True:
        print(f"Finished processing step {env.now + 1} / " \
              f"{max_steps}", end='\r')
        yield env.timeout(1)


def write_data(run_name: str, processed_data: dict):
    # Write collected data to multiple output files
    folder = f"results/{run_name}/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # First write simulation results
    path_results = folder + "results.json"
    with open(path_results, 'w', encoding="utf-8") as file:
        json.dump(processed_data['processing_results'], file)

    # Second write the yolo detection results, used for visualization.
    path_yolo = folder + "yolo_results.json"
    with open(path_yolo, 'w', encoding="utf-8") as file:
        data = processed_data['yolo_images']
        # Convert image output objects to dicts for json.
        for i, detection in enumerate(data):
            data[i] = detection.to_json()

        #print(type(data))
        json.dump(data, file)

def run_simulation(
        model_name: str, environment: str, use_rsu: bool, verbose: bool):
    """
    Simpy simulation for agents to combine data from sensors to create
    an overview of the situation by sharing data. The agents are represented
    as network nodes, while the processor represents a centralized processing unit
    that creates the overview. Simulation ticks correspond to ticks, at which data
    was collected from the Carla simulator.
    """
    env = simpy.Environment()
    dataloader = DataLoader(environment)
    yolo_model = Model(model_name)

    sim_length = dataloader.get_simulation_length()
    agent_ids = dataloader.get_entity_ids()
    # Use a dictionary entry for all agents and the value will be the latest output.
    # This will be a temporary communication pipe
    # Later this could use actual simpy store as done in
    # https://simpy.readthedocs.io/en/latest/examples/process_communication.html
    data_pipe = {}
    # yolo_images is dict with key agent value detections for each timestep
    result_storage_pipe = {
        'processing_results': {
            "agents": [],
            "intersection_statuses": []
        },
        'yolo_images': []
    }

    # Create nodes and add to simulation as processes
    for node_id in agent_ids:
        node = Node(env, node_id, dataloader, yolo_model, data_pipe, result_storage_pipe)
        env.process( node.run() )

    # Create the 'central processor' process.
    processor = Processor(env, data_pipe, result_storage_pipe, dataloader)
    env.process( processor.run() )

    if verbose:
        env.process( print_progress(env, sim_length))

    print(f"\n\nStarting simulation with {sim_length} timesteps")
    start_time = time.time()
    env.run(until=sim_length)
    final_time = time.time() - start_time
    loop_time = final_time / sim_length
    
    print("") # <- as previous prints may not have had line endings
    print(f"Simulation lasted {final_time:.1f} seconds.")
    print(f"Simulation for each timestep took approximitely {loop_time:.3f} seconds.")
    # Write processed results for visualization
    run_name = f"{model_name}-{environment}-rsu_used_{USE_RSU}-{int(time.time())}"
    write_data(run_name, result_storage_pipe)


def print_help(model_options):
    help_text = "Read the README before running. Carla runs must be stored under " \
        "the folder \"runs\"\n" \
        "Usage: python main.py. Possible arguments:\n" \
        "\t-h - Help\n" \
        "\t--no_verbose - No print statements inside the simulation \n" \
        "\t-no_rsu - Do not use RSUs in the simulation.\n" \
        "\t--model <model> - Can be either single model name or " \
        "a list of models separated by comma.\n" \
        f"\tOptions: {model_options}\n" \
        "\t--environment <env> - Name of the CARLA data file to be used.\n" \
        "\n\tExample: python main.py -no_rsu --model nano,medium " \
        "\ --environment intersection_5_vehicles.hdf5"
    print(help_text)


if __name__ == "__main__":
    # Parse command line arguments (-h for help)
    MODEL_OPTIONS = ["nano", "small", "medium", "large", "xlarge"]

    RUN = True
    MODEL = ["medium"]
    USE_RSU = True
    VERBOSE = True
    CARLA_ENVIRONMENT = "intersection_5_vehicles.hdf5"
    opts, args = getopt.getopt(sys.argv[1:], "h", ["model=", "environment="])
    for opt, arg in opts:
        if opt == "-h":
            print_help(MODEL_OPTIONS)
            RUN = False
        if opt == "--no_verbose":
            VERBOSE = False
        if opt == "-no_rsu":
            USE_RSU = False
        if opt == "--model":
            # If no commas, only model name is returned.
            MODEL = arg.split(',')
        if opt == "--environment":
            CARLA_ENVIRONMENT = arg

    # If model is a list, run each model in different simulation.
    if not RUN:
        sys.exit(0)

    # Ensure model is a list (even if only using a single model) 
    # and that all items are strings, which are in MODEL_OPTIONS.
    if isinstance(MODEL, list) and all(
        isinstance(item, str) and item in MODEL_OPTIONS for item in MODEL):
        print(f"Running model(s): {MODEL}")
        for model in MODEL:
            print("\n============================================")
            print(f"Running simulation with settings \n" \
                    f"- model: {model}\n" \
                    f"- environment: {CARLA_ENVIRONMENT}\n" \
                    f"- USE_RSU: {USE_RSU}\n" \
                    f"- verbose: {VERBOSE}\n")
            run_simulation(model, CARLA_ENVIRONMENT, USE_RSU, VERBOSE)
    else:
        print("MODEL argument is wrong. See -h for help.")
    print("\nDone")
    