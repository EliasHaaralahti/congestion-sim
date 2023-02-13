import simpy
from node import Node
from processor import processor
from data import DataLoader
from model import Model


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

    # Use a dictionary entry for all agents and the value will be the latest output.
    # This will be a temporary communication pipe
    # Later this could use actual simpy store as done in
    # https://simpy.readthedocs.io/en/latest/examples/process_communication.html
    data_pipe = {}

    # Create nodes and add to simulation as processes
    for node_id in dataloader.read_agent_ids():
        #nodes.append( Node(env, node_id, dataloader, model) )
        node = Node(env, node_id, dataloader, model, data_pipe)
        env.process( node.run() )

    # Create the 'central processor' process.
    env.process( processor(env, data_pipe) )

    print("Starting simulation")
    env.run(until=2)
    print("Simulation done")


if __name__ == "__main__":
    main()