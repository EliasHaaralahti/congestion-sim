import simpy
from node import Node
from processor import processor
from data import DataLoader
from model import Model


# Each environment tick corresponds
# to a tick in the simulation data.
env = simpy.Environment()
dataloader = DataLoader()
model = Model()

nodes = []
for node_id in dataloader.read_agent_ids():
    nodes.append( Node(env, node_id, dataloader, model) )

# Run each node
for node in nodes:
    env.process( node.run() )

# Extract node outputs from nodes for processing.
node_outputs = [node.output for node in nodes]
# Run the processor with data acquired from the nodes
env.process( processor(env, node_outputs) )

print("Starting simulation")
env.run(until=1)
print("Simulation done")
