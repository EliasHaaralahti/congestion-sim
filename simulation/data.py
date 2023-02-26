"""
All data from CARLA is stored in the HDF5 format and is processed 
via the h5py library.
"""
import io
import ast
import math
from typing import List, Tuple
import h5py
from numpy import ndarray, asarray
from PIL import Image
from data_models.agent_state import AgentState


class DataLoader():
    """
    Abstraction class for reading data from the hdf5 files.
    """
    def __init__(self, file='data.hdf5'):
        self.h5file = h5py.File(file, 'r')

    def read_agent_ids(self) -> list[str]:
        """
        Read from the simulation data all the agent identifiers
        that occur in the current scene.
        """
        return list(self.h5file['sensors'].keys())
    
    def get_simulation_length(self) -> int:
        data = self.h5file.get("sensors/", 'r')
        return len(data[list(data)[0]])

    def read_images(self, agent: str, simulation_step: int) -> ndarray:
        """
        Returns agent camera photo at simulation step
        """
        frames = self.h5file.get(f'sensors/{agent}', 'r')
        frame = frames[simulation_step]
        img = Image.open(io.BytesIO(frame))
        img_data = asarray(img)
        # BGR -> RGB
        return img_data[:,:,::-1]
    
    def get_map(self) -> List[Tuple[int, int]]:
        """
        Returns all map markers for visualization purposes.
        """
        data = self.h5file.get("metadata", 'r')
        # .value is old syntax. [()] does same now.
        data = data[()].decode("UTF-8")
        # Convert bytes to a dictionary
        data = ast.literal_eval(data)
        return data['waypoints']


    def read_agent_state(self, agent: str, simulation_step: int) -> AgentState:
        """
        Returns agent state at simulation step
        """
        agent_vehicle = agent.replace('camera_', 'vehicle_')
        data = self.h5file.get(f'state/{agent_vehicle}')[simulation_step]
        vel_x, vel_y = self.h5file.get(f'velocity/{agent_vehicle}')[simulation_step]
        velocity = math.hypot(vel_x, vel_y)
        return AgentState(x=data[0], y=data[1], direction=data[2], velocity=velocity)

    def __del__(self):
        self.h5file.close()


# Main for testing the class.
if __name__ == "__main__":
    print("Running dataloader as main for testing")
    dataloader = DataLoader()

    sim_length = dataloader.get_simulation_length()
    print(f"Simulation length: {sim_length}")

    map_points = dataloader.get_map()
    print(f"Map points len: {len(map_points)}, point 0: {map_points[0]}")

    agents = dataloader.read_agent_ids()
    print(f"Agent ids: {agents}")
    
    agent_test = agents[0]
    SIMULATION_STEP = 0

    state = dataloader.read_agent_state(agent_test, SIMULATION_STEP)
    print(f"Agent state: {state}")


    image = dataloader.read_images(agent_test, SIMULATION_STEP)
    print(f"Image shape: {image.shape}")
