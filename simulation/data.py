"""
All data from CARLA is stored in the HDF5 format and is processed 
via the h5py library.
"""
import h5py
from numpy import ndarray
from data_models.agent_state import AgentState


class DataLoader():
    """
    Abstraction class for reading data from the hdf5 files.
    Examples:

    sensors/agent contains images
    state/agent contains x,y,direction
    sensor_data_for_vehicle_1 = data['sensors/vehicle_1']
    sensor_data_for_vehicle_2 = data['sensors/vehicle_2']
    state_data_for_vehicle_1 = data['state/vehicle_1']
    state_data_for_vehicle_2 = data['state/vehicle_2']
    """
    def __init__(self, file='data.hdf5'):
        self.h5file = h5py.File(file, 'r')

    def read_agent_ids(self) -> list[str]:
        """
        Read from the simulation data all the agent identifiers
        that occur in the current scene.
        """
        return list(self.h5file['sensors'].keys())

    def read_images(self, agent:str, simulation_step: int) -> ndarray:
        """
        Returns agent camera photo at simulation step
        """
        data = self.h5file.get(f'sensors/{agent}')
        return data[simulation_step] 

    def read_agent_state(self, agent: str, simulation_step: int) -> AgentState:
        """
        Returns agent state at simulation step
        """
        data = self.h5file.get(f'state/{agent}')[simulation_step]
        return AgentState(x=data[0], y=data[1], direction=data[2])

    def __del__(self):
        self.h5file.close()
        

# Main for testing the class.
if __name__ == "__main__":
    print("Running dataloader as main")
    dataloader = DataLoader()

    agent = dataloader.read_agent_ids()[0]
    simulation_step = 0
    image = dataloader.read_images(agent, simulation_step)
    print(image.shape)



