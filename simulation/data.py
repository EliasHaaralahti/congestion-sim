"""
All data from CARLA is stored in the HDF5 format and is processed 
via the h5py library.
"""
import os
import io
import json
import math
from typing import List, Tuple
import h5py
from numpy import ndarray, asarray
from PIL import Image
from data_models.agent_state import EntityState


class DataLoader():
    """
    Abstraction class for reading data from the hdf5 files.
    """
    def __init__(self, file='intersection_5_vehicles.hdf5'):
        self.h5file = h5py.File(os.path.join("runs/", file), 'r')

    def get_entity_ids(self) -> list[str]:
        """
        Read from the simulation data all the entity identifiers
        that occur in the current scene. This includes RSUs.
        """
        return list(self.h5file['sensors'].keys())

    def get_intersections(self) -> list[object]:
        """
        Read intersection metadata to get the 
        locations of intersections
        """
        data = self.h5file.get("metadata", 'r')
        # .value is old syntax. [()] does same now.
        data = data[()].decode("UTF-8")
        # Convert bytes to a dictionary
        data = json.loads(data)
        return data.get("intersections", 'r')

    def get_simulation_length(self) -> int:
        data = self.h5file.get("sensors/", 'r')
        return len(data[list(data)[0]])

    def read_images(self, agent_name: str, simulation_step: int) -> ndarray:
        """
        Returns entity camera photo at simulation step
        """
        frames = self.h5file.get(f'sensors/{agent_name}', 'r')
        frame = frames[simulation_step]
        img = Image.open(io.BytesIO(frame))
        img_data = asarray(img)
        return img_data

    def get_metadata_summary(self):
        """
        Summary for visualization of scene metadata
        """
        data = self.h5file.get("metadata", 'r')
        # .value is old syntax. [()] does same now.
        data = data[()].decode("UTF-8")
        # Convert bytes to a dictionary
        data = json.loads(data)
        metadata_summary = {
            "timestamp": data['timestamp'],
            "map_name": data['map'],
            "n_frames": data['n_frames'],
            "fps": data['fps'],
            "n_vehicles": data['n_vehicles'],
            "n_sensors": data['n_sensors'],
        }
        return metadata_summary


    def get_map(self) -> List[Tuple[int, int]]:
        """
        Returns all map markers for visualization purposes.
        """
        data = self.h5file.get("metadata", 'r')
        # .value is old syntax. [()] does same now.
        data = data[()].decode("UTF-8")
        data = json.loads(data)
        return data['waypoints']


    def read_entity_state(self, entity: str, simulation_step: int) -> EntityState:
        """
        Returns entity state at simulation step for a vehicle. 
        Note RSUs do not have state.
        """
        entity_vehicle = entity.replace('camera_', 'vehicle_')
        
        data = self.h5file.get(f'state/{entity_vehicle}')
        if data is None: # If no state, the entity is a RSU
            data = self.h5file.get("metadata", 'r')
            data = data[()].decode("UTF-8")
            data = json.loads(data)
            
            # Find the RSU under metadata/sensors. A RSU does not have a parent and has a location.
            for entity_data in data.get('sensors'):
                if "location" in entity_data and entity_data['id'] == entity:
                    return EntityState(is_rsu=True, x=entity_data['location']['x'], y=entity_data['location']['y'], 
                                       direction=entity_data['rotation']['yaw'], velocity=0)

            return None # No RSU found.
        
        data = data[simulation_step]
        vel_x, vel_y = self.h5file.get(f'velocity/{entity_vehicle}')[simulation_step]
        velocity = math.hypot(vel_x, vel_y)
        return EntityState(is_rsu=False, x=data[0], y=data[1], direction=data[2], velocity=velocity)

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

    intersections = dataloader.get_intersections()
    print(f"Intersections: {intersections}")

    agents = dataloader.get_entity_ids()
    print(f"Agent ids: {agents}")
    
    print("Testing all agents")
    for agent in agents:
        print(f"\nAgent {agent}")
        SIMULATION_STEP = 0

        state = dataloader.read_entity_state(agent, SIMULATION_STEP)
        if state is None:
            raise Exception("ERROR: State should not be none!")
        
        print(f"Agent state: {state}")

        image = dataloader.read_images(agent, SIMULATION_STEP)
        print(f"Image shape: {image.shape}")

        metadata = dataloader.get_metadata_summary()
        print(f"Metadata: {metadata}")