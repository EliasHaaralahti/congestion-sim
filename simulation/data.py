"""
All data from CARLA is stored in the HDF5 format and is processed 
via the h5py library.
"""
import h5py
import numpy as np


class DataLoader():
    def __init__(self, file='test.h5'):
        self.h5file = h5py.File(file, 'r')

    def read_agent_ids(self) -> list[str]:
        """ TODO
        Read from the simulation data all the agent identifiers
        that occur in the current scene.
        """
        return ["1", "2"]

    def read_images(self) -> np.ndarray:
        print(self.h5file.keys())
        return self.h5file.get('images')

    def __del__(self):
        self.h5file.close()
        