"""Script containing a class for saving data from the CARLA simulation into HDF5."""
import io
from queue import Queue, Empty
import h5py
import numpy as np
from PIL import Image

class Recorder:
    """
    Class for processing data from the CARLA simulation. Specify the camera resolution
    and the name of the resulting HDF5 file.
    """
    def __init__(self, img_width: int, img_height: int, filename: str):
        self.img_width = img_width
        self.img_height = img_height
        self.sensor_queue = Queue()
        self.labels = []
        self.h5file = h5py.File(f'runs/{filename}.hdf5', 'w')
        self.sensors_group = self.h5file.create_group('sensors')
        self.state_group = self.h5file.create_group('state')
        self.velocity_group = self.h5file.create_group('velocity')

    def sensor_callback(self, sensor_data: object, sensor_name: str) -> None:
        """Add data from the sensor to a queue."""
        self.sensor_queue.put((sensor_data, sensor_name))

    def process_images(self, sensor_list: list, images: dict) -> None:
        """Process images from all sensors into `images` dictionary."""
        try:
            for _ in range(len(sensor_list)):
                data = self.sensor_queue.get(True, 1.0)
                sensor_name = data[1]
                image = self.process_img(data[0])
                images[sensor_name].append(image)
        except Empty:
            print('Some of the sensor information is missed')

    def process_img(self, image: object) -> np.ndarray:
        """Transform `image` object to binary format and convert it to numpy ndarray."""
        img_data = np.array(image.raw_data)
        reshaped_data = img_data.reshape((self.img_height, self.img_width, 4))[:, :, :3]
        rgb_img = reshaped_data[:, :, ::-1]
        img = Image.fromarray(rgb_img)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        byte_img = buf.getvalue()
        return np.asarray(byte_img)

    def process_transforms(self, vehicle_list: list, transforms: list) -> None:
        """Process the transforms of all the spawned vehicles into `transforms` list."""
        for i, vehicle in enumerate(vehicle_list):
            vehicle_id = f'vehicle_{i+1}'
            transform = vehicle.get_transform()
            state_tuple = (transform.location.x, transform.location.y, transform.rotation.yaw)
            transforms[vehicle_id].append(state_tuple)

    def process_velocities(self, vehicle_list: list, velocities: list) -> None:
        """Process the velocities of all the spawned vehicles into `velocities` list."""
        for i, vehicle in enumerate(vehicle_list):
            vehicle_id = f'vehicle_{i+1}'
            velocity = vehicle.get_velocity()
            velocity_tuple = (velocity.x, velocity.y)
            velocities[vehicle_id].append(velocity_tuple)

    def process_labels(self, avg_velocity: float, ratio: float, speed_limit: float=30.0) -> None:
        """
        Process the labels for the images. If `avg_velocity` / `speed_limit` is less than `ratio`,
        the images get `congested` label. Otherwise, the label is `not_congested`.
        """
        label = 'congested' if avg_velocity / speed_limit < ratio else 'not_congested'
        print(f'Avg velocity: {round(avg_velocity, 2)} km/h, label: {label}')
        self.labels.append(label)

    def create_datasets(self, transforms: dict, velocities: dict, images: dict, vehicle_list: list,
                        sensor_list: list, metadata: dict, training: bool=False) -> None:
        """Save vehicle transforms, vehicle velocities, images and metadata into HDF5."""
        self.h5file.create_dataset('metadata', data=metadata)
        if training:
            self.h5file.create_dataset('labels', data=self.labels)
        for i in range(len(vehicle_list)):
            vehicle_id = f'vehicle_{i+1}'
            self.state_group.create_dataset(vehicle_id, data=transforms[vehicle_id])
            self.velocity_group.create_dataset(vehicle_id, data=velocities[vehicle_id])
        for i in range(len(sensor_list)):
            camera_id = f'camera_{i+1}'
            self.sensors_group.create_dataset(camera_id, data=images[camera_id])

    def stop_recording(self) -> None:
        """Stop the recording and close the HDF5 file."""
        self.h5file.close()
