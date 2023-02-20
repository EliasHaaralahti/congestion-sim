import h5py
import numpy as np
import io
from queue import Queue, Empty
from PIL import Image

class Recorder:
    def __init__(self, img_width, img_height, filename='data'):
        self.img_width = img_width
        self.img_height = img_height
        self.sensor_queue = Queue()
        self.h5file = h5py.File(f'runs/{filename}.hdf5', 'w')
        self.sensors_group = self.h5file.create_group('sensors')
        self.state_group = self.h5file.create_group('state')

    def sensor_callback(self, sensor_data, sensor_name):
        self.sensor_queue.put((sensor_data, sensor_name))

    def process_images(self, sensor_list, images):
        try:
            for _ in range(len(sensor_list)):
                data = self.sensor_queue.get(True, 1.0)
                sensor_name = data[1]
                image = self.process_img(data[0])
                images[sensor_name].append(image)
        except Empty:
            print("Some of the sensor information is missed")

    def process_img(self, image):
        img_data = np.array(image.raw_data)
        reshaped_data = img_data.reshape((self.img_height, self.img_width, 4))[:, :, :3]
        img = Image.fromarray(reshaped_data)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        byte_img = buf.getvalue()
        return np.asarray(byte_img)
    
    def process_transforms(self, vehicle_list, transforms):
        for i in range(len(vehicle_list)):
            vehicle_id = f'vehicle_{i+1}'
            transform = self.process_transform(vehicle_list[i].get_transform())
            transforms[vehicle_id].append(transform)

    def process_transform(self, transform):
        state_tuple = (transform.location.x, transform.location.y, transform.rotation.yaw)
        return state_tuple
    
    def create_datasets(self, transforms, images, vehicle_list, sensor_list, metadata):
        self.h5file.create_dataset('metadata', data=metadata)
        for i in range(len(vehicle_list)):
            vehicle_id = f'vehicle_{i+1}'
            self.state_group.create_dataset(vehicle_id, data=transforms[vehicle_id])
        for i in range(len(sensor_list)):
            camera_id = f'camera_{i+1}'
            self.sensors_group.create_dataset(camera_id, data=images[camera_id])

    def stop_recording(self):
        self.h5file.close()