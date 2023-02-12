import h5py
import numpy as np
import io
from PIL import Image

class Recorder:
    def __init__(self, filename='data', img_width=1280, img_height=720):
        self.img_width = img_width
        self.img_height = img_height
        self.h5file = h5py.File(f'runs/{filename}.hdf5', 'w')
        self.sensors_group = self.h5file.create_group('sensors')
        self.state_group = self.h5file.create_group('state')

    def process_img(self, image):
        img_data = np.array(image.raw_data)
        reshaped_data = img_data.reshape((self.img_height, self.img_width, 4))[:, :, :3]
        img = Image.fromarray(reshaped_data)
        buf = io.BytesIO()
        img.save(buf, format='JPEG')
        byte_img = buf.getvalue()
        return np.asarray(byte_img)

    def process_transform(self, transform):
        state_tuple = (transform.location.x, transform.location.y, transform.rotation.yaw)
        return state_tuple

    def stop_recording(self):
        self.h5file.close()