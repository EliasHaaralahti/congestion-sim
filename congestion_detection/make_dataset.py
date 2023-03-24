import os
import shutil
import io
from random import sample
import h5py
import splitfolders
from PIL import Image

os.makedirs('classes')
os.makedirs('classes/congested')
os.makedirs('classes/not_congested')

data = h5py.File("../carla/runs/training_data.hdf5", 'r')
labels = data['labels']
sensors = data['sensors']

n_samples = {
    'congested': 0,
    'not_congested': 0
}

for i, sensor in enumerate(data['sensors']):
    for j, frame in enumerate(data[f'sensors/{sensor}']):
        print(f'Processing images from sensor: {i+1}/{len(sensors)}, timestep: {j+1}/{len(labels)}')
        img = Image.open(io.BytesIO(frame))
        label = labels[j].decode('utf-8')
        path = f'classes/{label}'
        img.save(f'{path}/{sensor}_frame_{j}.png')
        n_samples[label] += 1

max_label = max(n_samples, key=n_samples.get)
min_label = min(n_samples, key=n_samples.get)
class_diff = n_samples[max_label] - n_samples[min_label]

print(f'There are {class_diff} more images in class {max_label}')

print('Undersampling...')

path = f'classes/{max_label}'
files = os.listdir(path)

for file in sample(files, class_diff):
    os.remove(f'{path}/{file}')

print('Splitting data into train, validation and test sets')

splitfolders.ratio('classes', 'data')

shutil.rmtree('classes')
