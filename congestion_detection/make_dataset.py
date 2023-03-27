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

dataset_1 = h5py.File('../carla/runs/intersection_1_100_vehicles.hdf5', 'r')
dataset_2 = h5py.File('../carla/runs/intersection_2_100_vehicles.hdf5', 'r')

datasets = [dataset_1, dataset_2]

n_samples = {
    'congested': 0,
    'not_congested': 0
}

for i, dataset in enumerate(datasets):
    labels = dataset['labels']
    sensors = dataset['sensors']
    for j, sensor in enumerate(dataset['sensors']):
        for k, frame in enumerate(dataset[f'sensors/{sensor}']):
            print('Processing images from dataset: ' + str(i+1) + '/' + str(len(datasets)) +
                  ', sensor: ' + str(j+1) + '/' + str(len(sensors)) + ', timestep: ' + str(k+1) +
                  '/' + str(len(labels)))
            img = Image.open(io.BytesIO(frame))
            label = labels[k].decode('utf-8')
            path = f'classes/{label}'
            img.save(f'{path}/dataset_{i+1}_{sensor}_frame_{k+1}.png')
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
