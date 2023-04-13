"""Script for classifying the simulated scenarios using the CongestionDetector model"""
import io
import json
import h5py
from model import CongestionDetector
from PIL import Image

detector = CongestionDetector()
detector.load_weights('models/model.pt')

data = h5py.File('../carla/runs/intersection_20_vehicles.hdf5', 'r')

metadata = json.loads(data['metadata'][()])
ground_truth = metadata['congestion_statistics']

intersection_1 = 0
intersection_2 = 0

# Ground truth
for intersection in ground_truth:
    percentage = 100 * ground_truth[intersection] / 1000
    print(f'{intersection} was congested {percentage}% of the time')

# Intersection 1
for frame in data['sensors/camera_21']:
    img = Image.open(io.BytesIO(frame))
    prediction_probabilities = detector.predict(img)
    prediction = max(prediction_probabilities, key=prediction_probabilities.get)
    if prediction == 'congested':
        intersection_1 += 1

for frame in data['sensors/camera_22']:
    img = Image.open(io.BytesIO(frame))
    prediction_probabilities = detector.predict(img)
    prediction = max(prediction_probabilities, key=prediction_probabilities.get)
    if prediction == 'congested':
        intersection_1 += 1

intersection_1_percentage = 100 * intersection_1 / 2000
print(f'intersection_1 was predicted to be congested {intersection_1_percentage}% of the time')

# Intersection 2
for frame in data['sensors/camera_23']:
    img = Image.open(io.BytesIO(frame))
    prediction_probabilities = detector.predict(img)
    prediction = max(prediction_probabilities, key=prediction_probabilities.get)
    if prediction == 'congested':
        intersection_2 += 1

for frame in data['sensors/camera_24']:
    img = Image.open(io.BytesIO(frame))
    prediction_probabilities = detector.predict(img)
    prediction = max(prediction_probabilities, key=prediction_probabilities.get)
    if prediction == 'congested':
        intersection_2 += 1

intersection_2_percentage = 100 * intersection_2 / 2000
print(f'intersection_2 was predicted to be congested {intersection_2_percentage}% of the time')
