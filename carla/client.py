import carla
import json
from queue import Queue, Empty
from environment import CarlaEnv
from recorder import Recorder

img_width = 640
img_height = 480

n_frames = 800
fps = 60

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))

def create_metadata():
    metadata = {
    "n_frames": n_frames,
    "fps": fps
    }
    return json.dumps(metadata)

sensor_queue = Queue()

images_1 = []
images_2 = []

transforms_1 = []
transforms_2 = []

try:
    env = CarlaEnv(host='192.168.0.114')
    recorder = Recorder(img_width=img_width, img_height=img_height)
    
    env.set_sync_mode(fps=fps)

    vehicle_bp_1 = env.blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp_2 = env.blueprint_library.find('vehicle.tesla.cybertruck')
    
    spawn_point_1 = env.spawn_points[34]
    spawn_point_2 = env.spawn_points[129]
    route_indices_1 = [35, 79]
    route_indices_2 = [28, 79]
    route_1 = []
    route_2 = []

    for i in route_indices_1:
        route_1.append(env.spawn_points[i].location)

    for i in route_indices_2:
        route_2.append(env.spawn_points[i].location)

    vehicle_1 = env.world.spawn_actor(vehicle_bp_1, spawn_point_1)
    vehicle_2 = env.world.spawn_actor(vehicle_bp_2, spawn_point_2)
    
    env.vehicle_list.append(vehicle_1)
    env.vehicle_list.append(vehicle_2)

    camera_bp = env.blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{img_width}')
    camera_bp.set_attribute('image_size_y', f'{img_height}')

    transform_1 = carla.Transform(carla.Location(x=0, z=2))
    transform_2 = carla.Transform(carla.Location(x=1, z=2.5))
    
    camera_1 = env.world.spawn_actor(camera_bp, transform_1, attach_to=vehicle_1)
    camera_2 = env.world.spawn_actor(camera_bp, transform_2, attach_to=vehicle_2)
    
    env.sensor_list.append(camera_1)
    env.sensor_list.append(camera_2)

    camera_1.listen(lambda image: sensor_callback(image, sensor_queue, "camera_1"))
    camera_2.listen(lambda image: sensor_callback(image, sensor_queue, "camera_2"))

    env.set_autopilot()
    
    env.traffic_manager.set_path(vehicle_1, route_1)
    env.traffic_manager.set_path(vehicle_2, route_2)

    metadata_json = create_metadata()

    curr_frame = 0

    while curr_frame < n_frames:
        env.world.tick()
        transform_1 = recorder.process_transform(vehicle_1.get_transform())
        transform_2 = recorder.process_transform(vehicle_2.get_transform())
        transforms_1.append(transform_1)
        transforms_2.append(transform_2)
        try:
            for _ in range(len(env.sensor_list)):
                data = sensor_queue.get(True, 1.0)
                if data[1] == 'camera_1':
                    images_1.append(recorder.process_img(data[0]))
                else:
                    images_2.append(recorder.process_img(data[0]))
        except Empty:
            print("Some of the sensor information is missed")
        curr_frame += 1

except KeyboardInterrupt:
        print("\nCancelled by user")
finally:
    env.set_original_settings()
    env.clear_actors()

    recorder.h5file.create_dataset('metadata', data=metadata_json)
    recorder.sensors_group.create_dataset('vehicle_1', data=images_1)
    recorder.sensors_group.create_dataset('vehicle_2', data=images_2)
    recorder.state_group.create_dataset('vehicle_1', data=transforms_1)
    recorder.state_group.create_dataset('vehicle_2', data=transforms_2)

    recorder.stop_recording()
