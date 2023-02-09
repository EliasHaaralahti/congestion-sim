import carla
import numpy as np
import h5py
import io
from PIL import Image
from queue import Queue, Empty

img_width = 640
img_height = 480

n_frames = 800
frames_per_second = 60

def process_img(image, images):
    img_data = np.array(image.raw_data)
    reshaped_data = img_data.reshape((img_height, img_width, 4))[:, :, :3]
    img = Image.fromarray(reshaped_data)
    buf = io.BytesIO()
    img.save(buf, format='JPEG')
    byte_img = buf.getvalue()
    images.append(np.asarray(byte_img))

def process_transform(transform, transforms):
    state_tuple = (transform.location.x, transform.location.y, transform.rotation.yaw)
    transforms.append(state_tuple)

def sensor_callback(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data, sensor_name))

vehicle_list = []
sensor_list = []

sensor_queue = Queue()

images_1 = []
images_2 = []

transforms_1 = []
transforms_2 = []

try:
    client = carla.Client('192.168.0.114', 2000)
    world = client.get_world()

    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 1 / frames_per_second
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    blueprint_library = world.get_blueprint_library()

    vehicle_bp_1 = blueprint_library.find('vehicle.tesla.model3')
    vehicle_bp_2 = blueprint_library.find('vehicle.tesla.cybertruck')
    
    spawn_points = world.get_map().get_spawn_points()
    
    spawn_point_1 = spawn_points[34]
    spawn_point_2 = spawn_points[129]
    route_indices_1 = [35, 79]
    route_indices_2 = [28, 79]
    route_1 = []
    route_2 = []

    for i in route_indices_1:
        route_1.append(spawn_points[i].location)

    for i in route_indices_2:
        route_2.append(spawn_points[i].location)

    vehicle_1 = world.spawn_actor(vehicle_bp_1, spawn_point_1)
    vehicle_2 = world.spawn_actor(vehicle_bp_2, spawn_point_2)
    
    vehicle_list.append(vehicle_1)
    vehicle_list.append(vehicle_2)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{img_width}')
    camera_bp.set_attribute('image_size_y', f'{img_height}')

    transform_1 = carla.Transform(carla.Location(x=0, z=2))
    transform_2 = carla.Transform(carla.Location(x=1, z=2.5))
    
    camera_1 = world.spawn_actor(camera_bp, transform_1, attach_to=vehicle_1)
    camera_2 = world.spawn_actor(camera_bp, transform_2, attach_to=vehicle_2)
    
    sensor_list.append(camera_1)
    sensor_list.append(camera_2)

    camera_1.listen(lambda image: sensor_callback(image, sensor_queue, "camera_1"))
    camera_2.listen(lambda image: sensor_callback(image, sensor_queue, "camera_2"))
    
    vehicle_1.set_autopilot(True)
    vehicle_2.set_autopilot(True)
    
    traffic_manager.set_path(vehicle_1, route_1)
    traffic_manager.set_path(vehicle_2, route_2)

    filename = 'data' # TODO

    f = h5py.File(f'runs/{filename}.hdf5', 'w')
    sensors_group = f.create_group('metadata') # TODO
    sensors_group = f.create_group('sensors')
    state_group = f.create_group('state')

    curr_frame = 0

    while curr_frame < n_frames:
        world.tick()
        process_transform(camera_1.get_transform(), transforms_1)
        process_transform(camera_2.get_transform(), transforms_2)
        try:
            for _ in range(len(sensor_list)):
                data = sensor_queue.get(True, 1.0)
                if data[1] == 'camera_1':
                    process_img(data[0], images_1)
                else:
                    process_img(data[0], images_2)
        except Empty:
            print("Some of the sensor information is missed")
        curr_frame += 1

except KeyboardInterrupt:
        print("\nCancelled by user")
finally:
    camera_1.stop()
    camera_2.stop()
    world.apply_settings(original_settings)
    print("Saving data to HDF5 file")
    sensors_group.create_dataset('vehicle_1', data=images_1)
    sensors_group.create_dataset('vehicle_2', data=images_2)
    state_group.create_dataset('vehicle_1', data=transforms_1)
    state_group.create_dataset('vehicle_2', data=transforms_2)
    print("Destroying actors")
    for vehicle in vehicle_list:
        vehicle.destroy()
    for sensor in sensor_list:
        sensor.destroy()
    f.close()
    print("Done")
