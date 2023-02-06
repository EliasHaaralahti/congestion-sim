import carla
import time
import numpy as np
import h5py

img_width = 1280
img_height = 720

def process_img(image, images):
    img_data = np.array(image.raw_data)
    reshaped_data = img_data.reshape((img_height, img_width, 4))[:, :, :3]
    images.append(reshaped_data)

actor_list = []
images_1 = []
images_2 = []

try:
    client = carla.Client('localhost', 2000)
    world = client.get_world()

    traffic_manager = client.get_trafficmanager()

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
    
    actor_list.append(vehicle_1)
    actor_list.append(vehicle_2)

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{img_width}')
    camera_bp.set_attribute('image_size_y', f'{img_height}')

    transform_1 = carla.Transform(carla.Location(x=0, z=2))
    transform_2 = carla.Transform(carla.Location(x=1, z=2.5))
    
    camera_1 = world.spawn_actor(camera_bp, transform_1, attach_to=vehicle_1)
    camera_2 = world.spawn_actor(camera_bp, transform_2, attach_to=vehicle_2)
    
    actor_list.append(camera_1)
    actor_list.append(camera_2)

    camera_1.listen(lambda image: process_img(image, images_1))
    camera_2.listen(lambda image: process_img(image, images_2))
    
    vehicle_1.set_autopilot(True)
    vehicle_2.set_autopilot(True)
    
    traffic_manager.set_path(vehicle_1, route_1)
    traffic_manager.set_path(vehicle_2, route_2)

    f = h5py.File('data.hdf5', 'w')
    sensors_group = f.create_group('sensors')
    
    time.sleep(10)

finally:
    print("Saving recordings")
    sensors_group.create_dataset('vehicle_1', data=images_1)
    sensors_group.create_dataset('vehicle_2', data=images_2)
    print("Destroying actors")
    for actor in actor_list:
        actor.destroy()
    f.close()
    print("Done")
