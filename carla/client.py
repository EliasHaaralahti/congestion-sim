from environment import CarlaEnv
from recorder import Recorder

def main():
    try:
        host = '192.168.0.114'
        port = 2000
        img_width = 640
        img_height = 640
        n_frames = 1000
        fps = 60

        env = CarlaEnv(host, port, img_width, img_height, n_frames, fps)
        recorder = Recorder(img_width, img_height)

        env.set_sync_mode()

        env.spawn_pedestrians(spawn_point_indices=[1, 65, 24, 46])
        env.move_pedestrians()

        vehicle_1 = env.spawn_vehicle('vehicle.tesla.model3', 34)
        vehicle_2 = env.spawn_vehicle('vehicle.audi.tt', 129)
        vehicle_3 = env.spawn_vehicle('vehicle.audi.a2', 28)
        vehicle_4 = env.spawn_vehicle('vehicle.nissan.micra', 116)

        camera_1 = env.spawn_camera(vehicle_1, (0, 0, 2))
        camera_2 = env.spawn_camera(vehicle_2, (0, 0, 2))
        camera_3 = env.spawn_camera(vehicle_3, (0, 0, 2))
        camera_4 = env.spawn_camera(vehicle_4, (0, 0, 2))
        camera_5 = env.spawn_camera(vehicle_4, (0, 0, 2), (0, 180, 0))

        camera_1.listen(lambda image: recorder.sensor_callback(image, 'camera_1'))
        camera_2.listen(lambda image: recorder.sensor_callback(image, 'camera_2'))
        camera_3.listen(lambda image: recorder.sensor_callback(image, 'camera_3'))
        camera_4.listen(lambda image: recorder.sensor_callback(image, 'camera_4'))
        camera_5.listen(lambda image: recorder.sensor_callback(image, 'camera_5'))

        env.set_autopilot()

        env.set_route(vehicle_1, route_indices=[35, 79])
        env.set_route(vehicle_2, route_indices=[28, 124, 30, 31])
        env.set_route(vehicle_3, route_indices=[124, 30, 31, 33])
        env.set_route(vehicle_4, route_indices=[27, 122, 25])

        curr_frame = 0

        while curr_frame < n_frames:
            env.world.tick()
            env.increment_travelled_frames()
            recorder.process_transforms(env.vehicle_list, env.transforms)
            recorder.process_velocities(env.vehicle_list, env.velocities)
            recorder.process_images(env.sensor_list, env.images)
            curr_frame += 1

    except KeyboardInterrupt:
        print('\nCancelled by user')
    finally:
        metadata_json = env.create_metadata()
        env.set_original_settings()
        env.clear_actors()
        recorder.create_datasets(env.transforms, env.velocities, env.images, env.vehicle_list,
                                 env.sensor_list, metadata_json)
        recorder.stop_recording()

if __name__ == '__main__':
    main()
