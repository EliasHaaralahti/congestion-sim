"""Script for collecting image data for the congestion detection model."""
from environment import CarlaEnv
from recorder import Recorder

def main():
    """
    A main function that is responsible for creating the CARLA environment, spawning and
    moving vehicles and collecting image data using RSUs. Image data can only be collected
    at one intersection at a time. To collect data from the second intersection,
    comment the code related to the first intersection (location and cameras) and uncomment
    the corresponding code for the second intersection. The simulation is run for `n_frames`.
    """
    try:
        host = 'localhost'
        port = 2000
        img_width = 640
        img_height = 640
        n_frames = 500
        fps = 30
        filename = 'intersection_1_100_vehicles'
        n_vehicles = 100

        env = CarlaEnv(host, port, img_width, img_height, n_frames, fps)
        recorder = Recorder(img_width, img_height, filename)

        env.set_sync_mode()

        # First intersection location
        intersection = env.add_intersection(-47, 21)
        # Second intersection location
        # intersection = env.add_intersection(102, 21)

        env.spawn_pedestrians(50)
        env.move_pedestrians()

        env.spawn_vehicles(n_vehicles)

        # First intersection cameras
        camera_1 = env.spawn_camera((-62, 3, 20), (-37, 45, 0))
        camera_2 = env.spawn_camera((-31, 37, 20), (-42, -139, 0))
        # Second intersection cameras
        # camera_1 = env.spawn_camera((64, -1, 20), (-40, 31, 0))
        # camera_2 = env.spawn_camera((89, 39, 20), (-47, -63, 0))

        camera_1.listen(lambda image: recorder.sensor_callback(image, 'camera_1'))
        camera_2.listen(lambda image: recorder.sensor_callback(image, 'camera_2'))

        env.set_autopilot()

        curr_frame = 0

        while curr_frame < n_frames:
            env.world.tick()
            recorder.process_labels(env.get_avg_velocity(intersection), ratio=0.5)
            recorder.process_images(env.sensor_list, env.images)
            curr_frame += 1

    except KeyboardInterrupt:
        print('\nCancelled by user')
    finally:
        metadata_json = env.create_metadata()
        env.set_original_settings()
        env.clear_actors()
        recorder.create_datasets(env.transforms, env.velocities, env.images, env.vehicle_list,
                                 env.sensor_list, metadata_json, training=True)
        recorder.stop_recording()

if __name__ == '__main__':
    main()
