from environment import CarlaEnv
from recorder import Recorder

def main():
    try:
        img_width = 640
        img_height = 480
        n_frames = 800
        fps = 60

        env = CarlaEnv('192.168.0.114', 2000, img_width, img_height, n_frames, fps)
        recorder = Recorder(img_width, img_height)
        
        env.set_sync_mode()

        vehicle_1 = env.spawn_vehicle('vehicle.tesla.model3', 34)
        vehicle_2 = env.spawn_vehicle('vehicle.tesla.cybertruck', 129)

        camera_1 = env.spawn_camera(0, 0, 2, vehicle_1)
        camera_2 = env.spawn_camera(1, 0, 2.5, vehicle_2)

        camera_1.listen(lambda image: recorder.sensor_callback(image, "camera_1"))
        camera_2.listen(lambda image: recorder.sensor_callback(image, "camera_2"))

        env.set_autopilot()

        env.set_route(vehicle_1, route_indices=[35, 79])
        env.set_route(vehicle_2, route_indices=[28, 79])

        metadata_json = env.create_metadata()

        curr_frame = 0

        while curr_frame < n_frames:
            env.world.tick()
            recorder.process_transforms(env.vehicle_list, env.transforms)
            recorder.process_images(env.sensor_list, env.images)
            curr_frame += 1

    except KeyboardInterrupt:
            print("\nCancelled by user")
    finally:
        env.set_original_settings()
        env.clear_actors()
        recorder.create_datasets(env.transforms, env.images, env.vehicle_list, env.vehicle_list, metadata_json)
        recorder.stop_recording()

if __name__ == "__main__":
    main()