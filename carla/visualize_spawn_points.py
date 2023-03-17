import time
import carla

def main():
    try:
        client = carla.Client('192.168.0.114', 2000)

        world = client.get_world()
        spectator = world.get_spectator()

        spawn_points = world.get_map().get_spawn_points()
        pedestrian_spawn_points = []

        for _ in range(300):
            pedestrian_spawn_points.append(world.get_random_location_from_navigation())

        while True:
            world.wait_for_tick()
            for i, spawn_point in enumerate(spawn_points):
                world.debug.draw_string(spawn_point.location, str(i), life_time=1)
            for i, spawn_point in enumerate(pedestrian_spawn_points):
                world.debug.draw_string(spawn_point, str(i), color=carla.Color(0, 0, 255),
                                        life_time=1)
            print(spectator.get_transform())
            time.sleep(1)

    except KeyboardInterrupt:
        print('\nCancelled by user')

if __name__ == '__main__':
    main()
    