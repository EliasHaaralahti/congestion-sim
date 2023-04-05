"""Helper script for visualizing vehicle and pedestrian spawn points."""
import time
import carla

def main():
    """
    A main loop that is responsible for creating the CARLA environment and visualizing pedestrian
    and vehicle spawn points. Vehicle spawn points are visualized in red and pedestrian spawn points
    are visualized in blue. The location of the spectator is printed to the terminal every second.
    """
    try:
        client = carla.Client('localhost', 2000)

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
