import json
import random
import math
from typing import Optional
from datetime import datetime
import carla

class CarlaEnv:
    def __init__(self, host: str, port: int, img_width: int, img_height: int, n_frames: int,
                 fps: int):
        self.vehicle_list = []
        self.sensor_list = []
        self.ai_controller_list = []
        self.pedestrian_list = []
        self.intersections = []
        self.images = {}
        self.transforms = {}
        self.velocities = {}
        self.travelled_frames = {}
        self.destination_indices = {}
        self.reached_destination = {}
        self.vehicle_dimensions = {}
        self.pedestrian_dimensions = {}
        self.img_width = img_width
        self.img_height = img_height
        self.n_frames = n_frames
        self.fps = fps
        self.client = carla.Client(host, port)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager()
        self.blueprint_library = self.world.get_blueprint_library()
        self.map = self.world.get_map()
        self.spawn_points = self.map.get_spawn_points()
        self.camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        self.camera_bp.set_attribute('image_size_x', f'{self.img_width}')
        self.camera_bp.set_attribute('image_size_y', f'{self.img_height}')

    def set_sync_mode(self) -> None:
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.fps
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(True)

    def generate_spawn_points(self, n_points: int) -> list:
        spawn_points = []
        for _ in range(n_points):
            spawn_points.append(self.world.get_random_location_from_navigation())
        return spawn_points

    def spawn_pedestrians_to_points(self, spawn_point_indices: list) -> None:
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        spawn_points = self.generate_spawn_points(300)
        for i in spawn_point_indices:
            walker_bp = random.choice(self.blueprint_library.filter('walker.*'))
            transform = carla.Transform(spawn_points[i])
            pedestrian = self.world.try_spawn_actor(walker_bp, transform)
            if pedestrian is None:
                continue
            self.pedestrian_list.append(pedestrian)
            pedestrian_id = f'pedestrian_{len(self.pedestrian_list)}'
            self.pedestrian_dimensions[pedestrian_id] = self.process_dimensions(
                pedestrian.bounding_box)
            ai_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(),
                                                   attach_to=pedestrian)
            self.ai_controller_list.append(ai_controller)
        self.world.tick()

    def spawn_pedestrians(self, n_pedestrians: int) -> None:
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        for _ in range(n_pedestrians):
            walker_bp = random.choice(self.blueprint_library.filter('walker.*'))
            transform = carla.Transform(self.world.get_random_location_from_navigation())
            pedestrian = self.world.try_spawn_actor(walker_bp, transform)
            if pedestrian is None:
                continue
            self.pedestrian_list.append(pedestrian)
            pedestrian_id = f'pedestrian_{len(self.pedestrian_list)}'
            self.pedestrian_dimensions[pedestrian_id] = self.process_dimensions(
                pedestrian.bounding_box)
            ai_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(),
                                                   attach_to=pedestrian)
            self.ai_controller_list.append(ai_controller)
        self.world.tick()

    def move_pedestrians(self) -> None:
        for ai_controller in self.ai_controller_list:
            ai_controller.start()
            ai_controller.go_to_location(self.world.get_random_location_from_navigation())
            ai_controller.set_max_speed(1 + random.random())

    def process_dimensions(self, bounding_box: object) -> tuple:
        dimensions = (bounding_box.extent.x * 2, bounding_box.extent.y * 2,
                      bounding_box.extent.z * 2)
        return dimensions

    def spawn_vehicle(self, vehicle_id: int, spawn_point_id: Optional[int]=None) -> object:
        if spawn_point_id is None:
            spawn_point = random.choice(self.spawn_points)
        else:
            spawn_point = self.spawn_points[spawn_point_id]
        vehicle_bp = self.blueprint_library.find(vehicle_id)
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle_list.append(vehicle)
        vehicle_id = f'vehicle_{len(self.vehicle_list)}'
        self.transforms[vehicle_id] = []
        self.velocities[vehicle_id] = []
        self.vehicle_dimensions[vehicle_id] = self.process_dimensions(vehicle.bounding_box)
        self.travelled_frames[vehicle_id] = 0
        self.reached_destination[vehicle_id] = False
        return vehicle

    def add_intersection(self, x_coord: int, y_coord: int, z_coord: int=0) -> object:
        intersection = carla.Location(x_coord, y_coord, z_coord)
        self.intersections.append(intersection)
        return intersection

    def is_in_intersection(self, vehicle: object, intersection: object,
                           max_distance: int=50) -> bool:
        vehicle_location = vehicle.get_location()
        if intersection.distance(vehicle_location) < max_distance:
            return True
        return False

    def get_avg_velocity(self, intersection: object) -> float:
        velocities = 0
        n_vehicles = 0
        for vehicle in self.vehicle_list:
            if self.is_in_intersection(vehicle, intersection):
                if vehicle.get_traffic_light_state() == carla.TrafficLightState.Green:
                    n_vehicles += 1
                    velocity = math.hypot(vehicle.get_velocity().x, vehicle.get_velocity().y) * 3.6
                    velocities += velocity
        avg_velocity = velocities / n_vehicles if n_vehicles > 0 else 0
        return avg_velocity

    def spawn_vehicles(self, n_vehicles: int) -> None:
        for _ in range(n_vehicles):
            vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
            spawn_point = random.choice(self.spawn_points)
            vehicle = self.world.try_spawn_actor(vehicle_bp, spawn_point)
            if vehicle is None:
                continue
            self.vehicle_list.append(vehicle)
            vehicle_id = f'vehicle_{len(self.vehicle_list)}'
            self.transforms[vehicle_id] = []
            self.velocities[vehicle_id] = []
            self.vehicle_dimensions[vehicle_id] = self.process_dimensions(vehicle.bounding_box)
            self.travelled_frames[vehicle_id] = 0
            self.reached_destination[vehicle_id] = False

    def create_transform(self, location_tuple: tuple, rotation_tuple: tuple) -> object:
        location = carla.Location(location_tuple[0], location_tuple[1], location_tuple[2])
        rotation = carla.Rotation(rotation_tuple[0], rotation_tuple[1], rotation_tuple[2])
        return carla.Transform(location, rotation)

    def spawn_camera(self, location_tuple: tuple, rotation_tuple: tuple=(0, 0, 0),
                     vehicle: Optional[object]=None) -> object:
        transform = self.create_transform(location_tuple, rotation_tuple)
        camera = self.world.spawn_actor(self.camera_bp, transform, attach_to=vehicle)
        self.sensor_list.append(camera)
        camera_id = f'camera_{len(self.sensor_list)}'
        self.images[camera_id] = []
        return camera

    def set_autopilot(self) -> None:
        for vehicle in self.vehicle_list:
            vehicle.set_autopilot(True)

    def set_route(self, vehicle: object, route_indices: list) -> None:
        vehicle_id = f'vehicle_{self.vehicle_list.index(vehicle)+1}'
        self.destination_indices[vehicle_id] = route_indices[-1]
        route = []
        for i in route_indices:
            route.append(self.spawn_points[i].location)
        self.traffic_manager.set_path(vehicle, route)

    def increment_travelled_frames(self) -> None:
        for i, vehicle in enumerate(self.vehicle_list):
            vehicle_id = f'vehicle_{i+1}'
            if vehicle_id not in self.destination_indices:
                continue
            destination_index = self.destination_indices[vehicle_id]
            destination = self.spawn_points[destination_index].location
            destination_x = round(destination.x)
            destination_y = round(destination.y)
            vehicle_x = round(vehicle.get_location().x)
            vehicle_y = round(vehicle.get_location().y)
            if vehicle_x == destination_x and vehicle_y == destination_y:
                self.reached_destination[vehicle_id] = True
            if not self.reached_destination[vehicle_id]:
                self.travelled_frames[vehicle_id] += 1

    def calculate_travel_times(self) -> dict:
        travel_times = {}
        for vehicle_id, travelled_frames in self.travelled_frames.items():
            travel_times[vehicle_id] = travelled_frames / self.fps
        return travel_times

    def get_vehicle_information(self) -> list:
        vehicle_information = []
        for i, vehicle in enumerate(self.vehicle_list):
            vehicle_dict = {}
            vehicle_id = f'vehicle_{i+1}'
            vehicle_dict['id'] = vehicle_id
            vehicle_dict['model'] = vehicle.type_id
            vehicle_dict['width'] = self.vehicle_dimensions[vehicle_id][1]
            vehicle_dict['length'] = self.vehicle_dimensions[vehicle_id][0]
            vehicle_dict['height'] = self.vehicle_dimensions[vehicle_id][2]
            vehicle_information.append(vehicle_dict)
        return vehicle_information

    def get_sensor_information(self) -> list:
        sensor_information = []
        vehicle_ids = [vehicle.id for vehicle in self.vehicle_list]
        for i, sensor in enumerate(self.sensor_list):
            sensor_dict = {}
            camera_id = f'camera_{i+1}'
            sensor_dict['id'] = camera_id
            if sensor.parent is None:
                sensor_dict['parent_id'] = None
                transform = sensor.get_transform()
                location = {
                    'x': transform.location.x,
                    'y': transform.location.y,
                    'z': transform.location.z
                }
                sensor_dict['location'] = location
            else:
                index = vehicle_ids.index(sensor.parent.id)
                parent_id = f'vehicle_{index+1}'
                sensor_dict['parent_id'] = parent_id
            sensor_information.append(sensor_dict)
        return sensor_information

    def get_pedestrian_information(self) -> list:
        pedestrian_information = []
        for i in range(len(self.pedestrian_list)):
            pedestrian_dict = {}
            pedestrian_id = f'pedestrian_{i+1}'
            pedestrian_dict['id'] = pedestrian_id
            pedestrian_dict['width'] = self.pedestrian_dimensions[pedestrian_id][1]
            pedestrian_dict['length'] = self.pedestrian_dimensions[pedestrian_id][0]
            pedestrian_dict['height'] = self.pedestrian_dimensions[pedestrian_id][2]
            pedestrian_information.append(pedestrian_dict)
        return pedestrian_information

    def get_intersection_information(self) -> list:
        intersection_information = []
        for i, intersection in enumerate(self.intersections):
            intersection_id = f'intersection_{i+1}'
            intersection_dict = {}
            intersection_dict['id'] = intersection_id
            location = {
                    'x': intersection.x,
                    'y': intersection.y,
                }
            intersection_dict['location'] = location
            intersection_information.append(intersection_dict)
        return intersection_information

    def generate_waypoints(self) -> list:
        waypoints = []
        waypoints_list = self.map.generate_waypoints(1.0)
        for waypoint in waypoints_list:
            location = waypoint.transform.location
            waypoint_tuple = (location.x, location.y)
            waypoints.append(waypoint_tuple)
        return waypoints

    def create_metadata(self) -> dict:
        metadata = {
            'timestamp': str(datetime.now()),
            'map': self.map.name,
            'waypoints': self.generate_waypoints(),
            'img_width': self.img_width,
            'img_height': self.img_height,
            'n_frames': self.n_frames,
            'fps': self.fps,
            'n_vehicles': len(self.vehicle_list),
            'n_sensors': len(self.sensor_list),
            'n_pedestrians': len(self.pedestrian_list),
            'vehicles': self.get_vehicle_information(),
            'sensors': self.get_sensor_information(),
            'intersections': self.get_intersection_information()
        }
        return json.dumps(metadata)

    def clear_actors(self) -> None:
        for sensor in self.sensor_list:
            sensor.destroy()
        for vehicle in self.vehicle_list:
            vehicle.destroy()
        for ai_controller in self.ai_controller_list:
            ai_controller.destroy()
        for pedestrian in self.pedestrian_list:
            pedestrian.destroy()

    def set_original_settings(self) -> None:
        self.world.apply_settings(self.original_settings)
