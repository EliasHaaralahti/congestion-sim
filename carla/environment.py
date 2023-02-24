import json
import random
from typing import Optional
import carla

class CarlaEnv:
    def __init__(self, host: str, port: int, img_width: int, img_height: int, n_frames: int,
                 fps: int):
        self.vehicle_list = []
        self.sensor_list = []
        self.ai_controller_list = []
        self.pedestrian_list = []
        self.images = {}
        self.transforms = {}
        self.velocities = {}
        self.img_width = img_width
        self.img_height = img_height
        self.n_frames = n_frames
        self.fps = fps
        self.client = carla.Client(host, port)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()
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

    def spawn_pedestrians(self, spawn_point_indices: list) -> None:
        walker_controller_bp = self.blueprint_library.find('controller.ai.walker')
        spawn_points = self.generate_spawn_points(200)
        for i in spawn_point_indices:
            walker_bp = random.choice(self.blueprint_library.filter('walker.*'))
            transform = carla.Transform(spawn_points[i])
            pedestrian = self.world.try_spawn_actor(walker_bp, transform)
            if pedestrian is None:
                continue
            self.pedestrian_list.append(pedestrian)
            ai_controller = self.world.spawn_actor(walker_controller_bp, carla.Transform(),
                                                   attach_to=pedestrian)
            self.ai_controller_list.append(ai_controller)
        self.world.tick()

    def move_pedestrians(self) -> None:
        for ai_controller in self.ai_controller_list:
            ai_controller.start()
            ai_controller.go_to_location(self.world.get_random_location_from_navigation())
            ai_controller.set_max_speed(1 + random.random())

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
        return vehicle

    def create_transform(self, x_coord: int, y_coord: int, z_coord: int) -> object:
        location = carla.Location(x_coord, y_coord, z_coord)
        return carla.Transform(location)

    def spawn_camera(self, x_coord: int, y_coord: int, z_coord: int, vehicle: object) -> object:
        transform = self.create_transform(x_coord, y_coord, z_coord)
        camera = self.world.spawn_actor(self.camera_bp, transform, attach_to=vehicle)
        self.sensor_list.append(camera)
        camera_id = f'camera_{len(self.sensor_list)}'
        self.images[camera_id] = []
        return camera

    def set_autopilot(self) -> None:
        for vehicle in self.vehicle_list:
            vehicle.set_autopilot(True)

    def set_route(self, vehicle: object, route_indices: list) -> None:
        route = []
        for i in route_indices:
            route.append(self.spawn_points[i].location)
        self.traffic_manager.set_path(vehicle, route)

    def create_metadata(self) -> dict:
        metadata = {
            'img_width': self.img_width,
            'img_height': self.img_height,
            'n_frames': self.n_frames,
            'fps': self.fps
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
