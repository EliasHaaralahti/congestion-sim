import carla
import json
import random

class CarlaEnv:
    def __init__(self, host, port, img_width, img_height, n_frames, fps):
        self.vehicle_list = []
        self.sensor_list = []
        self.images = {}
        self.transforms = {}
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

    def set_sync_mode(self):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / self.fps
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(True)

    def spawn_vehicle(self, vehicle_id, spawn_point_id=None):
        if spawn_point_id is None:
            spawn_point = random.choice(self.spawn_points)
        else:
            spawn_point = self.spawn_points[spawn_point_id]
        vehicle_bp = self.blueprint_library.find(vehicle_id)
        vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
        self.vehicle_list.append(vehicle)
        vehicle_id = f'vehicle_{len(self.vehicle_list)}'
        self.transforms[vehicle_id] = []
        return vehicle
    
    def create_transform(self, x, y, z):
        return carla.Transform(carla.Location(x, y, z))
    
    def spawn_camera(self, x, y, z, vehicle):
        transform = self.create_transform(x, y, z)
        camera = self.world.spawn_actor(self.camera_bp, transform, attach_to=vehicle)
        self.sensor_list.append(camera)
        camera_id = f'camera_{len(self.sensor_list)}'
        self.images[camera_id] = []
        return camera

    def set_autopilot(self):
        for vehicle in self.vehicle_list:
            vehicle.set_autopilot(True)

    def set_route(self, vehicle, route_indices):
        route = []
        for i in route_indices:
            route.append(self.spawn_points[i].location)
        self.traffic_manager.set_path(vehicle, route)

    def create_metadata(self):
        metadata = {
            "img_width": self.img_width,
            "img_height": self.img_height,
            "n_frames": self.n_frames,
            "fps": self.fps
        }
        return json.dumps(metadata)

    def clear_actors(self):
        for sensor in self.sensor_list:
            sensor.destroy()
        for vehicle in self.vehicle_list:
            vehicle.destroy()

    def set_original_settings(self):
        self.world.apply_settings(self.original_settings)