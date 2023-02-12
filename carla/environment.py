import carla

class CarlaEnv:
    def __init__(self, host='localhost', port=2000):
        self.vehicle_list = []
        self.sensor_list = []
        self.client = carla.Client(host, port)
        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        self.traffic_manager = self.client.get_trafficmanager()
        self.blueprint_library = self.world.get_blueprint_library()
        self.spawn_points = self.world.get_map().get_spawn_points()

    def set_sync_mode(self, fps):
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 1 / fps
        self.world.apply_settings(settings)
        self.traffic_manager.set_synchronous_mode(True)

    def set_autopilot(self):
        for vehicle in self.vehicle_list:
            vehicle.set_autopilot(True)

    def clear_actors(self):
        for sensor in self.sensor_list:
            sensor.destroy()
        for vehicle in self.vehicle_list:
            vehicle.destroy()

    def set_original_settings(self):
        self.world.apply_settings(self.original_settings)