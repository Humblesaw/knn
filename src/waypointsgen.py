import math
import random
from typing import List, Tuple

class RouteGenerator:
    def __init__(self, start_lat: float, start_lon: float, start_alt: float, start_heading_deg: float):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.start_alt = start_alt
        self.start_heading = start_heading_deg
        
        # Approximation constants (from your WaypointTask)
        self.FT_PER_DEG_LAT = 364000.0

    def generate_route(self, num_waypoints: int, difficulty: float) -> List[Tuple[float, float, float]]:
        """
        Generates a 3D route.
        :param difficulty: 0.0 (straight line, level) to 1.0 (sharp turns, high elevation changes)
        """
        # Clamp difficulty
        difficulty = max(0.0, min(1.0, difficulty))
        
        waypoints = []
        current_lat = self.start_lat
        current_lon = self.start_lon
        current_alt = self.start_alt
        current_heading_rad = math.radians(self.start_heading)

        # Difficulty scalers
        # Easy: waypoints are far apart (time to react), Hard: closer together
        # Easy: waypoints are 5000 ft apart. Hard: 3000 ft apart.
        base_dist_ft = 5000.0 - (difficulty * 2000.0)
        
        # Easy: 0 deg turns, Hard: up to 30 deg turns
        max_turn_rad = math.radians(20.0 * difficulty)
        
        # Easy: 0 ft alt change, Hard: up to 1000 ft alt change per waypoint
        max_alt_change_ft = 100.0 * difficulty

        for _ in range(num_waypoints):
            # 1. Determine distance to next waypoint
            dist_ft = base_dist_ft
            
            # 2. Determine heading change
            turn_rad = random.uniform(-max_turn_rad, max_turn_rad)
            current_heading_rad += turn_rad
            
            # 3. Determine altitude change
            alt_change = random.uniform(-max_alt_change_ft, max_alt_change_ft)
            current_alt += alt_change
            
            # Prevent flying into the ground (assuming 2000 ft is safe floor)
            current_alt = max(2000.0, current_alt)

            # 4. Calculate new Lat/Lon using flat-earth approximation
            # lat_dist = distance * cos(heading)
            # lon_dist = distance * sin(heading)
            lat_dist_ft = dist_ft * math.cos(current_heading_rad)
            lon_dist_ft = dist_ft * math.sin(current_heading_rad)
            
            lat_change_deg = lat_dist_ft / self.FT_PER_DEG_LAT
            # Longitude shrinks as we move away from equator
            lon_change_deg = lon_dist_ft / (math.cos(math.radians(current_lat)) * self.FT_PER_DEG_LAT)
            
            current_lat += lat_change_deg
            current_lon += lon_change_deg
            
            waypoints.append((current_lat, current_lon, current_alt))

        return waypoints