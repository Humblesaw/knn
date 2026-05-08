import math
import random
from typing import List, Tuple

class RouteGenerator:
    def __init__(self, start_lat: float, start_lon: float, start_alt: float, start_heading_deg: float):
        self.start_lat = start_lat
        self.start_lon = start_lon
        self.start_alt = start_alt
        self.start_heading = start_heading_deg
        
        # approximation constants
        self.FT_PER_DEG_LAT = 364000.0

    def generate_route(self, num_waypoints: int, difficulty: float) -> List[Tuple[float, float, float]]:
        """
        Generates a 3D route.
        :param difficulty: 0.0 (straight line, level) to 1.0 (sharp turns, high elevation changes)
        """
        # clamp difficulty
        difficulty = max(0.0, min(1.0, difficulty))
        
        waypoints = []
        current_lat = self.start_lat
        current_lon = self.start_lon
        current_alt = self.start_alt
        current_heading_rad = math.radians(self.start_heading)

        # difficulty scalers
        # easy: waypoints are far apart (time to react), hard: closer together
        base_dist_ft = 200.0 - (difficulty * 100.0)
        
        # easy: 0 deg turns, hard: up to 30 deg turns
        max_turn_rad = math.radians(30.0 * difficulty)
        
        # easy: 0 ft alt change, hard: up to 1000 ft alt change per waypoint
        max_alt_change_ft = 1000.0 * difficulty

        for _ in range(num_waypoints):
            # determine distance to next waypoint
            dist_ft = base_dist_ft
            
            # determine heading change
            turn_rad = random.uniform(-max_turn_rad, max_turn_rad)
            current_heading_rad += turn_rad
            
            # determine altitude change
            alt_change = random.uniform(-max_alt_change_ft, max_alt_change_ft)
            current_alt += alt_change
            
            # prevent flying into the ground (assuming 2000 ft is safe floor)
            current_alt = max(2000.0, current_alt)

            # calculate new Lat/Lon using flat-earth approximation
            # lat_dist = distance * cos(heading)
            # lon_dist = distance * sin(heading)
            lat_dist_ft = dist_ft * math.cos(current_heading_rad)
            lon_dist_ft = dist_ft * math.sin(current_heading_rad)
            
            lat_change_deg = lat_dist_ft / self.FT_PER_DEG_LAT
            # longitude shrinks as we move away from equator
            lon_change_deg = lon_dist_ft / (math.cos(math.radians(current_lat)) * self.FT_PER_DEG_LAT)
            
            current_lat += lat_change_deg
            current_lon += lon_change_deg
            
            waypoints.append((current_lat, current_lon, current_alt))

        return waypoints
