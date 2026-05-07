import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from waypointsgen import RouteGenerator

def plot_route_3d(start_lat=51.4, start_lon=-2.3273, start_alt=5000, start_heading=0, num_waypoints=5, difficulty=0.5):
    """
    Generates and plots a 3D flight route using matplotlib.
    """
    # Initialize the generator
    generator = RouteGenerator(start_lat, start_lon, start_alt, start_heading)
    
    # Generate the route
    route = generator.generate_route(num_waypoints=num_waypoints, difficulty=difficulty)
    
    # Unpack the coordinates for plotting
    lats = [start_lat] + [wp[0] for wp in route]
    lons = [start_lon] + [wp[1] for wp in route]
    alts = [start_alt] + [wp[2] for wp in route]
    
    # Create the 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the path line
    ax.plot(lons, lats, alts, label='Flight Path', color='blue', marker='o', linestyle='dashed', linewidth=2, markersize=6)
    
    # Highlight the starting position
    ax.scatter(start_lon, start_lat, start_alt, color='green', s=100, label='Start Position')
    
    # Highlight the final destination
    ax.scatter(lons[-1], lats[-1], alts[-1], color='red', s=100, label='Final Waypoint')
    
    # Labels and Title
    ax.set_title(f'Generated 3D Flight Route (Difficulty: {difficulty})')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Altitude (ft)')
    ax.legend()
    
    plt.show()

# --- Test it out in your notebook ---
# Plot an easy, straight-line route
plot_route_3d(difficulty=0.0)

# Plot a hard, aggressive route
plot_route_3d(difficulty=0.5)
