import folium

def visualize_cvrp_solution(city_coords, tour_indices):
    """
    Visualize CVRP solution using folium.

    Parameters:
    - city_coords: List of (lat, lon) tuples representing city coordinates.
    - tour_indices: List of indices representing the order in which cities are visited.

    Returns:
    - m: folium map object.
    """

    # Create a folium map centered around the first city
    m = folium.Map(location=city_coords[tour_indices[0]], zoom_start=13)

    # Add points for each city to the map
    for coord in city_coords:
        folium.Marker(coord).add_to(m)

    # Draw lines between cities to represent the tour
    for i in range(1, len(tour_indices)):
        start = city_coords[tour_indices[i-1]]
        end = city_coords[tour_indices[i]]
        folium.PolyLine([start, end], color="blue").add_to(m)

    return m

# Example usage:
city_coords = [(40.6401, 22.9444), (40.6514 , 22.91050 ),
               (40.61736 , 23.04640 ), (40.780 , 22.979343364 ),
               (40.7814 , 22.6765 )]  # Example coordinates for Thessaloniki
tour_indices = [0, 1, 2, 3, 4, 0]  # Starting and ending at the first city for simplicity
map_obj = visualize_cvrp_solution(city_coords, tour_indices)
map_obj.save("cvrp_solution.html")


def normalize_coords(coords):
    # Unpack coordinates
    lats, longs = zip(*coords)

    # Find the minimum and maximum for each dimension
    min_lat, max_lat = min(lats), max(lats)
    min_long, max_long = min(longs), max(longs)

    # Normalize each dimension independently
    normalized_lats = [(lat - min_lat) / (max_lat - min_lat) for lat in lats]
    normalized_longs = [(long - min_long) / (max_long - min_long) for long in longs]

    # Combine normalized coordinates
    normalized_coords = list(zip(normalized_lats, normalized_longs))

    return normalized_coords


normalized_city_coords = normalize_coords(city_coords)
tour_indices = [0, 1, 2, 3, 4, 0]  # Starting and ending at the first city for simplicity
map_obj = visualize_cvrp_solution(city_coords, tour_indices)
map_obj.save("normalized_cvrp_solution.html")