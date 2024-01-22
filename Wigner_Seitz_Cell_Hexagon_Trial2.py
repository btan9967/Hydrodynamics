import matplotlib.pyplot as plt
import numpy as np

def hexagon(center, size):
    """Generate the vertices of a hexagon given a center and a size."""
    angle = np.linspace(0, 2*np.pi, 7)
    x_hexagon = center[0] + size * np.cos(angle)
    y_hexagon = center[1] + size * np.sin(angle)
    return x_hexagon, y_hexagon

def plot_corrected_hexagons(centers, size):
    """Plot a series of hexagons and the correctly ordered primitive cell."""
    fig, ax = plt.subplots()
    primitive_cell_x = []
    primitive_cell_y = []

    # Plot each hexagon and collect the centers
    for center in centers:
        x_hex, y_hex = hexagon(center, size)
        ax.plot(x_hex, y_hex, 'b-')
        primitive_cell_x.append(center[0])
        primitive_cell_y.append(center[1])
        ax.plot(center[0], center[1], 'ro')  # Mark the center

    # Reorder centers to form the correct primitive cell
    reordered_centers = [centers[0], centers[1], centers[3], centers[2]]
    primitive_cell_x = [c[0] for c in reordered_centers] + [reordered_centers[0][0]]
    primitive_cell_y = [c[1] for c in reordered_centers] + [reordered_centers[0][1]]

    # Connect the centers to form the primitive cell
    ax.plot(primitive_cell_x, primitive_cell_y, 'r--')

    ax.set_aspect('equal', adjustable='box')
    plt.title('Corrected Honeycomb Brillouin Zone with Primitive Cell')
    plt.show()
    
# Define the centers of the four hexagons
hexagon_centers = [(0, 0), (1.5, np.sqrt(3)/2), (1.5, -np.sqrt(3)/2), (3, 0)]
hexagon_size = 1  # Set the size of the hexagons

# Plot the hexagons and the corrected primitive cell
plot_corrected_hexagons(hexagon_centers, hexagon_size)
