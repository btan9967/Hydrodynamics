import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

def hexagon_vertices(a):
    """ Generate the vertices of a hexagon centered at the origin with a given side length a. """
    angle = np.linspace(0, 2*np.pi, 7)
    x = a * np.cos(angle)
    y = a * np.sin(angle)
    return np.vstack((x, y)).T

def inside_polygon(p, vertices):
    """ Check if a point p is inside a polygon defined by vertices. """
    return plt.contains_point(p)

def generate_uniform_mesh(vertices, density=8):
    """ Generate a uniform mesh of points within a hexagonal Wigner-Seitz cell. """
    # Extracting the boundary coordinates
    xmin, xmax = np.min(vertices[:, 0]), np.max(vertices[:, 0])
    ymin, ymax = np.min(vertices[:, 1]), np.max(vertices[:, 1])

    # Creating a grid of points
    x = np.linspace(xmin, xmax, density)
    y = np.linspace(ymin, ymax, density)
    X, Y = np.meshgrid(x, y)
    points = np.vstack((X.ravel(), Y.ravel())).T

    # Selecting points inside the hexagon
    hexagon = plt.Polygon(vertices)
    inside_points = [p for p in points if hexagon.contains_point(p)]

    return np.array(inside_points)

# Side length of the hexagon (you can adjust this based on your lattice parameters)
a = 1

# Generate vertices of the hexagon
vertices = hexagon_vertices(a)

# Generate mesh
mesh_points = generate_uniform_mesh(vertices)

# Plotting for visualization
plt.figure(figsize=(8, 6))
plt.scatter(mesh_points[:, 0], mesh_points[:, 1], s=10, color='blue')
plt.gca().add_patch(Polygon(vertices, fill=None, edgecolor='r'))
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Uniform Mesh in the Wigner-Seitz Cell of a Hexagonal Lattice")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
