import numpy as np
import matplotlib.pyplot as plt

def generate_hexagonal_grid(N):
    """
    Generate a structured grid of points in the first Brillouin zone of a hexagonal lattice.

    Parameters:
    N (int): The grid size (number of points along one direction).

    Returns:
    np.array: Array of points in the Brillouin zone.
    """
    a1 = np.array([1, 0])
    a2 = np.array([np.cos(np.pi / 3), np.sin(np.pi / 3)])

    points = []
    for i in range(-N, N):
        for j in range(-N, N):
            point = i * a1 + j * a2
            if np.linalg.norm(point) <= 1:
                points.append(point)

    return np.array(points)

def find_closest_point_index(point, points):
    """
    Find the index of the point in 'points' that is closest to 'point'.

    Parameters:
    point (np.array): A point in 2D space.
    points (np.array): Array of points.

    Returns:
    int: Index of the closest point in 'points'.
    """
    distances = np.linalg.norm(points - point, axis=1)
    return np.argmin(distances)

def rotation_60_deg(point):
    """
    Apply a 60-degree rotation to a point.

    Parameters:
    point (np.array): A point in 2D space.

    Returns:
    np.array: The rotated point.
    """
    rotation_matrix = np.array([[np.cos(np.pi / 3), -np.sin(np.pi / 3)],
                                [np.sin(np.pi / 3), np.cos(np.pi / 3)]])
    return np.dot(rotation_matrix, point)

# Generating a structured grid in the BZ
N_grid = 10  # Number of points along one direction
bz_points_structured = generate_hexagonal_grid(N_grid)

# Creating the permutation matrix for the 60-degree rotation
permutation_matrix_60_structured = np.zeros((len(bz_points_structured), len(bz_points_structured)), dtype=int)
for i, point in enumerate(bz_points_structured):
    rotated_point = rotation_60_deg(point)
    j = find_closest_point_index(rotated_point, bz_points_structured)
    permutation_matrix_60_structured[i, j] = 1

# Verifying if the permutation matrix is correct (permutation_matrix_60^6 should be the identity matrix)
identity_check_structured = np.linalg.matrix_power(permutation_matrix_60_structured, 6)

# Displaying the results
print("Permutation Matrix for 60-degree rotation:")
print(permutation_matrix_60_structured)
print("\nIdentity Check (Permutation Matrix ^ 6):")
print(identity_check_structured)
