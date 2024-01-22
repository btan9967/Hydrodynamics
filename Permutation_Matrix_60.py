import numpy as np
import matplotlib.pyplot as plt

def rotation_matrix(theta):
    """ Create a rotation matrix for a given angle theta (in radians). """
    return np.array([[np.cos(theta), -np.sin(theta)], 
                     [np.sin(theta), np.cos(theta)]])

def rotate_points(points, theta):
    """ Apply rotation to a set of points. """
    return np.dot(points, rotation_matrix(theta).T)

def create_symmetric_mesh(vertices, rows, cols):
    """ Create a symmetric mesh on the parallelogram that respects the 60-degree rotational symmetry. """
    # Calculate vectors for the parallelogram sides
    side_vector1 = (vertices[1] - vertices[0]) / cols
    side_vector2 = (vertices[3] - vertices[0]) / rows

    # Generate points in the parallelogram
    mesh_points = []
    for i in range(rows + 1):
        for j in range(cols + 1):
            point = vertices[0] + i * side_vector2 + j * side_vector1
            mesh_points.append(point)

    return np.array(mesh_points)

# Define the vertices of the parallelogram (Wigner-Seitz cell of a hexagonal lattice)
parallelogram_vertices = np.array([[0, 0], [1, 0], [1.5, np.sqrt(3)/2], [0.5, np.sqrt(3)/2]])

# Define the number of rows and columns for the mesh
rows, cols = 3, 3  # This will create a mesh of 4x4 points

# Create the symmetric mesh within the parallelogram
symmetric_mesh_points = create_symmetric_mesh(parallelogram_vertices, rows, cols)

# Define the rotation angle (60 degrees in radians)
theta = np.pi / 3  # 60 degrees

# Apply a 60-degree rotation to the mesh points
rotated_symmetric_mesh_points = rotate_points(symmetric_mesh_points, theta)

# Construct the permutation matrix for the 60-degree rotation
n = len(symmetric_mesh_points)
permutation_matrix = np.zeros((n, n))

for i, original_point in enumerate(symmetric_mesh_points):
    for j, rotated_point in enumerate(rotated_symmetric_mesh_points):
        if np.allclose(original_point, rotated_point, atol=1e-2):  # Allowing a tolerance for floating-point errors
            permutation_matrix[j, i] = 1

# Plotting for visualization
plt.figure(figsize=(8, 6))
plt.scatter(symmetric_mesh_points[:, 0], symmetric_mesh_points[:, 1], color='blue', label='Original Mesh')
plt.scatter(rotated_symmetric_mesh_points[:, 0], rotated_symmetric_mesh_points[:, 1], color='green', label='Rotated Mesh (60 degrees)')
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Symmetric Mesh and Rotated Mesh in a Parallelogram")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# Print the permutation matrix
print("Permutation Matrix for the 60-Degree Rotation:")
print(permutation_matrix)
