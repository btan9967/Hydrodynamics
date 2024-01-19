import numpy as np
import pandas as pd
from scipy.stats import norm, uniform
import matplotlib.pyplot as plt

def generate_permutation_matrices(N):
    """
    Generate permutation matrices for a hexagonal lattice.
    :param N: The size of the matrices (number of discrete states)
    :return: A dictionary of permutation matrices
    """
    if N % 6 != 0:
        raise ValueError("N must be divisible by 6 for six-fold rotation symmetry.")

    R = np.zeros((N, N), dtype=int)
    M = np.zeros((N, N), dtype=int)
    I = np.zeros((N, N), dtype=int)

    for i in range(N):
        R[i, (i + N // 6) % N] = 1
        M[i, N - i - 1] = 1
        I[i, (N - i) % N] = 1

    return {"R": R, "M": M, "I": I}

def construct_centralizers(perm_matrices, N):
    """
    Construct centralizers for permutation matrices.
    :param perm_matrices: Dictionary of permutation matrices
    :param N: Size of the matrices
    :return: Dictionary of centralizers
    """
    centralizers = {}
    for key, P in perm_matrices.items():
        centralizer = []
        for X in [np.eye(N)] + [np.roll(np.eye(N), i, axis=1) for i in range(1, N)]:
            if np.array_equal(np.dot(P, X), np.dot(X, P)):
                centralizer.append(X)
        centralizers[key] = centralizer

    return centralizers

def find_intersection_of_centralizers(centralizers):
    """
    Find the intersection of centralizers across permutation matrices.
    :param centralizers: Dictionary of centralizers
    :return: List of matrices in the intersection
    """
    for key in centralizers:
        centralizers[key] = [tuple(matrix.flatten()) for matrix in centralizers[key]]

    intersection = set(centralizers[list(centralizers.keys())[0]])
    for matrices in centralizers.values():
        intersection = intersection.intersection(set(matrices))

    return [np.array(matrix).reshape(N, N) for matrix in intersection]

def sample_scattering_matrix(intersection, N):
    """
    Sample independent components according to a normal distribution to yield a scattering matrix.
    :param intersection: List of matrices in the intersection of centralizers
    :param N: Size of the matrices
    :return: Random scattering matrix
    """
    R_0 = np.zeros((N, N))
    for matrix in intersection:
        R_0 += norm.rvs(size=(N, N)) * matrix

    return R_0

def enforce_positive_semidefiniteness(matrix):
    """
    Enforce positive semidefiniteness of a matrix.
    :param matrix: The matrix to be modified
    :return: Positive semi-definite matrix
    """
    U, Lambda, _ = np.linalg.svd(matrix)
    D = np.diag(uniform.rvs(size=Lambda.shape))

    R_0_PSD = np.dot(U, np.dot(D, U.T))
    return R_0_PSD

def construct_collision_operator(N):
    perm_matrices = generate_permutation_matrices(N)
    centralizers = construct_centralizers(perm_matrices, N)
    intersection = find_intersection_of_centralizers(centralizers)
    R_0 = sample_scattering_matrix(intersection, N)
    # R_0_PSD = enforce_positive_semidefiniteness(R_0)
    return R_0

def plot_matrix(matrix, title):
    """
    Plot a matrix as a heatmap for visualization.
    :param matrix: The matrix to be plotted
    :param title: Title of the plot
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.xlabel('States')
    plt.ylabel('States')
    plt.show()

# Example usage with N = 6
# Generate permutation matrices for N=6 and plot them
N = 48
collision_operator = construct_collision_operator(N)
print(collision_operator)
perm_matrices = generate_permutation_matrices(N)
for key, matrix in perm_matrices.items():
    plot_matrix(matrix, f"Permutation Matrix for {key}")

# Construct the collision operator and plot it
collision_operator = construct_collision_operator(N)
collision_operator_PSD = enforce_positive_semidefiniteness(collision_operator)

plot_matrix(collision_operator, "Random Collision Operator (R_0)")
plot_matrix(collision_operator_PSD, "Random Collision Operator (R_0^{PSD})")

df = pd.DataFrame(collision_operator)
df.to_csv('random_matrix.csv', index=False, header=False)

df_PSD = pd.DataFrame(collision_operator_PSD)
df_PSD.to_csv('random_matrix_PSD.csv', index=False, header=False)
