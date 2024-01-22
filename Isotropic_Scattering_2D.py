import numpy as np
import matplotlib.pyplot as plt

def isotropic_scattering_matrices_symmetric_2d(n):
    # Define the projection operator
    def projection_operator(v):
        return np.outer(v, v)

    # Generate points and velocities on a unit circle
    angles = np.linspace(np.pi/n, 2 * np.pi + np.pi/n, n, endpoint=False)
    kpts = np.array([np.cos(angles), np.sin(angles)]).T
    vels = kpts.copy()

    # Compute the velocity projection
    orthogonal_basis = np.linalg.qr(np.vstack([np.ones(n), vels.T]).T)[0]
    velocity_projection = np.linalg.multi_dot([projection_operator(v) for v in orthogonal_basis.T])

    # Normalized energy and its projection
    normalized_energy = np.ones(n) / np.sqrt(n)
    energy_projection = projection_operator(normalized_energy)

    # Scattering matrix
    sm = np.diag(np.full(n, n-1)) / n - np.ones((n, n)) / n

    # Scattering matrices with relaxed and projected velocities
    sm_velocity_relaxed = np.linalg.multi_dot([energy_projection, np.diag(np.diag(sm)), energy_projection])
    sm_velocity_projected = np.linalg.multi_dot([velocity_projection, sm, velocity_projection])

    return kpts, vels, sm, sm_velocity_projected, sm_velocity_relaxed

def scaled_isotropic_scattering_matrix_2d(m, tau_MC):
    # Placeholder for meanLifetime function
    def mean_lifetime(matrix):
        return np.mean(matrix)

    kpts, vels, numeric, sMC, sMR = isotropic_scattering_matrices_symmetric_2d(m)
    alpha = 1 / tau_MC * mean_lifetime(sMC)
    scaled_sMC = alpha * sMC

    return kpts, vels, numeric, sMC, sMR, scaled_sMC

# Testing the functions
n = 48 # Number of states
tau_MC = 100  # Tau value for scaling

# Run the isotropic scattering matrix function
kpts, vels, sm, sm_velocity_projected, sm_velocity_relaxed = isotropic_scattering_matrices_symmetric_2d(n)

# Run the scaled isotropic scattering matrix function
scaled_results = scaled_isotropic_scattering_matrix_2d(n, tau_MC)

# Plotting
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Scattering Matrix
axs[0].imshow(sm, cmap='viridis', interpolation='nearest')
axs[0].set_title('Scattering Matrix')
axs[0].set_xlabel('States')
axs[0].set_ylabel('States')

# Scaled Scattering Matrix
axs[1].imshow(scaled_results[5], cmap='viridis', interpolation='nearest')
axs[1].set_title('Scaled Scattering Matrix')
axs[1].set_xlabel('States')
axs[1].set_ylabel('States')

plt.tight_layout()
plt.show()

