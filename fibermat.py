import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Set up the 3D grid
x = np.linspace(-5, 5, 50)  # in mm
y = np.linspace(-5, 5, 50)  # in mm
z = np.linspace(0, 5, 25)   # in mm (height of the mat and air above)
X, Y, Z = np.meshgrid(x, y, z)

# Simulate the fiber mat as a porous structure with a fibrous texture
# Use Perlin-like noise to mimic tangled fibers
fiber_density = np.random.rand(len(x), len(y), len(z))
fiber_density = gaussian_filter(fiber_density, sigma=2)  # Smooth the noise for a fibrous texture
fiber_density = np.clip(fiber_density, 0, 1)

# Define the cylindrical mat region (height 2 mm, radius 4 mm)
mat_radius = 4.0
mat_height = 2.0
mat_mask = (np.sqrt(X**2 + Y**2) <= mat_radius) & (Z <= mat_height)

# 1. Surface evaporation rate (highest at the top surface of the mat)
evaporation_rate = np.zeros_like(X)
evaporation_rate[mat_mask] = (1 - fiber_density[mat_mask]) * 0.015  # in µm/s, higher where fiber density is lower
evaporation_rate[~mat_mask] = 0  # No evaporation outside the mat

# 2. Air velocity field (flow driven by evaporation)
# Simplified radial and upward flow above the mat
U = np.zeros_like(X)
V = np.zeros_like(X)
W = np.zeros_like(X)
above_mat = (np.sqrt(X**2 + Y**2) <= mat_radius) & (Z > mat_height)
U[above_mat] = -Y[above_mat] / (np.sqrt(X[above_mat]**2 + Y[above_mat]**2 + 1e-2)) * 0.05  # Radial component
V[above_mat] = X[above_mat] / (np.sqrt(X[above_mat]**2 + Y[above_mat]**2 + 1e-2)) * 0.05   # Radial component
W[above_mat] = 0.05 * np.exp(-(Z[above_mat] - mat_height) / 2)  # Upward component, decays with height

# 3. Air relative humidity field (higher near the mat surface)
relative_humidity = np.zeros_like(X)
relative_humidity[mat_mask] = 90  # High humidity inside the mat
relative_humidity[~mat_mask] = 50 * np.exp(-(Z[~mat_mask] - mat_height) / 2)  # Decays with height above the mat

# Create the figure with 3 subplots
fig = plt.figure(figsize=(15, 5))

# Plot 1: Surface evaporation rate
ax1 = fig.add_subplot(131, projection='3d')
# Plot the fiber mat structure as a semi-transparent surface
mat_surface = (Z <= mat_height) & (np.sqrt(X**2 + Y**2) <= mat_radius)
evap_slice = evaporation_rate * mat_surface
sc1 = ax1.scatter(X, Y, Z, c=evap_slice, cmap='jet', s=10, alpha=0.5)
ax1.set_title('Surface Evaporation Rate (µm/s)')
ax1.set_xlabel('X (mm)')
ax1.set_ylabel('Y (mm)')
ax1.set_zlabel('Z (mm)')
fig.colorbar(sc1, ax=ax1, label='Evaporation Rate (µm/s)')

# Plot 2: Air velocity field with quiver (arrows)
ax2 = fig.add_subplot(132, projection='3d')
# Plot the fiber mat structure
mat_surface = (Z <= mat_height) & (np.sqrt(X**2 + Y**2) <= mat_radius)
ax2.scatter(X[mat_surface], Y[mat_surface], Z[mat_surface], c='gray', s=10, alpha=0.3)
# Plot velocity field using quiver
z_slice = slice(0, len(z), 5)
x_slice = slice(0, len(x), 5)
y_slice = slice(0, len(y), 5)
X_slice = X[x_slice, y_slice, z_slice]
Y_slice = Y[x_slice, y_slice, z_slice]
Z_slice = Z[x_slice, y_slice, z_slice]
U_slice = U[x_slice, y_slice, z_slice]
V_slice = V[x_slice, y_slice, z_slice]
W_slice = W[x_slice, y_slice, z_slice]
ax2.quiver(X_slice, Y_slice, Z_slice, U_slice, V_slice, W_slice, color='red', length=1.0, normalize=True)
ax2.set_title('Air Velocity Field (m/s)')
ax2.set_xlabel('X (mm)')
ax2.set_ylabel('Y (mm)')
ax2.set_zlabel('Z (mm)')

# Plot 3: Air relative humidity field
ax3 = fig.add_subplot(133, projection='3d')
# Plot the fiber mat structure
mat_surface = (Z <= mat_height) & (np.sqrt(X**2 + Y**2) <= mat_radius)
sc3 = ax3.scatter(X, Y, Z, c=relative_humidity, cmap='YlGnBu', s=10, alpha=0.5)
ax3.set_title('Air Relative Humidity Field (%)')
ax3.set_xlabel('X (mm)')
ax3.set_ylabel('Y (mm)')
ax3.set_zlabel('Z (mm)')
fig.colorbar(sc3, ax=ax3, label='Relative Humidity (%)')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('fiber_mat_evaporation_3d_visualization.png')