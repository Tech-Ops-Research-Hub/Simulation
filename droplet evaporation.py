import numpy as np
import matplotlib.pyplot as plt

# Set up the grid for the 2D domain (cross-section of the droplet)
x = np.linspace(-5, 5, 100)  # in mm
y = np.linspace(-5, 5, 100)  # in mm
X, Y = np.meshgrid(x, y)

# Simulate the droplet as a circular region (radius ~2 mm)
droplet_radius = 2.0
droplet_mask = np.sqrt(X**2 + Y**2) <= droplet_radius

# 1. Vapor diffusion flux (simplified as a radial diffusion from the droplet)
vapor_flux = np.exp(-np.sqrt(X**2 + Y**2) / 2)  # Exponential decay from the droplet
vapor_flux[droplet_mask] = 0  # Inside the droplet, no vapor flux

# 2. Velocity field (convective flow around the droplet)
# Simplified as a radial outward flow with some tangential component
U = -Y / (X**2 + Y**2 + 1e-2)  # Tangential component
V = X / (X**2 + Y**2 + 1e-2)   # Radial component
velocity_magnitude = np.sqrt(U**2 + V**2)

# 3. Relative humidity (higher near the droplet due to evaporation)
relative_humidity = 100 * np.exp(-np.sqrt(X**2 + Y**2) / 3)  # 100% at the surface, decaying outward
relative_humidity[droplet_mask] = 100  # Inside the droplet, assume 100% humidity

# 4. Surface evaporation rate (highest at the droplet boundary)
evaporation_rate = np.zeros_like(X)
evaporation_rate[~droplet_mask] = 0.5 * np.exp(-np.sqrt(X[~droplet_mask]**2 + Y[~droplet_mask]**2) / 2)

# Create the figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Plot 1: Vapor diffusion flux
cf1 = axes[0, 0].contourf(X, Y, vapor_flux, cmap='Blues', levels=20)
axes[0, 0].set_title('Vapor Diffusion Flux (mol/m²·s)')
fig.colorbar(cf1, ax=axes[0, 0])
axes[0, 0].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')  # Droplet boundary

# Plot 2: Velocity field
cf2 = axes[0, 1].contourf(X, Y, velocity_magnitude, cmap='Reds', levels=20)
axes[0, 1].quiver(X[::10, ::10], Y[::10, ::10], U[::10, ::10], V[::10, ::10], color='white')
axes[0, 1].set_title('Velocity Magnitude of Convective Flow (m/s)')
fig.colorbar(cf2, ax=axes[0, 1])
axes[0, 1].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 3: Surface evaporation rate
cf3 = axes[1, 0].contourf(X, Y, evaporation_rate, cmap='Oranges', levels=20)
axes[1, 0].set_title('Surface Evaporation Rate')
fig.colorbar(cf3, ax=axes[1, 0])
axes[1, 0].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 4: Relative humidity
cf4 = axes[1, 1].contourf(X, Y, relative_humidity, cmap='YlGnBu', levels=20)
axes[1, 1].set_title('Relative Humidity (%)')
fig.colorbar(cf4, ax=axes[1, 1])
axes[1, 1].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('droplet_evaporation_visualization.png')