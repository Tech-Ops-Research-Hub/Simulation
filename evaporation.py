import numpy as np
import matplotlib.pyplot as plt

# Set up the grid for the 2D domain (cross-section of the droplet)
x = np.linspace(-5, 5, 100)  # in mm
y = np.linspace(-5, 5, 100)  # in mm
X, Y = np.meshgrid(x, y)

# Simulate the droplet (radius ~2 mm)
droplet_radius = 2.0
droplet_mask = np.sqrt(X**2 + Y**2) <= droplet_radius

# Simulate a spiral droplet structure (for variation)
spiral_mask = (np.sqrt(X**2 + Y**2) <= droplet_radius) & (np.sin(5 * np.arctan2(Y, X)) > 0)

# 1. Vapor diffusion flux (flat and spiral structures)
vapor_flux_flat = np.exp(-np.sqrt(X**2 + Y**2) / 2)
vapor_flux_flat[droplet_mask] = 0
vapor_flux_spiral = np.exp(-np.sqrt(X**2 + Y**2) / 2)
vapor_flux_spiral[spiral_mask] = 0

# 2. Velocity field (flat and spiral structures)
U_flat = -Y / (X**2 + Y**2 + 1e-2)
V_flat = X / (X**2 + Y**2 + 1e-2)
velocity_magnitude_flat = np.sqrt(U_flat**2 + V_flat**2)

U_spiral = -Y / (X**2 + Y**2 + 1e-2) * (1 + 0.5 * np.sin(5 * np.arctan2(Y, X)))
V_spiral = X / (X**2 + Y**2 + 1e-2) * (1 + 0.5 * np.sin(5 * np.arctan2(Y, X)))
velocity_magnitude_spiral = np.sqrt(U_spiral**2 + V_spiral**2)

# 3. Surface evaporation rate (flat and spiral structures)
evaporation_rate_flat = np.zeros_like(X)
evaporation_rate_flat[~droplet_mask] = 0.5 * np.exp(-np.sqrt(X[~droplet_mask]**2 + Y[~droplet_mask]**2) / 2)

evaporation_rate_spiral = np.zeros_like(X)
evaporation_rate_spiral[~spiral_mask] = 0.5 * np.exp(-np.sqrt(X[~spiral_mask]**2 + Y[~spiral_mask]**2) / 2)

# 4. Relative humidity (flat and spiral structures)
relative_humidity_flat = 100 * np.exp(-np.sqrt(X**2 + Y**2) / 3)
relative_humidity_flat[droplet_mask] = 100

relative_humidity_spiral = 100 * np.exp(-np.sqrt(X**2 + Y**2) / 3)
relative_humidity_spiral[spiral_mask] = 100

# 5. Temperature distribution (simplified, cooling due to evaporation)
temperature = 25 - 5 * np.exp(-np.sqrt(X**2 + Y**2) / 2)  # Cooler near the droplet surface
temperature[droplet_mask] = 20  # Inside the droplet

# 6. Pressure field (simplified, lower pressure near the droplet due to convection)
pressure = 101325 - 100 * np.exp(-np.sqrt(X**2 + Y**2) / 2)  # in Pa
pressure[droplet_mask] = 101325  # Inside the droplet

# 7. Concentration gradient (simplified, higher near the droplet)
concentration = np.exp(-np.sqrt(X**2 + Y**2) / 2)
concentration[droplet_mask] = 1

# Create the figure with 12 subplots (3x4 grid)
fig, axes = plt.subplots(3, 4, figsize=(16, 12))

# Plot 1: Vapor diffusion flux (flat)
cf1 = axes[0, 0].contourf(X, Y, vapor_flux_flat, cmap='Blues', levels=20)
axes[0, 0].set_title('Vapor Flux (Flat) (mol/m²·s)')
fig.colorbar(cf1, ax=axes[0, 0])
axes[0, 0].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 2: Vapor diffusion flux (spiral)
cf2 = axes[0, 1].contourf(X, Y, vapor_flux_spiral, cmap='Blues', levels=20)
axes[0, 1].set_title('Vapor Flux (Spiral) (mol/m²·s)')
fig.colorbar(cf2, ax=axes[0, 1])
axes[0, 1].contour(X, Y, spiral_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 3: Velocity field (flat)
cf3 = axes[0, 2].contourf(X, Y, velocity_magnitude_flat, cmap='Reds', levels=20)
axes[0, 2].quiver(X[::10, ::10], Y[::10, ::10], U_flat[::10, ::10], V_flat[::10, ::10], color='white')
axes[0, 2].set_title('Velocity (Flat) (m/s)')
fig.colorbar(cf3, ax=axes[0, 2])
axes[0, 2].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 4: Velocity field (spiral)
cf4 = axes[0, 3].contourf(X, Y, velocity_magnitude_spiral, cmap='Reds', levels=20)
axes[0, 3].quiver(X[::10, ::10], Y[::10, ::10], U_spiral[::10, ::10], V_spiral[::10, ::10], color='white')
axes[0, 3].set_title('Velocity (Spiral) (m/s)')
fig.colorbar(cf4, ax=axes[0, 3])
axes[0, 3].contour(X, Y, spiral_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 5: Surface evaporation rate (flat)
cf5 = axes[1, 0].contourf(X, Y, evaporation_rate_flat, cmap='Oranges', levels=20)
axes[1, 0].set_title('Evaporation Rate (Flat)')
fig.colorbar(cf5, ax=axes[1, 0])
axes[1, 0].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 6: Surface evaporation rate (spiral)
cf6 = axes[1, 1].contourf(X, Y, evaporation_rate_spiral, cmap='Oranges', levels=20)
axes[1, 1].set_title('Evaporation Rate (Spiral)')
fig.colorbar(cf6, ax=axes[1, 1])
axes[1, 1].contour(X, Y, spiral_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 7: Relative humidity (flat)
cf7 = axes[1, 2].contourf(X, Y, relative_humidity_flat, cmap='YlGnBu', levels=20)
axes[1, 2].set_title('Humidity (Flat) (%)')
fig.colorbar(cf7, ax=axes[1, 2])
axes[1, 2].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 8: Relative humidity (spiral)
cf8 = axes[1, 3].contourf(X, Y, relative_humidity_spiral, cmap='YlGnBu', levels=20)
axes[1, 3].set_title('Humidity (Spiral) (%)')
fig.colorbar(cf8, ax=axes[1, 3])
axes[1, 3].contour(X, Y, spiral_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 9: Temperature distribution
cf9 = axes[2, 0].contourf(X, Y, temperature, cmap='coolwarm', levels=20)
axes[2, 0].set_title('Temperature (°C)')
fig.colorbar(cf9, ax=axes[2, 0])
axes[2, 0].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 10: Pressure field
cf10 = axes[2, 1].contourf(X, Y, pressure, cmap='viridis', levels=20)
axes[2, 1].set_title('Pressure (Pa)')
fig.colorbar(cf10, ax=axes[2, 1])
axes[2, 1].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 11: Concentration gradient
cf11 = axes[2, 2].contourf(X, Y, concentration, cmap='Purples', levels=20)
axes[2, 2].set_title('Concentration Gradient')
fig.colorbar(cf11, ax=axes[2, 2])
axes[2, 2].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Plot 12: Evaporation rate at a later time step (simplified evolution)
evaporation_rate_later = evaporation_rate_flat * 0.8  # Reduced rate over time
cf12 = axes[2, 3].contourf(X, Y, evaporation_rate_later, cmap='Oranges', levels=20)
axes[2, 3].set_title('Evaporation Rate (Later Time)')
fig.colorbar(cf12, ax=axes[2, 3])
axes[2, 3].contour(X, Y, droplet_mask, levels=[0.5], colors='white', linestyles='--')

# Adjust layout
plt.tight_layout()

# Save the plot
plt.savefig('droplet_evaporation_visualization_12_tables.png')