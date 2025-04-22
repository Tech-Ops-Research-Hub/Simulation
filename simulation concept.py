import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the domain and grid
x = np.linspace(-0.25, 0.75, 100)  # x-axis: 1m wide domain
y = np.linspace(-0.25, 0.25, 50)   # y-axis: 0.5m tall domain
X, Y = np.meshgrid(x, y)

# Step 2: Define the flow field (uniform flow + vortex)
# Uniform flow: u = 0.1 m/s (in x-direction), v = 0 m/s (in y-direction)
u_uniform = 0.1 * np.ones_like(X)
v_uniform = np.zeros_like(Y)

# Add a vortex to simulate the swirling pattern behind the obstacle
x_vortex, y_vortex = 0.1, 0.0
gamma = 0.5  # Vortex strength
r_squared = (X - x_vortex)**2 + (Y - y_vortex)**2
u_vortex = -gamma * (Y - y_vortex) / (2 * np.pi * r_squared)
v_vortex = gamma * (X - x_vortex) / (2 * np.pi * r_squared)

# Avoid division by zero near the vortex center
u_vortex = np.where(r_squared < 1e-4, 0, u_vortex)
v_vortex = np.where(r_squared < 1e-4, 0, v_vortex)

# Combine the uniform flow and vortex
u = u_uniform + u_vortex
v = v_uniform + v_vortex

# Step 3: Simulate water vapor transport (advection-diffusion)
# Initialize vapor concentration (c) - source at the rectangle's right edge
c = np.zeros_like(X)
rect_x_min, rect_x_max = -0.05, 0.05  # Rectangle bounds (0.1m wide)
rect_y_min, rect_y_max = -0.025, 0.025  # Rectangle bounds (0.05m tall)
source_mask = (X >= rect_x_max - 0.01) & (X <= rect_x_max) & (Y >= rect_y_min) & (Y <= rect_y_max)
c[source_mask] = 1.0  # Vapor source concentration (normalized)

# Simplified advection-diffusion: c_new = c - dt * (u * dc/dx + v * dc/dy) + D * (d2c/dx2 + d2c/dy2)
D = 2.5e-5  # Diffusion coefficient of water vapor in air (m^2/s)
dx = x[1] - x[0]
dy = y[1] - y[0]
dt = 0.01  # Time step (small for stability)

# Simulate for a few time steps to let the vapor spread
for _ in range(50):
    # Compute gradients for advection
    dc_dx = (np.roll(c, -1, axis=1) - np.roll(c, 1, axis=1)) / (2 * dx)
    dc_dy = (np.roll(c, -1, axis=0) - np.roll(c, 1, axis=0)) / (2 * dy)
    
    # Compute second derivatives for diffusion
    d2c_dx2 = (np.roll(c, -1, axis=1) - 2 * c + np.roll(c, 1, axis=1)) / dx**2
    d2c_dy2 = (np.roll(c, -1, axis=0) - 2 * c + np.roll(c, 1, axis=0)) / dy**2
    
    # Update concentration (advection + diffusion)
    c = c - dt * (u * dc_dx + v * dc_dy) + D * dt * (d2c_dx2 + d2c_dy2)
    
    # Reapply the source
    c[source_mask] = 1.0
    
    # Boundary condition: vapor concentration goes to 0 at domain edges
    c[:, 0] = c[:, -1] = c[0, :] = c[-1, :] = 0

# Step 4: Plot the streamlines with vapor concentration overlay
plt.figure(figsize=(10, 5))
# Plot vapor concentration as a color gradient (vapor effect)
plt.contourf(X, Y, c, cmap='Reds', alpha=0.5, levels=20)
plt.colorbar(label='Vapor Concentration')

# Plot streamlines on top
plt.streamplot(X, Y, u, v, color='white', linewidth=1, density=1.5)

# Add the rectangular obstacle
rect = plt.Rectangle((-0.05, -0.025), 0.1, 0.05, color='gray', alpha=0.5)
plt.gca().add_patch(rect)

# Customize the plot to match the image
plt.title("Top View", loc='left', color='black')
plt.gca().set_facecolor('blue')  # Background color
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(False)
plt.axis('equal')
plt.show()