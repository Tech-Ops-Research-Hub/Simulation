import numpy as np
import matplotlib.pyplot as plt

# Step 1: Define the domain and grid
x = np.linspace(-0.25, 0.75, 100)  # x-axis: 1m wide domain
y = np.linspace(-0.25, 0.25, 50)   # y-axis: 0.5m tall domain
X, Y = np.meshgrid(x, y)

# Step 2: Define the flow field (uniform flow + vortex to approximate the rectangular obstacle)
# Uniform flow: u = 0.1 m/s (in x-direction), v = 0 m/s (in y-direction)
u_uniform = 0.1 * np.ones_like(X)
v_uniform = np.zeros_like(Y)

# Add a vortex to simulate the swirling pattern behind the obstacle
# Vortex center at (0.1, 0) - slightly behind the rectangle (0.1m x 0.05m centered at origin)
x_vortex, y_vortex = 0.1, 0.0
gamma = 0.5  # Vortex strength (adjusted to create a visible swirl)
r_squared = (X - x_vortex)**2 + (Y - y_vortex)**2
u_vortex = -gamma * (Y - y_vortex) / (2 * np.pi * r_squared)
v_vortex = gamma * (X - x_vortex) / (2 * np.pi * r_squared)

# Avoid division by zero near the vortex center
u_vortex = np.where(r_squared < 1e-4, 0, u_vortex)
v_vortex = np.where(r_squared < 1e-4, 0, v_vortex)

# Combine the uniform flow and vortex
u = u_uniform + u_vortex
v = v_uniform + v_vortex

# Step 3: Plot the streamlines
plt.figure(figsize=(10, 5))
plt.streamplot(X, Y, u, v, color='white', linewidth=1, density=1.5)

# Add the rectangular obstacle (0.1m x 0.05m centered at origin)
rect = plt.Rectangle((-0.05, -0.025), 0.1, 0.05, color='gray', alpha=0.5)
plt.gca().add_patch(rect)

# Customize the plot to match the image
plt.title("Top View", loc='left', color='black')
plt.gca().set_facecolor('blue')  # Background color similar to the image
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.grid(False)
plt.axis('equal')  # Ensure aspect ratio is correct
plt.show()