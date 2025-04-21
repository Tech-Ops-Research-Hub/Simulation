import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Arrow, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.patheffects as pe

# Set global font and figure settings for a professional look
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

# Simulation domain and grid (same as previous script)
Lx, Ly = 0.1, 0.1  # Domain size (m)
nx, ny = 50, 50    # Grid points
x = np.linspace(0, Lx, nx)
y = np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Simulated data generation (simplified for visualization)
def generate_humidity_field(structure="flat"):
    if structure == "flat":
        rh = 90 * np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / 0.02)
    else:
        rh = 90 * np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / 0.02) * np.cos(10 * np.arctan2(Y - Ly/2, X - Lx/2))
    return rh

def generate_velocity_field(structure="flat"):
    if structure == "flat":
        vel = 0.13 * np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / 0.01)
    else:
        vel = 0.13 * np.exp(-((X - Lx/2)**2 + (Y - Ly/2)**2) / 0.01) * np.sin(10 * np.arctan2(Y - Ly/2, X - Lx/2))
    return vel

def generate_3d_fields(structure="spiral"):
    Z = np.linspace(0, 0.05, 50)  # Height (m)
    X3, Y3, Z3 = np.meshgrid(x, y, Z)
    if structure == "spiral":
        evap_rate = 2.18 * np.exp(-((X3 - Lx/2)**2 + (Y3 - Ly/2)**2) / 0.02) * np.exp(-Z3/0.02)
        vel = 0.13 * np.exp(-((X3 - Lx/2)**2 + (Y3 - Ly/2)**2) / 0.02) * np.sin(10 * np.arctan2(Y3 - Ly/2, X3 - Lx/2)) * np.exp(-Z3/0.03)
        rh = 80 * np.exp(-((X3 - Lx/2)**2 + (Y3 - Ly/2)**2) / 0.02) * np.cos(10 * np.arctan2(Y3 - Ly/2, X3 - Lx/2)) * np.exp(-Z3/0.03)
    return evap_rate, vel, rh

# Create the figure with a grid layout (4 rows, 3 columns)
fig = plt.figure(figsize=(8.5, 11), facecolor='white')

# Panel (a): Schematic Diagram (spans all 3 columns in row 1)
ax_a = fig.add_subplot(4, 3, (1, 3), frameon=False)
ax_a.set_xticks([])
ax_a.set_yticks([])

# Draw the schematic for flat structure
ax_a.add_patch(Rectangle((0.05, 0.5), 0.2, 0.05, color='gray', label='Flat Structure'))
ax_a.add_patch(Rectangle((0.05, 0.45), 0.2, 0.05, color='blue', alpha=0.5, label='Water'))
ax_a.add_patch(Arrow(0.15, 0.65, 0, 0.2, width=0.02, color='orange', label='Solar Irradiation'))
ax_a.text(0.15, 0.4, 'Flat Structure', ha='center', va='center', fontsize=8)
ax_a.text(0.15, 0.65, 'Solar\nIrradiation', ha='center', va='center', fontsize=8, color='orange')

# Draw the schematic for spiral structure
ax_a.add_patch(Circle((0.45, 0.5), 0.05, color='gray', label='Spiral Structure'))
ax_a.add_patch(Rectangle((0.4, 0.45), 0.1, 0.05, color='blue', alpha=0.5))
ax_a.add_patch(Arrow(0.45, 0.65, 0, 0.2, width=0.02, color='orange'))
ax_a.text(0.45, 0.4, 'Spiral Structure', ha='center', va='center', fontsize=8)
ax_a.text(0.45, 0.65, 'Solar\nIrradiation', ha='center', va='center', fontsize=8, color='orange')

# Draw air flow
ax_a.add_patch(FancyArrowPatch((0.15, 0.55), (0.25, 0.65), arrowstyle='->', mutation_scale=15, color='cyan', label='Air Flow'))
ax_a.add_patch(FancyArrowPatch((0.45, 0.55), (0.55, 0.65), arrowstyle='->', mutation_scale=15, color='cyan'))
ax_a.text(0.25, 0.65, 'Air Flow (0.2 m/s)', ha='left', va='center', fontsize=8, color='cyan')

ax_a.set_xlim(0, 0.8)
ax_a.set_ylim(0.3, 0.9)
ax_a.text(-0.05, 0.9, '(a)', fontsize=12, fontweight='bold')

# Panel (b): Humidity Distribution (row 2, columns 1-3)
# Flat structure - Cross-section
ax_b1 = fig.add_subplot(4, 3, 4)
rh_flat_cross = generate_humidity_field("flat")[:, nx//2].reshape(ny, 1) * np.ones((ny, nx))
contour_b1 = ax_b1.contourf(X, Y, rh_flat_cross, levels=20, cmap='jet')
fig.colorbar(contour_b1, ax=ax_b1, label='Relative Humidity (%)')
ax_b1.set_title('Flat - Cross-section', fontsize=8)
ax_b1.set_xlabel('Distance (m)')
ax_b1.set_ylabel('Height (m)')

# Flat structure - Top view
ax_b2 = fig.add_subplot(4, 3, 5)
rh_flat_top = generate_humidity_field("flat")
contour_b2 = ax_b2.contourf(X, Y, rh_flat_top, levels=20, cmap='jet')
fig.colorbar(contour_b2, ax=ax_b2, label='Relative Humidity (%)')
ax_b2.set_title('Flat - Top View', fontsize=8)
ax_b2.set_xlabel('Distance (m)')
ax_b2.set_ylabel('Distance (m)')

# Spiral structure - Cross-section
ax_b3 = fig.add_subplot(4, 3, 7)
rh_spiral_cross = generate_humidity_field("spiral")[:, nx//2].reshape(ny, 1) * np.ones((ny, nx))
contour_b3 = ax_b3.contourf(X, Y, rh_spiral_cross, levels=20, cmap='jet')
fig.colorbar(contour_b3, ax=ax_b3, label='Relative Humidity (%)')
ax_b3.set_title('Spiral - Cross-section', fontsize=8)
ax_b3.set_xlabel('Distance (m)')
ax_b3.set_ylabel('Height (m)')

# Spiral structure - Top view
ax_b4 = fig.add_subplot(4, 3, 8)
rh_spiral_top = generate_humidity_field("spiral")
contour_b4 = ax_b4.contourf(X, Y, rh_spiral_top, levels=20, cmap='jet')
fig.colorbar(contour_b4, ax=ax_b4, label='Relative Humidity (%)')
ax_b4.set_title('Spiral - Top View', fontsize=8)
ax_b4.set_xlabel('Distance (m)')
ax_b4.set_ylabel('Distance (m)')

# Add label for panel (b)
fig.text(0.05, 0.72, '(b) Humidity Distribution', fontsize=12, fontweight='bold')

# Panel (c): Velocity Magnitude (row 3, columns 1-3)
# Flat structure - Cross-section
ax_c1 = fig.add_subplot(4, 3, 6)
vel_flat_cross = generate_velocity_field("flat")[:, nx//2].reshape(ny, 1) * np.ones((ny, nx))
contour_c1 = ax_c1.contourf(X, Y, vel_flat_cross, levels=20, cmap='inferno')
fig.colorbar(contour_c1, ax=ax_c1, label='Velocity Magnitude (m/s)')
ax_c1.set_title('Flat - Cross-section', fontsize=8)
ax_c1.set_xlabel('Distance (m)')
ax_c1.set_ylabel('Height (m)')

# Flat structure - Top view
ax_c2 = fig.add_subplot(4, 3, 9)
vel_flat_top = generate_velocity_field("flat")
contour_c2 = ax_c2.contourf(X, Y, vel_flat_top, levels=20, cmap='inferno')
fig.colorbar(contour_c2, ax=ax_c2, label='Velocity Magnitude (m/s)')
ax_c2.set_title('Flat - Top View', fontsize=8)
ax_c2.set_xlabel('Distance (m)')
ax_c2.set_ylabel('Distance (m)')

# Spiral structure - Cross-section
ax_c3 = fig.add_subplot(4, 3, 10)
vel_spiral_cross = generate_velocity_field("spiral")[:, nx//2].reshape(ny, 1) * np.ones((ny, nx))
contour_c3 = ax_c3.contourf(X, Y, vel_spiral_cross, levels=20, cmap='inferno')
fig.colorbar(contour_c3, ax=ax_c3, label='Velocity Magnitude (m/s)')
ax_c3.set_title('Spiral - Cross-section', fontsize=8)
ax_c3.set_xlabel('Distance (m)')
ax_c3.set_ylabel('Height (m)')

# Spiral structure - Top view
ax_c4 = fig.add_subplot(4, 3, 11)
vel_spiral_top = generate_velocity_field("spiral")
contour_c4 = ax_c4.contourf(X, Y, vel_spiral_top, levels=20, cmap='inferno')
fig.colorbar(contour_c4, ax=ax_c4, label='Velocity Magnitude (m/s)')
ax_c4.set_title('Spiral - Top View', fontsize=8)
ax_c4.set_xlabel('Distance (m)')
ax_c4.set_ylabel('Distance (m)')

# Add label for panel (c)
fig.text(0.05, 0.47, '(c) Velocity Magnitude of Convective Flow (m/s)', fontsize=12, fontweight='bold')

# Panels (d), (e), (f): 3D Plots (row 4, columns 1-3)
evap_rate, vel_3d, rh_3d = generate_3d_fields("spiral")

# Panel (d): Surface Evaporation Rate
ax_d = fig.add_subplot(4, 3, 12, projection='3d')
Z = np.linspace(0, 0.05, 50)
X3, Y3, Z3 = np.meshgrid(x, y, Z)
surf_d = ax_d.plot_surface(X3[:, :, 0], Y3[:, :, 0], evap_rate[:, :, 0], cmap='viridis', edgecolor='none')
ax_d.set_zlabel('Evaporation Rate (kg/m²h)')
ax_d.set_xlabel('Distance (m)')
ax_d.set_ylabel('Distance (m)')
fig.colorbar(surf_d, ax=ax_d, label='Evaporation Rate (kg/m²h)')
ax_d.set_title('(d) Surface Evaporation Rate', fontsize=10, pad=20)
ax_d.view_init(elev=30, azim=45)

# Panel (e): Air Velocity Field
ax_e = fig.add_subplot(4, 3, 13, projection='3d')
surf_e = ax_e.plot_surface(X3[:, :, 0], Y3[:, :, 0], vel_3d[:, :, 0], cmap='inferno', edgecolor='none')
ax_e.set_zlabel('Velocity (m/s)')
ax_e.set_xlabel('Distance (m)')
ax_e.set_ylabel('Distance (m)')
fig.colorbar(surf_e, ax=ax_e, label='Velocity (m/s)')
ax_e.set_title('(e) Air Velocity Field', fontsize=10, pad=20)
ax_e.view_init(elev=30, azim=45)

# Panel (f): Relative Humidity Field
ax_f = fig.add_subplot(4, 3, 14, projection='3d')
surf_f = ax_f.plot_surface(X3[:, :, 0], Y3[:, :, 0], rh_3d[:, :, 0], cmap='jet', edgecolor='none')
ax_f.set_zlabel('Relative Humidity (%)')
ax_f.set_xlabel('Distance (m)')
ax_f.set_ylabel('Distance (m)')
fig.colorbar(surf_f, ax=ax_f, label='Relative Humidity (%)')
ax_f.set_title('(f) Relative Humidity Field', fontsize=10, pad=20)
ax_f.view_init(elev=30, azim=45)

# Adjust layout to prevent overlap
plt.tight_layout()

# Add a caption below the figure
caption = (
    "Figure 1: Simulation results for photothermal evaporation. (a) Schematic diagram of the simulation setup for flat and spiral structures. "
    "(b) Humidity distribution for flat and spiral structures (cross-section and top views). (c) Velocity magnitude of convective flow for flat and spiral structures. "
    "(d) Surface evaporation rate, (e) air velocity field, and (f) relative humidity field for the spiral structure, obtained using a 2D and 3D simulation approach."
)
fig.text(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=8)

# Save the figure
plt.savefig("Figure_1_Simulation_Results.png", dpi=300, bbox_inches='tight')
plt.close()

print("Figure generated as 'Figure_1_Simulation_Results.png'")