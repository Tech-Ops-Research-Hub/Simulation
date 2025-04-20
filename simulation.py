import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
Lx, Ly = 0.1, 0.1  # Domain size (m)
nx, ny = 50, 50    # Grid points
dx, dy = Lx/nx, Ly/ny
x, y = np.linspace(0, Lx, nx), np.linspace(0, Ly, ny)
X, Y = np.meshgrid(x, y)

# Physical constants
solar_flux = 1000  # W/m^2 (one-sun)
absorptivity = 0.95  # Absorptivity of carbon fiber felt
T_ambient = 298  # Ambient temperature (K)
h_LV = 2.26e6  # Latent heat of vaporization (J/kg)
rho_water = 1000  # Density of water (kg/m^3)
D_va = 2.6e-5  # Vapor-air diffusivity (m^2/s)
mu_air = 1.8e-5  # Dynamic viscosity of air (Pa·s)
P0 = 101325  # Atmospheric pressure (Pa)

# Porosity levels to simulate
porosities = [0.3, 0.6, 0.9]
results = []

for porosity in porosities:
    # Effective diffusivity (Millington-Quirk, Eq. S8)
    D_eff = D_va * (porosity ** (4/3)) * (0.5 ** (10/3))  # Assuming Sg = 0.5 for simplicity

    # Permeability (approximation based on porosity)
    k = 1e-10 * (porosity / 0.6) ** 3  # Scaling permeability with porosity

    # Temperature distribution (simplified photothermal effect)
    T_surface = T_ambient + (solar_flux * absorptivity) / 100  # Rough estimate (K)
    T = np.ones((ny, nx)) * T_ambient
    T[-1, :] = T_surface  # Top surface heated by solar flux

    # Initialize vapor concentration (kg/m^3)
    c_vapor = np.zeros((ny, nx))
    c_vapor[-1, :] = 0.023  # Saturation vapor concentration at T_surface (approx.)

    # Initialize velocity field (m/s)
    u = np.zeros((ny, nx))  # x-direction
    v = np.zeros((ny, nx))  # y-direction

    # Simulate vapor diffusion and velocity field
    for _ in range(100):  # Iterations for steady state
        # Vapor diffusion (Eq. S6, simplified)
        c_new = c_vapor.copy()
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                c_new[i, j] = c_vapor[i, j] + D_eff * (
                    (c_vapor[i+1, j] - 2*c_vapor[i, j] + c_vapor[i-1, j]) / dy**2 +
                    (c_vapor[i, j+1] - 2*c_vapor[i, j] + c_vapor[i, j-1]) / dx**2
                )
        c_new[0, :] = 0  # Boundary: vapor removed at bottom (air)
        c_vapor = c_new

        # Velocity field using Darcy's Law (Eq. S4, simplified)
        for i in range(1, ny-1):
            for j in range(1, nx-1):
                grad_p_x = -(c_vapor[i, j+1] - c_vapor[i, j-1]) / (2*dx) * P0 / 0.023  # Pressure gradient
                grad_p_y = -(c_vapor[i+1, j] - c_vapor[i-1, j]) / (2*dy) * P0 / 0.023
                u[i, j] = -k / mu_air * grad_p_x
                v[i, j] = -k / mu_air * grad_p_y

    # Relative humidity (RH) from vapor concentration
    RH = c_vapor / 0.023 * 100  # Assuming 0.023 kg/m^3 as saturation at T_ambient

    # Evaporation rate (kg/m^2/h)
    evap_flux = D_eff * (c_vapor[-1, :] - c_vapor[-2, :]) / dy * rho_water  # kg/m^2/s
    evap_rate = np.mean(evap_flux) * 3600  # kg/m^2/h

    results.append((u, v, RH, evap_rate))

# Plotting results
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for idx, (porosity, (u, v, RH, evap_rate)) in enumerate(zip(porosities, results)):
    # Velocity field
    axes[idx, 0].quiver(X, Y, u, v, scale=0.1)
    axes[idx, 0].set_title(f"Velocity Field (Porosity={porosity})")
    axes[idx, 0].set_xlabel("X (m)")
    axes[idx, 0].set_ylabel("Y (m)")

    # Relative humidity field
    im = axes[idx, 1].imshow(RH, extent=[0, Lx, 0, Ly], origin='lower', cmap='Blues')
    plt.colorbar(im, ax=axes[idx, 1], label="Relative Humidity (%)")
    axes[idx, 1].set_title(f"Relative Humidity (Porosity={porosity})")
    axes[idx, 1].set_xlabel("X (m)")
    axes[idx, 1].set_ylabel("Y (m)")

    # Evaporation rate
    axes[idx, 2].bar(0, evap_rate, color='blue')
    axes[idx, 2].set_title(f"Evaporation Rate: {evap_rate:.2f} kg/m²h")
    axes[idx, 2].set_xticks([])
    axes[idx, 2].set_ylabel("Evaporation Rate (kg/m²h)")

plt.tight_layout()
plt.savefig("evaporation_simulation.png")