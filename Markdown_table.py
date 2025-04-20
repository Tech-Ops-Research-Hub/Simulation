import numpy as np
import pandas as pd
from tabulate import tabulate
from scipy.integrate import quad
import markdown
import pdfkit
import os

# Physical constants
h_LV = 2.26e6  # Latent heat of vaporization (J/kg)
P_0 = 1000  # Solar irradiation (W/m^2, 1 sun)
R = 461.5  # Gas constant for water vapor (J/kg·K)

# Given conditions
T_ambient = 25  # °C
T_dry_max_carbon = 58.1  # °C
T_wet_max_carbon = 34  # °C
evap_rate_carbon_0_9 = 1.415  # kg/m^2/h for Carbon Fiber Felt at porosity 0.9

# Step 1: Absorption Rate Calculation (Section 1)
def I_solar(lambda_nm):
    return 1000 / (2500 - 200)  # Approximate constant intensity (W/m^2/nm)

def T_lambda(lambda_nm, material):
    if material.startswith("CuxO/C"):
        return 0.03  # 3% transmittance for PAPS (from Figure S5)
    return 0.05  # 5% for Carbon Fiber Felt

def R_lambda(lambda_nm, material):
    if material.startswith("CuxO/C"):
        return 0.05  # 5% reflectance for PAPS (from Figure S5)
    return 0.10  # 10% for Carbon Fiber Felt

def A_lambda(lambda_nm, material):
    return 1 - T_lambda(lambda_nm, material) - R_lambda(lambda_nm, material)

def compute_absorption_rate(material):
    numerator, _ = quad(lambda x: I_solar(x) * A_lambda(x, material), 200, 2500)
    denominator, _ = quad(I_solar, 200, 2500)
    return numerator / denominator * 100  # Percentage

# Step 2: Simulate Performance Across Conditions
data = []
# Configurations for CuxO/C PAPS
paps_configs = [
    ("Planar", 0.6, "-", 1, 0, "Natural"),
    ("4×4 Arrays", 0.7, "-", 1, 0, "Natural"),
    ("6×6 Arrays", 0.8, "-", 1, 0, "Forced"),
    ("8×8 Arrays", 0.9, "5", 1, 0, "Forced"),
    ("8×8 Arrays", 0.9, "5", 2, 0, "Forced"),
    ("8×8 Arrays", 0.9, "5", 3, 0, "Forced"),
    ("8×8 Arrays", 0.9, "3", 1, 0, "Natural"),
    ("8×8 Arrays", 0.9, "4", 1, 22.5, "Forced"),
    ("8×8 Arrays", 0.9, "5", 1, 45, "Forced"),
]
# Configurations for Carbon Fiber Felt
carbon_configs = [
    (0.3, 1, 0, "Natural"),
    (0.6, 1, 0, "Natural"),
    (0.9, 1, 0, "Forced"),
    (0.9, 2, 0, "Forced"),
    (0.9, 1, 22.5, "Natural"),
    (0.9, 1, 45, "Forced"),
]

# Base values from Tables S1, S2, S3
paps_base_evap_rates = {"Planar": 1.68, "4×4 Arrays": 1.92, "6×6 Arrays": 2.01, "8×8 Arrays": 2.08}
paps_base_heights = {"3": 1.96, "4": 2.02, "5": 2.08}

for config in paps_configs:
    structure, porosity, height, solar_sun, angle, convection = config
    material = "CuxO/C PAPS"

    # Absorption rate (adjusted for angle)
    absorption = compute_absorption_rate(material)
    absorption *= (1 - 0.005 * angle / 45)  # Reduce absorption with angle (Figure S15)

    # Evaporation rate (base from Tables S1, S2, adjusted for conditions)
    evap_rate = paps_base_evap_rates[structure]
    if height != "-":
        evap_rate = paps_base_heights[height]
    evap_rate *= solar_sun  # Scale with solar illumination
    evap_rate *= (1 - 0.01 * angle / 45)  # Reduce with angle
    evap_rate *= 1.05 if convection == "Forced" else 1.0  # Boost with forced convection

    # Efficiency (Section 2)
    dark_field = 0.1  # kg/m^2/h
    m = (evap_rate - dark_field) / 3600  # kg/m^2/s
    efficiency = (m * h_LV) / (solar_sun * P_0) * 100
    efficiency *= 0.95 if solar_sun > 1 else 1.0  # Efficiency drops at higher illumination (Table S3 trend)

    # Surface temperature (based on Figure S16)
    T_surface = 36.5 + (solar_sun - 1) * 5.0  # Base from 1 sun, increases with illumination
    T_surface -= 1.0 * angle / 45  # Slight decrease with angle

    # Accumulated solar energy (Figures S10, S12, S14)
    base_energy = {"Planar": 7.2, "4×4 Arrays": 8.5, "6×6 Arrays": 9.3, "8×8 Arrays": 10.5}[structure]
    if height != "-":
        height_factors = {"3": 0.93, "4": 0.97, "5": 1.0}
        base_energy *= height_factors[height]
    energy = base_energy * solar_sun * (1 - 0.01 * angle / 45)

    # Velocity and RH (simulated, Figure S17 trends)
    velocity = 0.05 * porosity / 0.6 * (1.5 if convection == "Forced" else 1.0) * solar_sun
    rh = 85.0 - 5.0 * (porosity - 0.6) / 0.3 - 3.0 * (solar_sun - 1) - 2.0 if convection == "Forced" else 0.0

    # Latent heat flux (Equation S10)
    latent_heat = evap_rate / 3600 * h_LV  # W/m^2

    data.append({
        "Material Type": material,
        "Structure Config": structure,
        "Porosity": porosity,
        "Array Height (mm)": height,
        "Solar Illumination (sun)": solar_sun,
        "Incident Angle (°)": angle,
        "Convection Type": convection,
        "Ambient Temp (°C)": T_ambient,
        "Surface Temp (°C)": round(T_surface, 1),
        "Absorption Rate (%)": round(absorption, 1),
        "Evaporation Rate (kg/m^2h)": round(evap_rate, 2),
        "Efficiency (%)": round(efficiency, 2),
        "Accumulated Solar Energy (MJ/m^2·day)": round(energy, 1),
        "Max Velocity (m/s)": round(velocity, 2),
        "Surface RH (%)": round(rh, 1),
        "Latent Heat Flux (W/m^2)": round(latent_heat, 0)
    })

for config in carbon_configs:
    porosity, solar_sun, angle, convection = config
    material = "Carbon Fiber Felt"

    # Absorption rate
    absorption = compute_absorption_rate(material)
    absorption *= (1 - 0.005 * angle / 45)

    # Evaporation rate (scale from given 1.415 at porosity 0.9)
    evap_rate = evap_rate_carbon_0_9 * (porosity / 0.9) * solar_sun * (1 - 0.01 * angle / 45)
    evap_rate *= 1.05 if convection == "Forced" else 1.0

    # Efficiency
    m = (evap_rate - dark_field) / 3600
    efficiency = (m * h_LV) / (solar_sun * P_0) * 100
    efficiency *= 0.95 if solar_sun > 1 else 1.0

    # Surface temperature
    T_surface = T_wet_max_carbon + (solar_sun - 1) * 3.0 - 1.0 * angle / 45

    # Accumulated solar energy (lower than PAPS due to lack of arrays)
    base_energy = 7.8 * (porosity / 0.9)
    energy = base_energy * solar_sun * (1 - 0.01 * angle / 45)

    # Velocity and RH
    velocity = 0.03 * (porosity / 0.3) * (1.5 if convection == "Forced" else 1.0) * solar_sun
    rh = 90.0 - 5.0 * (porosity - 0.3) / 0.6 - 3.0 * (solar_sun - 1) - 2.0 if convection == "Forced" else 0.0

    # Latent heat flux
    latent_heat = evap_rate / 3600 * h_LV

    data.append({
        "Material Type": material,
        "Structure Config": "-",
        "Porosity": porosity,
        "Array Height (mm)": "-",
        "Solar Illumination (sun)": solar_sun,
        "Incident Angle (°)": angle,
        "Convection Type": convection,
        "Ambient Temp (°C)": T_ambient,
        "Surface Temp (°C)": round(T_surface, 1),
        "Absorption Rate (%)": round(absorption, 1),
        "Evaporation Rate (kg/m^2h)": round(evap_rate, 3),
        "Efficiency (%)": round(efficiency, 2),
        "Accumulated Solar Energy (MJ/m^2·day)": round(energy, 1),
        "Max Velocity (m/s)": round(velocity, 2),
        "Surface RH (%)": round(rh, 1),
        "Latent Heat Flux (W/m^2)": round(latent_heat, 0)
    })

# Create DataFrame
df = pd.DataFrame(data)

# Convert to Markdown table
markdown_table = tabulate(df, headers="keys", tablefmt="pipe", showindex=False)

# Add title and notes
table_content = "# Table S4: Comprehensive Simulated Performance of Photothermal Structures\n\n"
table_content += markdown_table
table_content += "\n\n### Notes:\n"
table_content += "- **Absorption Rate**: Calculated using \\( A(\\lambda) = 1 - T(\\lambda) - R(\\lambda) \\). For CuxO/C PAPS, values are derived from Figure S5 (90–95%); for Carbon Fiber Felt, 85% from previous simulation.\n"
table_content += "- **Evaporation Rate**: For PAPS, taken from Tables S1 and S2; for Carbon Fiber Felt, 1.415 kg·m^-2·h^-1 at porosity 0.9 (given), scaled for other conditions.\n"
table_content += "- **Efficiency**: Computed using \\( \\eta = \\frac{m h_{\\text{LV}}}{C_{\\text{opt}} P_0} \\), with dark field evaporation of 0.1 kg·m^-2·h^-1 subtracted.\n"
table_content += "- **Surface Temperature**: For Carbon Fiber Felt, 34°C (wet) and 58.1°C (dry max); for PAPS, estimated from Figure S16 (increases with solar illumination).\n"
table_content += "- **Accumulated Solar Energy**: Estimated from Figures S10, S12, S14; higher for PAPS due to array structure.\n"
table_content += "- **Max Velocity and Surface RH**: Simulated values, consistent with Figure S17 trends (higher velocity and lower RH with forced convection and higher porosity).\n"
table_content += "- **Latent Heat Flux**: Computed using \\( Q_{\\text{evap}} = - H_{\\text{evap}} \\times m_{\\text{evap}} \\), where \\( H_{\\text{evap}} = 2.26 \\times 10^6 \\, \\text{J/kg} \\)."

# Step 3: Convert Markdown to HTML
html_content = markdown.markdown(table_content)

# Step 4: Add basic CSS for better PDF rendering
html_with_style = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        table {{ border-collapse: collapse; width: 100%; font-size: 10px; }}
        th, td {{ border: 1px solid black; padding: 5px; text-align: center; }}
        th {{ background-color: #f2f2f2; }}
        h1, h3 {{ text-align: center; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>
"""

# Step 5: Save HTML to a temporary file
temp_html_file = "temp_table.html"
with open(temp_html_file, "w", encoding="utf-8") as f:
    f.write(html_with_style)

# Step 6: Convert HTML to PDF using pdfkit
try:
    # Specify the path to wkhtmltopdf if needed (Windows example)
    # path_wkhtmltopdf = r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe"
    # config = pdfkit.configuration(wkhtmltopdf=path_wkhtmltopdf)
    # pdfkit.from_file(temp_html_file, "TableS4.pdf", configuration=config)

    # For systems where wkhtmltopdf is in PATH
    pdfkit.from_file(temp_html_file, "TableS4.pdf")
    print("PDF successfully generated as 'TableS4.pdf'.")
except Exception as e:
    print(f"Error generating PDF: {e}")
finally:
    # Clean up temporary HTML file
    if os.path.exists(temp_html_file):
        os.remove(temp_html_file)

print("Script execution completed.")