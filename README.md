# Droplet Evaporation Visualization

This project provides a Python-based visualization tool to simulate and display the evaporation effects of a liquid droplet. It uses synthetic data to mimic physical phenomena such as vapor diffusion flux, velocity fields, surface evaporation rate, and relative humidity, as studied in COMSOL simulations. The visualizations are generated using matplotlib and numpy, with a focus on making the evaporation effect clear through color gradients and vector fields.
Features

Visualizes 12 different aspects of droplet evaporation, including vapor flux, velocity fields, evaporation rate, humidity, temperature, pressure, and concentration gradient.
Supports both flat and spiral droplet structures.
Shows the evolution of the evaporation process over time.
Uses a 3x4 subplot grid for comprehensive visualization.

# Requirements

Python 3.6+
numpy (for numerical computations)
matplotlib (for plotting)

# Installation

Clone the repository


Navigate to the project directory:cd droplet-evaporation-visualization


Install the required dependencies:pip install numpy matplotlib



# Usage

Ensure you have the required dependencies installed.
Run the main script to generate the visualization:python droplet_visualization.py


The script will generate a plot saved as droplet_evaporation_visualization_12_tables.png in the project directory.

# File Structure

droplet_visualization.py: Main script containing the visualization code.
README.md: Project documentation (this file).
droplet_evaporation_visualization_12_tables.png: Output visualization (generated after running the script).

# Example Output
The script generates a 3x4 grid of subplots showing:

Vapor diffusion flux (flat and spiral structures)
Velocity fields (flat and spiral structures)
Surface evaporation rate (flat, spiral, and at a later time)
Relative humidity (flat and spiral structures)
Temperature distribution
Pressure field
Concentration gradient

The evaporation effect is highlighted through color gradients and vector fields, with droplet boundaries marked for clarity.
# Contributing
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure your code follows the existing style and includes appropriate comments.
# License
This project is licensed under the MIT License. See the LICENSE file for details.
# Acknowledgments
Inspired by COMSOL simulation studies on droplet evaporation. Thanks to the open-source community for providing tools like matplotlib and numpy.
