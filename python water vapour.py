import subprocess
import os

# Define paths
comsol_path = "comsolbatch"  # Assumes comsolbatch is in your system Path
model_path = r"C:\Users\BRANDON\Desktop\Simulation\Simulation\water_vapor_flow.mph"
output_path = r"C:\Users\BRANDON\Desktop\Simulation\Simulation\water_vapor_flow_updated.mph"

# Check if the model file exists
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found. Please create it in COMSOL first.")
    

# Run COMSOL in batch mode
try:
    subprocess.run([
        comsol_path,
        "-inputfile", model_path,
        "-outputfile", output_path,
        "-batchlog", "comsol_batch.log"
    ], check=True)
    print("Simulation completed successfully! Results saved to:", output_path)
except subprocess.CalledProcessError as e:
    print("Error running COMSOL batch mode:", e)
    print("Check comsol_batch.log for more details.")
except FileNotFoundError:
    print("Error: COMSOL batch executable not found. Ensure COMSOL is installed and added to your system Path.")