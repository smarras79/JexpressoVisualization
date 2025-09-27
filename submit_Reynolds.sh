#!/bin/bash -l
#SBATCH --job-name=Reynolds_Spectra_Analysis
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=general         # Verify this is the correct partition for your cluster
#SBATCH --qos=standard              # Verify this is the correct QoS
#SBATCH --account=smarras           # Verify your account name
#SBATCH --nodes=1                   # The Python script is serial, so it only needs one node
#SBATCH --ntasks-per-node=1         # The script is a single process, so only one task is needed
#SBATCH --cpus-per-task=10          # Request 10 CPU cores for this single task
#SBATCH --time=23:59:00             # Max walltime
#SBATCH --mem=40G                   # Request total memory for the job (e.g., 4GB/core * 10 cores)

#======================================================================
#  ENVIRONMENT SETUP
#======================================================================
echo "================================================================"
echo "Job starting at: $(date)"
echo "Job running on: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "CPUs requested: $SLURM_CPUS_PER_TASK"
echo "Memory requested: ${SLURM_MEM_PER_NODE}M"
echo "================================================================"

# Purge any existing modules to ensure a clean environment
module purge

# Load necessary modules. ParaView is often required for PyVista's backend
# rendering libraries (like OSMesa) to work on a headless cluster node.
# Use `module avail ParaView` or `module spider ParaView` on your cluster to find the exact name.
module load foss/2024a ParaView  # This is an example, adjust to your cluster's available modules

echo "Modules loaded."

# Activate your Python virtual environment
# Ensure this path is correct for your user on the cluster's filesystem.
VENV_PATH="/mmfs1/project/smarras/smarras/JexpressoVisualization/venv_pyvista/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "Python virtual environment activated."
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

#======================================================================
#  EXECUTION
#======================================================================
echo "Starting Python analysis script..."

# Execute the Python script.
# IMPORTANT: Ensure the DATA_DIR path and other configurations inside
# the Python script are correct for the cluster's filesystem.
python3 Reynolds_triple_spectra.py

echo "Python script finished."

#======================================================================
#  CLEANUP
#======================================================================
# Deactivate the virtual environment
deactivate
echo "Virtual environment deactivated."
echo "Job finished at: $(date)"
echo "================================================================"
