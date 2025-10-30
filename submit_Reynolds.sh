#!/bin/bash -l
#SBATCH --job-name=Reynolds_Spectra_Analysis
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=smarras
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=23:59:00
#SBATCH --mem=40G

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

# CRITICAL: Change to working directory FIRST
WORK_DIR="/mmfs1/project/smarras/smarras/JexpressoVisualization"
cd "$WORK_DIR" || {
    echo "ERROR: Cannot change to directory $WORK_DIR"
    exit 1
}
echo "Working directory: $(pwd)"

# Check for numpy naming conflicts
echo "Checking for numpy naming conflicts..."
if [ -f "numpy.py" ]; then
    echo "ERROR: Found numpy.py in current directory!"
    echo "Please rename it: mv numpy.py my_numpy_utils.py"
    exit 1
fi
if [ -d "numpy" ]; then
    echo "ERROR: Found numpy/ directory in current directory!"
    echo "Please rename it: mv numpy/ numpy_backup/"
    exit 1
fi
echo "✓ No naming conflicts found"

# Load ParaView module ONLY (needed for rendering libraries)
# DO NOT load foss or Python modules - the venv provides Python packages
module load foss/2024a ParaView
echo "✓ ParaView module loaded"

# CRITICAL: Clear Python-related environment variables
# This prevents conflicts between system modules and venv
unset PYTHONPATH
unset PYTHONHOME
unset PYTHON_ROOT
echo "✓ Python environment variables cleared"

# Activate virtual environment
VENV_PATH="/mmfs1/project/smarras/smarras/JexpressoVisualization/venv_pyvista/bin/activate"
if [ -f "$VENV_PATH" ]; then
    source "$VENV_PATH"
    echo "✓ Python virtual environment activated"
    echo "  Python: $(which python3)"
    echo "  Version: $(python3 --version)"
else
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
fi

# IMPORTANT: Test numpy import before running main script
echo ""
echo "Testing critical imports..."
python3 << 'PYTEST'
import sys
print(f"Python executable: {sys.executable}")
print(f"sys.path (first 3 entries):")
for i, p in enumerate(sys.path[:3]):
    print(f"  {i}: {p}")

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__} - {np.__file__}")
except Exception as e:
    print(f"✗ NumPy import FAILED: {e}")
    sys.exit(1)

try:
    import pyvista as pv
    print(f"✓ PyVista {pv.__version__}")
except Exception as e:
    print(f"⚠ PyVista import failed: {e}")

try:
    import xarray as xr
    print(f"✓ xarray {xr.__version__}")
except Exception as e:
    print(f"⚠ xarray import failed: {e}")

try:
    import scipy
    print(f"✓ scipy {scipy.__version__}")
except Exception as e:
    print(f"⚠ scipy import failed: {e}")
PYTEST

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Import test failed. Cannot proceed."
    exit 1
fi

#======================================================================
#  EXECUTION
#======================================================================
echo ""
echo "================================================================"
echo "Starting Python analysis script..."
echo "================================================================"

python3 Reynolds_triple_spectra.py

EXIT_CODE=$?

#======================================================================
#  CLEANUP
#======================================================================
echo ""
echo "================================================================"
deactivate
echo "✓ Virtual environment deactivated"
echo "Job finished at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "================================================================"

exit $EXIT_CODE
