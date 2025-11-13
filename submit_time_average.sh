#!/bin/bash
#SBATCH --job-name=pvtu_time_avg
#SBATCH --nodes=2
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=16
#SBATCH --time=02:00:00
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=smarras
#SBATCH --output=time_average_%j.out
#SBATCH --error=time_average_%j.err

#=============================================================================
# SLURM submission script for PVTU time averaging
#
# This script runs the pvtu_time_average_parallel.py script in parallel
# using MPI and ParaView's pvpython.
#
# Usage:
#   1. Edit the USER SETTINGS section below
#   2. Edit the configuration in pvtu_time_average_parallel.py
#   3. Submit: sbatch submit_time_average.sh
#
# Monitor job:
#   squeue -u $USER
#   tail -f time_average_<jobid>.out
#=============================================================================

#-----------------------------------------------------------------------------
# USER SETTINGS
#-----------------------------------------------------------------------------

# ParaView module (adjust for your system)
PARAVIEW_MODULE="ParaView/5.11.0"

# MPI module (adjust for your system)
MPI_MODULE="mpich/ge/gcc/64"

# Python script to run
SCRIPT="pvtu_time_average_parallel.py"

# Working directory (where the script and data are located)
WORK_DIR="${SLURM_SUBMIT_DIR}"

#-----------------------------------------------------------------------------
# END USER SETTINGS
#-----------------------------------------------------------------------------

echo "========================================================================"
echo "PVTU Time Averaging Job"
echo "========================================================================"
echo "Job ID:           ${SLURM_JOB_ID}"
echo "Job Name:         ${SLURM_JOB_NAME}"
echo "Nodes:            ${SLURM_JOB_NUM_NODES}"
echo "Tasks:            ${SLURM_NTASKS}"
echo "Tasks per node:   ${SLURM_NTASKS_PER_NODE}"
echo "Working dir:      ${WORK_DIR}"
echo "Submit dir:       ${SLURM_SUBMIT_DIR}"
echo "Script:           ${SCRIPT}"
echo "Start time:       $(date)"
echo "========================================================================"
echo ""

# Change to working directory
cd ${WORK_DIR}

# Load required modules
echo "Loading modules..."
module purge

# Try to load ParaView module
if module load ${PARAVIEW_MODULE} 2>/dev/null; then
    echo "  Loaded: ${PARAVIEW_MODULE}"
else
    echo "  Warning: Could not load ${PARAVIEW_MODULE}"
    echo "  Attempting to use system ParaView..."
fi

# Try to load MPI module
if module load ${MPI_MODULE} 2>/dev/null; then
    echo "  Loaded: ${MPI_MODULE}"
else
    echo "  Warning: Could not load ${MPI_MODULE}"
    echo "  Using default MPI..."
fi

echo ""
echo "Loaded modules:"
module list
echo ""

# Check if pvpython is available
if command -v pvpython &> /dev/null; then
    echo "pvpython found: $(which pvpython)"
    pvpython --version
else
    echo "ERROR: pvpython not found in PATH"
    echo "PATH: ${PATH}"
    exit 1
fi
echo ""

# Check if script exists
if [ ! -f "${SCRIPT}" ]; then
    echo "ERROR: Script not found: ${SCRIPT}"
    echo "Contents of working directory:"
    ls -la
    exit 1
fi
echo ""

# Print system information
echo "System information:"
echo "  Hostname:       $(hostname)"
echo "  CPU info:       $(lscpu | grep 'Model name' | cut -d: -f2 | xargs)"
echo "  Total memory:   $(free -h | grep Mem | awk '{print $2}')"
echo ""

# Run the time averaging script
echo "========================================================================"
echo "Starting time averaging..."
echo "Command: srun -n ${SLURM_NTASKS} pvpython ${SCRIPT}"
echo "========================================================================"
echo ""

# Record start time
START_TIME=$(date +%s)

# Execute with srun (SLURM's MPI launcher)
srun -n ${SLURM_NTASKS} pvpython ${SCRIPT}

# Capture exit status
EXIT_STATUS=$?

# Record end time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo ""
echo "========================================================================"
echo "Job completed"
echo "========================================================================"
echo "Exit status:      ${EXIT_STATUS}"
echo "End time:         $(date)"
echo "Elapsed time:     ${ELAPSED} seconds ($(($ELAPSED / 60)) minutes)"
echo "========================================================================"

# Exit with the script's exit status
exit ${EXIT_STATUS}
