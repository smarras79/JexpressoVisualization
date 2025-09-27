#!/bin/bash -l
#SBATCH --job-name=ParaView_Simple
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --partition=general
#SBATCH --qos=standard
#SBATCH --account=smarras
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10  # Adjust based on your needs
#SBATCH --time=23:59:00
#SBATCH --mem-per-cpu=4000M

module purge
module spider ParaView shared mpich/ge/gcc/64
#module load bright shared mpich/ge/gcc/64 ParaView
#module load bright shared mpich/ge/gcc/64 foss/2024a ParaView

echo "Starting simple parallel ParaView processing..."
echo "Job ID: $SLURM_JOB_ID, CPUs: $SLURM_NTASKS_PER_NODE"

source /mmfs1/project/smarras/smarras/JexpressoVisualization/venv_pyvista/bin/activate
pip3 install numpy pyvista tqdm xarray scipy

#=============================================
# CUSTOMIZE THESE RANGES BASED ON YOUR FILES:
# First, run this to see what files you have:
# python3 batch_paraview_analysis.py --dry-run
#=============================================
# Then split the work across processes:
python3 Reynold_triple_spectra.py


# Wait for all processes to complete
wait

echo "All processes completed! Check batch_output/ for results."

deactivate
module purge
