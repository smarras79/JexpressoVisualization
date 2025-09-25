# JexpressoVisualization

 Simple Python-Paraview script to extract slices from 3D data in a series of pvtu files

 This is designed to handle output from Jexpresso although it may work out of the box for
 any pvtu file coming from elsewhere.

## How to use it

From single core:
```python3 batch_paraview_analysis.py```

From a cluster:
Edit ```submit_batch_paraview_analysis.sh``` to modify your system-related strings:

Replace the two following lines with what your specific ones:
```
module load bright shared mpich/ge/gcc/64 foss/2024a ParaView
cd /scratch/smarras/smarras/output/64x64x24/CompEuler/LESsmago/output/
```
then run it as a usual slurm batch script.
