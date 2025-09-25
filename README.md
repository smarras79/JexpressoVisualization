# JexpressoVisualization

 Simple Python-Paraview script to extract slices from 3D data in a series of pvtu files

 This is designed to handle output from Jexpresso although it may work out of the box for
 any pvtu file coming from elsewhere.

## How to use it

**From single core:**

- Edit ```batch_paraview_analysis.py```:
- Set the correct path to ```pvpython``` in 

```'paraview_executable': '/Applications/ParaView-5.11.2.app/Contents/bin/pvpython',```

- Launch it:
```python3 batch_paraview_analysis.py```

**From a cluster:**
Edit ```submit_batch_paraview_analysis.sh``` to modify your system-related strings:

Replace the two following lines with what your specific ones:
```
module load bright shared mpich/ge/gcc/64 foss/2024a ParaView
cd /scratch/smarras/smarras/output/64x64x24/CompEuler/LESsmago/output/
```
then run it as a usual slurm batch script.
