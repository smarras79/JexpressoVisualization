# Quick Start Guide: PVTU Time Averaging

This guide will get you started with time-averaging your PVTU files in 5 minutes.

## Step 1: Verify Your Files

Make sure you have PVTU files with numeric indices:
```bash
ls -1 *.pvtu
# Should show something like:
# iter_1.pvtu
# iter_2.pvtu
# iter_3.pvtu
# ...
```

Each PVTU file should have a corresponding directory with the parallel VTU files:
```bash
ls -d iter_*/
# Should show:
# iter_1/
# iter_2/
# iter_3/
# ...
```

## Step 2: Configure the Script

Open `pvtu_time_average_parallel.py` and edit the USER SETTINGS section:

```python
# Base directory containing PVTU files
base_directory = "./"

# File pattern for PVTU files
file_pattern = "iter_*.pvtu"

# Output file name
output_file = "time_averaged.pvtu"

# Optional: Limit the range
start_index = None  # Set to a number like 10 to start from iter_10.pvtu
end_index = None    # Set to a number like 100 to end at iter_100.pvtu
```

**That's it!** The script will automatically:
- Find all matching PVTU files
- Detect all data arrays
- Average everything except ID/ghost fields

## Step 3: Run the Script

### Option A: Quick Test (Serial)
```bash
pvpython pvtu_time_average_parallel.py
```

### Option B: Production Run (Parallel - Recommended)
```bash
# Use 8 MPI processes
mpiexec -n 8 pvpython pvtu_time_average_parallel.py

# Or use all available cores
mpiexec -n $(nproc) pvpython pvtu_time_average_parallel.py
```

### Option C: HPC Cluster (SLURM)
```bash
# Edit submit_time_average.sh if needed, then:
sbatch submit_time_average.sh

# Monitor the job
squeue -u $USER
tail -f time_average_*.out
```

## Step 4: Check the Output

After completion, you'll have:
```bash
time_averaged.pvtu           # Main file
time_averaged/               # Directory with parallel VTU files
  time_averaged_0.vtu
  time_averaged_1.vtu
  ...
```

All averaged fields will have `_avg` suffix:
- `U` → `U_avg`
- `p` → `p_avg`
- `T` → `T_avg`

## Visualize Results

Open the output in ParaView:
```bash
paraview time_averaged.pvtu
```

Or use Python:
```python
from paraview.simple import *
data = XMLPartitionedUnstructuredGridReader(FileName='time_averaged.pvtu')
Show(data)
```

## Common Scenarios

### Scenario 1: Average last 50 iterations only
```python
start_index = None  # Process until the end
end_index = None    # Start from beginning
```
Then manually move/rename the files you want to process.

Or set specific indices:
```python
start_index = 50
end_index = 100
```

### Scenario 2: Different file naming
If your files are named differently:
```python
file_pattern = "output_*.pvtu"  # For output_1.pvtu, output_2.pvtu, ...
file_pattern = "flow_*.pvtu"    # For flow_1.pvtu, flow_2.pvtu, ...
file_pattern = "result*.pvtu"   # For result1.pvtu, result2.pvtu, ...
```

### Scenario 3: Exclude specific arrays
If you have custom ID or flag fields:
```python
exclude_arrays = [
    'vtkGhostType',
    'GlobalNodeId',
    'GlobalElementId',
    'my_particle_id',      # Add your custom IDs
    'my_flag_field',       # Add your flags
]
```

## Troubleshooting

### "No files found"
- Check `base_directory` is correct
- Verify `file_pattern` matches your files
- Make sure files have numbers in their names

### "mpi4py not available"
- Script runs in serial mode automatically
- For parallel: install mpi4py in ParaView's Python

### Script is slow
- Use more MPI processes: `mpiexec -n 32 pvpython ...`
- Process fewer timesteps at once
- Use faster storage (local disk vs network)

### Memory error
- Use more MPI ranks to distribute data
- Process in smaller batches (set start_index/end_index)
- Use a machine with more RAM

## Next Steps

- Read `README_TIME_AVERAGING.md` for detailed documentation
- Check `example_time_average_config.py` for more configuration examples
- Modify `submit_time_average.sh` for your HPC system

## Quick Reference

```bash
# Test with first file only
pvpython pvtu_time_average_parallel.py  # After setting end_index = 1

# Run parallel on 16 cores
mpiexec -n 16 pvpython pvtu_time_average_parallel.py

# Submit to SLURM
sbatch submit_time_average.sh

# Check available arrays in a PVTU file
pvpython -c "
from paraview.simple import *
r = XMLPartitionedUnstructuredGridReader(FileName='iter_1.pvtu')
r.UpdatePipeline()
info = r.GetDataInformation()
print('Point arrays:', [info.GetPointDataInformation().GetArrayInformation(i).GetName()
                        for i in range(info.GetPointDataInformation().GetNumberOfArrays())])
print('Cell arrays:', [info.GetCellDataInformation().GetArrayInformation(i).GetName()
                       for i in range(info.GetCellDataInformation().GetNumberOfArrays())])
"
```

## Support Files

- `pvtu_time_average_parallel.py` - Main script
- `README_TIME_AVERAGING.md` - Full documentation
- `example_time_average_config.py` - Configuration examples
- `submit_time_average.sh` - SLURM submission script
- `QUICKSTART_TIME_AVERAGE.md` - This file

Happy averaging! 🚀
